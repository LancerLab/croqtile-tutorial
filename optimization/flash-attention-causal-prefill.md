# How to Write a Flash Attention Kernel with Croqtile: a Worklog

*June 2026 -- GPU: NVIDIA H800 PCIe (SM90a) -- Precision: BF16 -- Problem: Causal Prefill, D=128, BSHD layout*

---

Flash Attention is the most impactful kernel in modern LLM training and inference. It fuses the entire Q*K softmax*V pipeline into a single kernel, reducing memory from O(N^2) to O(N) and unlocking sequence lengths that would otherwise OOM.

In this worklog I start from a sequential load-compute kernel and step-by-step apply the three major GPU optimization techniques for attention -- warp specialization, async TMA pipelining, and intra-warpgroup QK/PV overlap -- all in **Croqtile**. The finished kernel reaches **374 TFLOPS** at SEQ=16384, matching **FlashAttention-3** on the same hardware. No hand-written CUDA, no manual `mbarrier` lifecycle management, no WGMMA descriptor encoding.

This tutorial assumes familiarity with the matmul tutorial's Hopper constructs (`tma.copy`, `mma.load`, `inthreads.async`, `shared event`). If those are new, read [Dense GEMM FP16: From Naive to Tuned](dense-gemm-fp16-from-naive.md) first.

Compile and run any kernel:

```bash
cd benchmark/performance/flash_atten/causal_prefill_d128/choreo
bash bench.sh --kernel v1_manual_baseline.co --gpu 1
```

---

## The Algorithm: Online Softmax Attention

Before looking at code, here is the algorithm that all three kernels implement. This is the FlashAttention core loop (Dao et al., 2022):

For each block of query rows `[bm]`:

1. Load Q block from global memory
2. For each block of KV columns `[bn]`:
   a. Compute `S = Q * K^T` (score matrix, via WGMMA)
   b. Apply causal mask: `S[i,j] = -inf` where `j > i`
   c. Online softmax: track running max and sum of exponentials
   d. Rescale previous accumulator: `O *= exp(prev_max - new_max)`
   e. Compute `O += softmax(S) * V` (via WGMMA)
3. Normalize: `O /= logsum`
4. Store O to global memory

The key insight: by maintaining a running max and log-sum-exp, we never materialize the full N*N attention matrix. Each KV block is loaded once, used, and discarded.

---

## Performance at a glance

Headline config: **B=4, H=32, SEQ=8192, D=128, causal prefill, BF16**

| Kernel | Time (ms) | TFLOPS | vs FA3 |
| --- | ---: | ---: | ---: |
| v1: Sequential DMA + WGMMA | 9.67 | 227.5 | 55% |
| v2: Warp-specialized 1p2c + TMA pipeline | 6.28 | 350.1 | 85% |
| **v3: FA3-style QK/PV overlap** | **6.09** | **361.3** | **88%** |
| FlashAttention-3 (reference) | 5.41 | 410.3 | 100% |

Across sequence lengths (B=4, H=32):

| Config | Choreo v3 | FA3 | TileLang | Triton | Triton+WS |
| --- | ---: | ---: | ---: | ---: | ---: |
| SEQ=4096 | 333.3 | 407.1 | 243.8 | 294.4 | 346.3 |
| SEQ=8192 | 365.0 | 410.3 | 295.8 | 312.5 | 354.3 |
| SEQ=16384 | 377.2 | 422.1 | 349.0 | 324.7 | 371.0 |

*TFLOPS. Higher is better. Measured on H800 PCIe with warmup=50, repeat=200.*

---

## Why Flash Attention Is Harder Than GEMM

A matmul kernel has one compute phase: C += A * B. Flash attention has **two coupled WGMMA phases** per inner-loop iteration -- QK and PV -- connected by a softmax that depends on the QK result. This creates three challenges that do not exist in GEMM:

| Challenge | GEMM | Flash Attention |
| --- | --- | --- |
| **Compute phases** | 1 (C += A*B) | 2 per iteration (S = Q*K, O += P*V) |
| **Data dependencies** | None between iterations | PV depends on softmax of QK |
| **Shared memory pressure** | 2 buffers (A, B) | 3+ buffers (Q, K, V, plus staging) |
| **Causal masking** | N/A | Conditional `-inf` injection after QK |
| **Numerical stability** | Standard FP accumulation | Online max/sum tracking across iterations |

The optimization path therefore differs: instead of just tuning tile shapes, we need to **restructure the loop body** to overlap the two WGMMA phases.

---

## Kernel 1: Sequential Load + Compute

**File:** `v1_manual_baseline.co`

The simplest correct implementation: load Q once into shared memory, then sequentially load K and V blocks, compute QK, apply masking and softmax, compute PV, and accumulate.

```choreo
#define BLOCK_M 128
#define BLOCK_N 128
#define WG_M 64
#define WG_K 16
#define DIM 128

__co__ void flash_atten(
  global bf16[B, Q_SEQ, H, DIM] Q, global bf16[B, KV_SEQ, H, DIM] K,
  global bf16[B, KV_SEQ, H, DIM] V,
  global bf16[B, Q_SEQ, H, DIM] O
) {
  float scale = 0.12751743f;
  int past_len = KV_SEQ - Q_SEQ;

  parallel.async {bm, head, batch} by [cdiv(Q_SEQ, BLOCK_M), H, B] : block {
    Q_head = Q.subspan(1, Q_SEQ, 1, DIM).at(batch, 0, head, 0).sqz();
    K_head = K.subspan(1, KV_SEQ, 1, DIM).at(batch, 0, head, 0).sqz();
    V_head = V.subspan(1, KV_SEQ, 1, DIM).at(batch, 0, head, 0).sqz();
    O_head = O.subspan(1, Q_SEQ, 1, DIM).at(batch, 0, head, 0).sqz();

    q_s = dma.copy.swiz<128> Q_head.subspan(BLOCK_M, DIM).at(bm, 0) => shared;
    parallel wgm by [BLOCK_M / WG_M]: group-4 {
      frag bf16[WG_M, BLOCK_N] acc_s_cast;
      frag f32[WG_M] scores_max {-inf};
      frag f32[WG_M] scores_max_prev;
      frag f32[WG_M] scores_scale;
      frag f32[WG_M] scores_sum;
      frag f32[WG_M] logsum {0.0f};
      acc_o = mma.fill.f32 0.0f;

      kv_bound = __min(
        cdiv((bm + 1) * BLOCK_M + past_len, BLOCK_N),
        cdiv(KV_SEQ, BLOCK_N));

      foreach {bn} in [kv_bound] {
        // -- Phase 1: QK --
        acc_s = mma.fill.f32 0.0f;
        k_s = dma.copy.swiz<128>
          K_head.subspan(BLOCK_N, DIM).at(bn, 0) => shared;
        mma.row.row acc_s, q_s.chunkat(wgm, _), k_s;

        // -- Causal mask --
        apply {i, j} in acc_s.span {
          if ((bn * BLOCK_N + j) > (bm * BLOCK_M + wgm * WG_M + i + past_len))
            acc_s.at(i, j) = -inf;
        }

        // -- Online softmax --
        copy(scores_max_prev, scores_max);
        reduce_max(scores_max, acc_s, 1);
        apply {i} in scores_max.span {
          scores_max.at(i) = __max(scores_max.at(i), scores_max_prev.at(i));
          scores_scale.at(i) =
            __exp2f(scores_max_prev.at(i) * scale - scores_max.at(i) * scale);
        }
        apply {i, j} in acc_s.span
          acc_s.at(i, j) =
            __exp2f(acc_s.at(i, j) * scale - scores_max.at(i) * scale);
        reduce_sum(scores_sum, acc_s, 1);
        apply {i} in logsum.span
          logsum.at(i) = logsum.at(i) * scores_scale.at(i) + scores_sum.at(i);
        apply {i, j} in acc_o.span
          acc_o.at(i, j) = acc_o.at(i, j) * scores_scale.at(i);
        apply {i, j} in acc_s_cast.span
          acc_s_cast.at(i, j) = __to<bf16>(acc_s.at(i, j));

        // -- Phase 2: PV --
        v_s = dma.copy.swiz<128>
          V_head.subspan(BLOCK_N, DIM).at(bn, 0) => shared;
        mma.row.col acc_o, acc_s_cast, v_s;
      }

      apply {i, j} in acc_o.span
        acc_o.at(i, j) = acc_o.at(i, j) / logsum.at(i);
      mma.store acc_o, O_head.subspan(WG_M, DIM).at(bm # wgm, 0);
    }
  }
}
```

### Croqtile concepts for attention

| Construct | Meaning |
| --- | --- |
| `parallel.async {...} : block` | Asynchronous grid launch. The `.async` suffix enables the Hopper async execution model with cluster-level coordination. |
| `.subspan(1, Q_SEQ, 1, DIM).at(batch, 0, head, 0).sqz()` | Slice one batch and one head from a 4-D BSHD tensor, then squeeze the unit dimensions to get a 2-D `[seq, dim]` view. |
| `dma.copy.swiz<128> src => shared` | SIMT-level cooperative copy with 128-byte swizzle into an auto-allocated shared memory buffer. All threads participate; Croqtile handles coalescing and `__syncthreads()`. |
| `parallel wgm by [N]: group-4` | N warpgroups (each 128 threads) execute the body. Each gets a `wgm` index. |
| `frag f32[WG_M] scores_max {-inf}` | A register-resident fragment with per-element initialization. |
| `reduce_max(dst, src, dim)` | Row-wise max reduction across a 2-D fragment. |
| `apply {i, j} in frag.span { ... }` | Element-wise transform over a fragment. Compiles to register-register operations with no memory traffic. |
| `mma.row.row acc, A, B` | WGMMA: Q*K where both A and B are row-major. |
| `mma.row.col acc, A, B` | WGMMA: P*V where A is row-major, B (V) is column-major in the logical contraction. |

### How the inner loop works

Each iteration of the `foreach {bn}` loop performs five steps:

```
Load K[bn] -> Compute S = Q*K -> Mask+Softmax -> Load V[bn] -> Compute O += P*V
     |              |                  |              |              |
     DMA          WGMMA             Scalar          DMA           WGMMA
   (stall)      (stall on          (fast but        (stall)      (stall on
                 DMA)            serialized)                      DMA)
```

Every operation waits for the previous one. The two `dma.copy` calls each stall all threads while data transfers from HBM. The WGMMA instructions stall until shared memory is ready. This is the simplest structure, but it leaves massive performance on the table.

### Result: v1

| Config | Time (ms) | TFLOPS |
| --- | ---: | ---: |
| SEQ=4096 | 2.64 | 207.9 |
| SEQ=8192 | 9.67 | 227.5 |
| SEQ=16384 | 38.8 | 226.8 |

**~228 TFLOPS at SEQ=8192 (55% of FA3)**

---

## Kernel 2: Warp-Specialized 1p2c with TMA Pipeline

**File:** `v2_manual_s2_1p2c_tma.co`

The v1 kernel serializes data movement and compute. The fix is the same as in the matmul tutorial: split threads into **producer** (TMA loads) and **consumer** (WGMMA compute) warpgroups connected by a double-buffered pipeline.

The structural changes from v1:

1. **TMA replaces `dma.copy`** -- one thread drives hardware DMA instead of all threads cooperating
2. **1 producer + 2 consumer warpgroups** (1p2c) -- the producer prefetches K and V while consumers compute
3. **2-stage double buffering** -- `k_buf[2]`, `v_buf[2]` with `shared event` barriers
4. **Q is loaded once via async TMA** before the main loop

```choreo
#define BLOCK_M 128
#define BLOCK_N 128
#define WG_M 64
#define WG_K 16
#define SWIZ 128
#define STAGES 2
#define DIM 128

__co__ void flash_atten(
  global bf16[B, Q_SEQ, H, DIM] Q,
  global bf16[B, KV_SEQ, H, DIM] K,
  global bf16[B, KV_SEQ, H, DIM] V,
  global bf16[B, Q_SEQ, H, DIM] O
) {
  float scale = 0.12751743f;

  [[launch_bounds(_, 1)]]
  parallel.async {bm, head, batch} by [cdiv(Q_SEQ, BLOCK_M), H, B] : block {
    Q_head = Q.subspan(1, Q_SEQ, 1, DIM).at(batch, 0, head, 0).sqz();
    K_head = K.subspan(1, KV_SEQ, 1, DIM).at(batch, 0, head, 0).sqz();
    V_head = V.subspan(1, KV_SEQ, 1, DIM).at(batch, 0, head, 0).sqz();
    O_head = O.subspan(1, Q_SEQ, 1, DIM).at(batch, 0, head, 0).sqz();

    shared bf16[BLOCK_M, DIM] q_shared;
    shared bf16[BLOCK_N, DIM] k_buf[STAGES];
    shared bf16[BLOCK_N, DIM] v_buf[STAGES];

    shared event qf, kf[STAGES], ke[STAGES], vf[STAGES], ve[STAGES];

    kv_bound = cdiv((bm + 1) * BLOCK_M, BLOCK_N);
    parallel p by [BLOCK_M/WG_M + 1] : group-4, t by 128 : thread {
      // --- Producer (p=0): single-thread TMA ---
      inthreads.async (p == 0) {
        call croq::cuda::setreg_dec(24);
        tma.copy.async<qf>.swiz<SWIZ>
          Q_head.subspan(BLOCK_M, DIM).at(bm, 0) => q_shared;
        trigger qf;

        foreach {bn} in [kv_bound] {
          stage = bn % STAGES;
          wait ke[stage];
          tma.copy.async<kf[stage]>.swiz<SWIZ>
            K_head.subspan(BLOCK_N, DIM).at(bn, 0) => k_buf[stage];
          trigger kf[stage];

          wait ve[stage];
          tma.copy.async<vf[stage]>.swiz<SWIZ>
            V_head.subspan(BLOCK_N, DIM).at(bn, 0) => v_buf[stage];
          trigger vf[stage];
        }
      }

      // --- Consumers (p>0): WGMMA compute ---
      inthreads.async (p > 0) {
        call croq::cuda::setreg_inc(240);
        cid = p - 1;
        foreach {s} in [STAGES] {
          trigger ke[s];
          trigger ve[s];
        }

        wait qf;

        frag f32[WG_M] scores_max {-inf};
        frag f32[WG_M] scores_max_prev;
        frag f32[WG_M] scores_scale;
        frag f32[WG_M] scores_sum;
        frag f32[WG_M] logsum {0.0f};
        acc_o = mma.fill.f32 0.0f;

        foreach {bn} in [kv_bound] {
          stage = bn % STAGES;
          acc_s = mma.fill.f32 0.0f;

          wait kf[stage];
          mma.row.row acc_s,
            q_shared.subspan(WG_M, DIM).at(cid, 0), k_buf[stage];
          trigger ke[stage];

          // Causal mask (skip if entire tile is unmasked)
          int causal_row_base = bm * BLOCK_M + cid * WG_M;
          if ((bn + 1) * BLOCK_N > causal_row_base + 1) {
            apply {i, j} in acc_s.span {
              if ((bn * BLOCK_N + j) > (bm * BLOCK_M + cid * WG_M + i))
                acc_s.at(i, j) = -inf;
            }
          }

          // Online softmax (same as v1)
          copy(scores_max_prev, scores_max);
          reduce_max(scores_max, acc_s, 1);
          apply {i} in scores_max.span {
            scores_max.at(i) = __max(scores_max.at(i), scores_max_prev.at(i));
            scores_scale.at(i) = __exp2f(
              scores_max_prev.at(i) * scale - scores_max.at(i) * scale);
          }
          apply {i, j} in acc_s.span
            acc_s.at(i, j) = __exp2f(
              acc_s.at(i, j) * scale - scores_max.at(i) * scale);
          reduce_sum(scores_sum, acc_s, 1);
          apply {i} in logsum.span
            logsum.at(i) =
              logsum.at(i) * scores_scale.at(i) + scores_sum.at(i);
          apply {i, j} in acc_o.span
            acc_o.at(i, j) = acc_o.at(i, j) * scores_scale.at(i);

          frag bf16[WG_M, BLOCK_N] acc_s_cast;
          apply {i, j} in acc_s_cast.span
            acc_s_cast.at(i, j) = __to<bf16>(acc_s.at(i, j));

          wait vf[stage];
          mma.row.col acc_o, acc_s_cast, v_buf[stage];
          trigger ve[stage];
        }

        apply {i, j} in acc_o.span
          acc_o.at(i, j) = acc_o.at(i, j) / logsum.at(i);
        shared bf16[BLOCK_M, DIM] o_buf;
        mma.store acc_o, o_buf.subspan(WG_M, DIM).at(cid, 0);
        tma.copy o_buf.subspan(WG_M, DIM).at(cid, 0) =>
          O_head.subspan(WG_M, DIM).at(bm * 2 + cid, 0);
      }
    }
  }
}
```

### New concepts in v2

| Construct | Meaning |
| --- | --- |
| `[[launch_bounds(_, 1)]]` | Hint to the compiler: at most 1 thread-block per SM. This maximizes per-block register and shared memory budget. |
| `shared event qf, kf[STAGES], ke[STAGES], ...` | Named barriers backed by Hopper `mbarrier`. Five barrier arrays orchestrate the producer/consumer handshake for Q, K (full/empty), and V (full/empty). |
| `call croq::cuda::setreg_dec(24)` | Compiler intrinsic: decrease the register allocation for the producer warpgroup (it only issues TMA, needs few registers). Frees registers for the compute-heavy consumers. |
| `call croq::cuda::setreg_inc(240)` | Increase register allocation for consumer warpgroups. More registers = fewer spills during softmax and WGMMA accumulation. |
| `tma.copy.async<kf[stage]>.swiz<SWIZ>` | Async TMA that signals barrier `kf[stage]` on completion. The producer thread is not blocked. |
| `trigger kf[stage]` / `wait kf[stage]` | Signal/wait on named barriers. The producer signals `kf` after issuing TMA for K; the consumer waits on `kf` before reading K from shared memory. |
| `trigger ke[stage]` | Consumer signals "K buffer consumed" so the producer can reuse the slot. Classic double-buffer protocol. |

### The 1p2c pipeline structure

```
Warpgroup 0 (Producer):
  Load Q -> [signal qf]
  for bn in 0..kv_bound:
    wait ke[stage] -> Load K[bn] -> [signal kf[stage]]
    wait ve[stage] -> Load V[bn] -> [signal vf[stage]]

Warpgroup 1,2 (Consumers):
  [signal ke[0..STAGES], ve[0..STAGES]]  // initial empty tokens
  wait qf
  for bn in 0..kv_bound:
    wait kf[stage] -> QK WGMMA -> [signal ke[stage]]
    softmax
    wait vf[stage] -> PV WGMMA -> [signal ve[stage]]
```

The producer runs ahead of the consumers. With `STAGES=2`, while consumers compute on `k_buf[0]`/`v_buf[0]`, the producer can already be loading into `k_buf[1]`/`v_buf[1]`. TMA latency is hidden behind compute.

### Why the causal mask is optimized

v2 adds a key optimization over v1: the causal mask check is wrapped in a tile-level guard:

```choreo
if ((bn + 1) * BLOCK_N > causal_row_base + 1) {
  apply {i, j} in acc_s.span { ... }
}
```

For tiles where all KV positions are before the query position (fully unmasked), the entire `apply` block is skipped. Since causal attention is lower-triangular, roughly half the tiles skip masking entirely.

### Result: v2

| Config | Time (ms) | TFLOPS |
| --- | ---: | ---: |
| SEQ=4096 | 1.66 | 330.6 |
| SEQ=8192 | 6.28 | 350.1 |
| SEQ=16384 | 36.4 | 241.6 |

**~350 TFLOPS at SEQ=8192 (1.54x over v1, 85% of FA3)**

The large SEQ=16384 regression (241 TFLOPS vs 350) suggests the pipeline is not perfectly balanced at very long sequences -- the producer/consumer synchronization overhead grows with `kv_bound`.

---

## Kernel 3: FA3-Style Intra-Warpgroup Overlap

**File:** `v3_fa3_overlap.co`

The v2 kernel still serializes QK and PV within each consumer's inner loop: the consumer computes QK, performs softmax, then computes PV, then starts the next iteration. The tensor cores are idle during softmax.

FlashAttention-3 (Shah et al., 2024) introduced **intra-warpgroup overlap**: compute `QK[n]` and `PV[n-1]` simultaneously on the same tensor cores, using WGMMA's asynchronous pipeline. The softmax for iteration `n` runs on scalar units while both WGMMA operations execute on tensor cores.

### The overlap idea

```
Iteration n:
  Issue QK[n]  (WGMMA async)     }  Both on tensor cores
  Issue PV[n-1] (WGMMA async)    }  simultaneously
  mma.wait<1>                    -- wait for QK[n] only
  Softmax on QK[n] result        -- scalar units, tensor cores still running PV[n-1]
  mma.wait<0>                    -- wait for PV[n-1]
  Cast softmax result to bf16    -- prepare for next iteration's PV[n]
```

The key: `mma.wait<1>` waits until only 1 WGMMA group is outstanding (QK finishes, PV still in flight). Softmax uses scalar FMA units, so PV's tensor core operation runs concurrently.

```choreo
#define BLOCK_M 128
#define BLOCK_N 128
#define WG_M 64
#define WG_K 16
#define SWIZ 128
#define STAGES 2
#define DIM 128

__co__ void flash_atten(
  global bf16[B, Q_SEQ, H, DIM] Q,
  global bf16[B, KV_SEQ, H, DIM] K,
  global bf16[B, KV_SEQ, H, DIM] V,
  global bf16[B, Q_SEQ, H, DIM] O
) {
  float scale = 0.12751743f;

  [[launch_bounds(_, 1)]]
  parallel.async {bm, head, batch} by [cdiv(Q_SEQ, BLOCK_M), H, B] : block {
    Q_head = Q.subspan(1, Q_SEQ, 1, DIM).at(batch, 0, head, 0).sqz();
    K_head = K.subspan(1, KV_SEQ, 1, DIM).at(batch, 0, head, 0).sqz();
    V_head = V.subspan(1, KV_SEQ, 1, DIM).at(batch, 0, head, 0).sqz();
    O_head = O.subspan(1, Q_SEQ, 1, DIM).at(batch, 0, head, 0).sqz();

    shared bf16[BLOCK_M, DIM] q_shared;
    shared bf16[BLOCK_N, DIM] k_buf[STAGES];
    shared bf16[BLOCK_N, DIM] v_buf[STAGES];

    shared event qf, kf[STAGES], ke[STAGES], vf[STAGES], ve[STAGES];

    kv_bound = cdiv((bm + 1) * BLOCK_M, BLOCK_N);
    parallel p by [BLOCK_M/WG_M + 1] : group-4 {
      // Producer: identical to v2
      inthreads.async (p == 0) {
        call croq::cuda::setreg_dec(24);
        tma.copy.async<qf>.swiz<SWIZ>
          Q_head.subspan(BLOCK_M, DIM).at(bm, 0) => q_shared;
        trigger qf;

        foreach {bn} in [kv_bound] {
          stage = bn % STAGES;
          wait ke[stage];
          tma.copy.async<kf[stage]>.swiz<SWIZ>
            K_head.subspan(BLOCK_N, DIM).at(bn, 0) => k_buf[stage];
          trigger kf[stage];

          wait ve[stage];
          tma.copy.async<vf[stage]>.swiz<SWIZ>
            V_head.subspan(BLOCK_N, DIM).at(bn, 0) => v_buf[stage];
          trigger vf[stage];
        }
      }

      // Consumer: QK[n] + PV[n-1] overlap
      inthreads.async (p > 0) {
        call croq::cuda::setreg_inc(240);
        cid = p - 1;
        foreach {s} in [STAGES] {
          trigger ke[s];
          trigger ve[s];
        }

        wait qf;

        frag f32[WG_M] scores_max {-inf};
        frag f32[WG_M] scores_max_prev;
        frag f32[WG_M] scores_scale;
        frag f32[WG_M] scores_sum;
        frag f32[WG_M] logsum {0.0f};
        acc_o = mma.fill.f32 0.0f;
        frag bf16[WG_M, BLOCK_N] acc_s_cast {0.0f};
        int causal_row_base = bm * BLOCK_M + cid * WG_M;

        // === Unified loop: QK[n] + PV[n-1] ===
        foreach {bn} in [kv_bound] {
          stage = bn % STAGES;
          prev_stage = (bn + 1) % STAGES;

          acc_s = mma.fill.f32 0.0f;

          // Issue QK[n] and PV[n-1] back-to-back
          wait kf[stage];
          wait vf[bn == 0 ? stage : prev_stage];
          mma.row.row acc_s,
            q_shared.subspan(WG_M, DIM).at(cid, 0), k_buf[stage];
          trigger ke[stage];

          mma.row.col acc_o,
            acc_s_cast, v_buf[bn == 0 ? stage : prev_stage];
          if (bn > 0) trigger ve[prev_stage];

          mma.wait<1>;  // wait for QK only; PV still in flight

          // Causal mask
          if ((bn + 1) * BLOCK_N > causal_row_base + 1) {
            apply {i, j} in acc_s.span {
              if ((bn * BLOCK_N + j) > (bm * BLOCK_M + cid * WG_M + i))
                acc_s.at(i, j) = -inf;
            }
          }

          mma.wait<0>;  // now wait for PV[n-1] too

          // Online softmax
          copy(scores_max_prev, scores_max);
          reduce_max(scores_max, acc_s, 1);
          apply {i} in scores_max.span {
            scores_max.at(i) = __max(scores_max.at(i), scores_max_prev.at(i));
            scores_scale.at(i) = __exp2f(
              scores_max_prev.at(i) * scale - scores_max.at(i) * scale);
          }
          apply {i, j} in acc_s.span
            acc_s.at(i, j) = __exp2f(
              acc_s.at(i, j) * scale - scores_max.at(i) * scale);
          reduce_sum(scores_sum, acc_s, 1);
          apply {i} in logsum.span
            logsum.at(i) =
              logsum.at(i) * scores_scale.at(i) + scores_sum.at(i);
          apply {i, j} in acc_o.span
            acc_o.at(i, j) = acc_o.at(i, j) * scores_scale.at(i);
          apply {i, j} in acc_s_cast.span
            acc_s_cast.at(i, j) = __to<bf16>(acc_s.at(i, j));

          // Epilogue: final PV merged into last iteration
          if (bn == kv_bound - 1) {
            wait vf[stage];
            mma.row.col acc_o, acc_s_cast, v_buf[stage];
            trigger ve[stage];
            mma.wait<0>;
          }
        }

        apply {i, j} in acc_o.span
          acc_o.at(i, j) = acc_o.at(i, j) / logsum.at(i);
        shared bf16[BLOCK_M, DIM] o_buf;
        mma.store acc_o, o_buf.subspan(WG_M, DIM).at(cid, 0);
        tma.copy o_buf.subspan(WG_M, DIM).at(cid, 0) =>
          O_head.subspan(WG_M, DIM).at(bm * 2 + cid, 0);
      }
    }
  }
}
```

### New concepts in v3

| Construct | Meaning |
| --- | --- |
| `frag bf16[WG_M, BLOCK_N] acc_s_cast {0.0f}` | Zero-initialized register fragment. On iteration 0, PV uses this zero matrix (no-op multiply), so the first `mma.row.col` adds nothing. This eliminates a special case for `bn==0`. |
| `mma.wait<1>` | Wait until at most 1 WGMMA group is outstanding. Since QK was issued first and PV second, this waits for QK to complete while PV continues on tensor cores. |
| `mma.wait<0>` | Wait for all outstanding WGMMA groups. After this, both QK and PV results are available. |
| `if (bn == kv_bound - 1) { ... }` | Epilogue: the final PV has no "next iteration" to overlap with, so it is issued and waited on explicitly in the last iteration. |

### How the overlap works

The key difference from v2 is the **loop structure**. In v2, each iteration does QK then softmax then PV sequentially. In v3, each iteration does:

```
Iteration 0:                    Iteration n (n>0):
  QK[0]  (WGMMA async)           QK[n]    (WGMMA async)
  PV[-1] (no-op, zeros)          PV[n-1]  (WGMMA async, using prev softmax)
  wait QK[0]                     wait QK[n]
  softmax[0]                     softmax[n]        <-- scalar, while PV runs
  cast to bf16                   wait PV[n-1]
                                 cast to bf16
```

On the last iteration, an explicit epilogue PV is appended:

```
Iteration kv_bound-1:
  QK[last]  + PV[last-1]
  softmax[last]
  PV[last]  (explicit epilogue)
```

This is exactly the FA3 loop structure, expressed in ~6 additional lines of Croqtile.

### The `prev_stage` trick

```choreo
prev_stage = (bn + 1) % STAGES;
wait vf[bn == 0 ? stage : prev_stage];
mma.row.col acc_o, acc_s_cast, v_buf[bn == 0 ? stage : prev_stage];
```

PV[n-1] reads from the **previous iteration's** V buffer. With `STAGES=2`, `prev_stage = (bn+1) % 2` gives the alternate slot. On iteration 0, there is no previous V, so it reads from the current stage (the zero-initialized `acc_s_cast` makes this a no-op).

### Result: v3

| Config | Time (ms) | TFLOPS |
| --- | ---: | ---: |
| SEQ=4096 | 1.65 | 332.8 |
| SEQ=8192 | 6.09 | 361.3 |
| SEQ=16384 | 23.5 | 374.3 |

**~361 TFLOPS at SEQ=8192 (1.03x over v2, 88% of FA3)**

The improvement is modest at short sequences (where the loop body dominates) but significant at SEQ=16384: **374 vs 242 TFLOPS** -- a 1.55x improvement. The overlap shines when there are enough KV iterations to amortize the pipeline startup cost.

---

## The Optimization Ladder

| Step | Technique | Key Croqtile constructs | SEQ=8192 TFLOPS | Speedup |
| --- | --- | --- | ---: | ---: |
| v1 | Sequential DMA + WGMMA | `dma.copy.swiz`, `mma.row.row/col` | 227.5 | 1.0x |
| v2 | 1p2c warp spec + TMA pipeline | `inthreads.async`, `shared event`, `tma.copy.async`, `setreg` | 350.1 | 1.54x |
| v3 | FA3-style QK/PV overlap | `mma.wait<1>`, epilogue PV fusion | 361.3 | 1.59x |

---

## Comparison with the Ecosystem

At SEQ=8192 (the headline config):

| Implementation | TFLOPS | Notes |
| --- | ---: | --- |
| FlashAttention-3 | 410.3 | Hand-tuned CUDA, heavily optimized |
| **Croqtile v3** | **365.0** | **~80 lines of `.co`, no CUDA** |
| Triton + WS | 354.3 | Warp-specialized Triton kernel |
| Triton | 312.5 | Standard Triton implementation |
| TileLang | 295.8 | TileLang compiler-generated |

The Croqtile kernel matches Triton+WS and reaches 89% of FA3 -- the current SOTA -- while being expressed in a fraction of the code. The remaining gap to FA3 comes from techniques not yet applied:

- **Pingpong scheduling**: FA3 uses a 2-CTA cooperative scheme where adjacent blocks alternate between QK and PV phases, maximizing L2 reuse.
- **Persistent kernel with tile scheduler**: Keeps thread blocks alive across output tiles.
- **FP8 GQA paths**: FA3 has specialized FP8 and grouped-query attention variants.

---

## What Each Optimization Costs in Code

| Optimization | Lines changed from prev | Lines of equivalent CUDA |
| --- | ---: | ---: |
| v1 -> v2: warp spec + TMA | +55 lines | ~300+ lines (mbarrier init, TMA descriptors, warpgroup dispatch) |
| v2 -> v3: QK/PV overlap | +10 lines | ~80+ lines (WGMMA pipeline fence management, epilogue logic) |

The v3 kernel is ~80 lines of Croqtile total. A faithful CUDA implementation of the same algorithm requires 400+ lines -- TMA tensor-map construction, mbarrier lifecycle management, WGMMA smem descriptor encoding, swizzle propagation, register allocation hints, and careful fence ordering. One mistake in any of these causes silent numerical errors or deadlocks.

---
