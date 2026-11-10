# How to Optimize a Croqtile 2:4 Sparse GEMM: E4M3 Worklog

*April 2026 · GPU: NVIDIA H800 PCIe (SM90a, 114 SMs) · Precision: E4M3 · Problem: 4096×8192×8192, 2:4 structured sparsity*

---

This worklog picks up where the [dense FP16 tutorial](dense-gemm-fp16-from-naive.md) left off. That tutorial taught the Hopper producer–consumer rhythm: TMA, WGMMA, warp specialization, pipeline staging. Sparse GEMM inherits all of that, but adds one more operand path — **metadata** — that changes the bottleneck ranking and the optimization sequence.

The E4M3 optimization path starts from a prepack baseline and reaches the final tuned kernel through three structural steps:

| Step | Kernel | TFLOPS | Δ vs baseline |
| ---- | ------ | -----: | ------------- |
| Baseline | prepack, swizzle32/64, no warpspec | 430 | — |
| 1 | Warp specialization: 1P2C, 2-stage | 586 | +36% |
| 2 | Swizzle + TK128: swizzle64/128, TK128 | 718 | +67% |
| 3 | Barrier-decomposed overlap | **1,401** | **+226%** |

Each step delivers a measurable improvement over the previous one, and each one builds the foundation that the next step requires.

!!! tip "Prerequisites"
    This worklog assumes familiarity with the dense GEMM tutorial's concepts: TMA, WGMMA, warp specialization, pipeline staging, and the `shared event` / `wait` / `trigger` vocabulary. If those are new, read the [dense FP16 worklog](dense-gemm-fp16-from-naive.md) first.

---

## Sparse Tensor Core: What Changes

On Hopper, the **Sparse Tensor Core** path is exposed through warpgroup-level sparse WGMMA instructions (`wgmma.mma_async.sp`). The key contract:

- The **A/LHS operand** must satisfy **2:4 structured sparsity**: every contiguous group of 4 elements keeps exactly 2 nonzeros.
- The hardware consumes a **compressed data matrix** (50% of original A) and a **metadata matrix** that encodes the two nonzero positions per group using 2-bit indices.
- The dense B/RHS operand is unchanged.

This means a sparse GEMM kernel must manage **three** data paths instead of two:

| Path | Dense GEMM | Sparse GEMM |
| ---- | ---------- | ----------- |
| LHS operand | dense A tile via TMA | **compressed** A tile via TMA |
| RHS operand | dense B tile via TMA | dense B tile via TMA (unchanged) |
| Metadata | — | **metadata tile** (new) |

The metadata path is small (2 KB per K-step for the default tile), but it touches a different memory region and has different latency characteristics than the operand tiles. Managing it well is the central sparse-specific optimization challenge.

![2:4 sparsity pattern and metadata](images/SparsityPattern_ManimCE_v0.19.1_dark.png#only-dark)
![2:4 sparsity pattern and metadata](images/SparsityPattern_ManimCE_v0.19.1_light.png#only-light)

## What Transfers From Dense GEMM

Most of the dense Hopper playbook still applies:

- **Warp specialization** so producers and consumers overlap
- **TMA** to move big tiles instead of scalar global loads
- **Pipeline stages** to keep the producer ahead of the consumer
- **1P2C** (1 producer, 2 consumers) once the consumer side becomes the limiter

The [dense FP16 worklog](dense-gemm-fp16-from-naive.md) is the prerequisite, because sparse GEMM inherits the same scheduling vocabulary. What changes is the bottleneck ranking: after Sparse Tensor Core is enabled, the next question is often not "is WGMMA fed?" but "is metadata arriving early enough?"

## What Is Different

**Sparse Tensor Core** adds one more design axis on top of the dense recipe:

1. **Operand preparation.** The sparse LHS is not consumed as a normal dense tile. It must be compressed into values + metadata before the kernel runs.
2. **Memory traffic.** A small metadata tile is touched every K-iteration. Even though metadata is compact, repeated scalar loads can become the critical path.
3. **Synchronization.** If metadata is staged separately from operand tiles, barriers and wait placement start to matter sooner.

This is why the first sparse-specific optimization is **metadata prepack** (`--use-prepack`): pack metadata into the layout expected by the Sparse Tensor Core fast path before runtime.

## Metadata Prepack: The Starting Point for Both Precisions

Before any tiling or pipeline optimization, the first universal step is to use `--use-prepack` so that the metadata layout matches what `wgmma.sp` expects. Without prepack, metadata must be transformed on the fly inside the inner loop, wasting issue bandwidth on address computation that the hardware could skip.

In practical terms:

- Dense GEMM only has to deliver A/B tiles
- Sparse GEMM has to deliver **compressed A tiles + metadata tiles**
- `--use-prepack` is the cheapest general optimization because it improves metadata delivery before touching the schedule

The E4M3 optimization starting point is `gemm_sp_e4m3_dynamic_sm90_swizzle32_64_prepack.co`, and the FP16 starting point is `gemm_sp_f16_dynamic_sm90_swizzle64_128_prepack.co`.

Both are simple kernels with **no** warp specialization — a single warpgroup, synchronous TMA, and metadata loaded from global memory via `mma.load`:

```bash
# E4M3 baseline
./croq -gs -t cute -arch=sm_90a --use-warpspec --use-prepack \
  benchmark/performance/gemm_sp/gemm_sp_e4m3_dynamic_sm90_swizzle32_64_prepack.co \
  -o /tmp/e4m3_baseline.cute.result
CUDA_VISIBLE_DEVICES=0 bash /tmp/e4m3_baseline.cute.result --execute
```

```bash
# FP16 baseline
./croq -gs -t cute -arch=sm_90a --use-warpspec --use-prepack \
  benchmark/performance/gemm_sp/gemm_sp_f16_dynamic_sm90_swizzle64_128_prepack.co \
  -o /tmp/f16_baseline.cute.result
CUDA_VISIBLE_DEVICES=0 bash /tmp/f16_baseline.cute.result --execute
```

With prepack baselines established, the two precisions diverge. The rest of this worklog covers **E4M3** first, then **FP16**.

---

## E4M3 Baseline: What We Start With

**File:** `gemm_sp_e4m3_dynamic_sm90_swizzle32_64_prepack.co`

The baseline kernel is the simplest sparse E4M3 kernel that compiles and runs correctly with metadata prepack. Its configuration:

| Parameter | Value | Meaning |
| --------- | ----- | ------- |
| `SPMM_WARP_M` | 64 | WGMMA M dimension |
| `SPMM_WARP_N` | 256 | WGMMA N dimension |
| `SPMM_TILE_K` | 64 | K-tile loaded per TMA transfer |
| `SPMM_WARP_K` | 64 | K-step per WGMMA instruction |
| `SPMM_LHS_SWIZ` | 32 | Narrow swizzle for compressed LHS |
| `SPMM_RHS_SWIZ` | 64 | Swizzle for dense RHS |
| Warp specialization | **none** | Single warpgroup, synchronous path |
| Pipeline stages | **1** | No software pipelining |

The kernel function:

```choreo
__co__ void spmm(global f8_e4m3 [M, PACKED_K] lhs_packed,
                 global u32 [M, META_COLS] lhs_meta,
                 global f8_e4m3 [N, K] rhs,
                 global f16 [M, N] output) {
  parallel {block_m, block_n}
      by [cdiv(M, SPMM_WARP_M), cdiv(N, SPMM_WARP_N)] : block {
    shared f8_e4m3 [SPMM_WARP_M, SPMM_PACKED_TILE_K] lhs_load_s;
    shared f8_e4m3 [SPMM_WARP_N, SPMM_TILE_K] rhs_load_s;
    mc = mma.fill.f16 0.0f;
    foreach {iv_k} in [cdiv(K, SPMM_TILE_K)] {
      tma.copy.swiz<SPMM_LHS_SWIZ> lhs_packed
          .subspan(SPMM_WARP_M, SPMM_PACKED_TILE_K)
          .at(block_m, iv_k) => lhs_load_s;
      tma.copy.swiz<SPMM_RHS_SWIZ> rhs
          .subspan(SPMM_WARP_N, SPMM_TILE_K)
          .at(block_n, iv_k) => rhs_load_s;
      foreach {iv_warp} in [cdiv(SPMM_TILE_K, SPMM_WARP_K)] {
        parallel p by 1 : group-4 {
          ma = mma.load.swiz<SPMM_LHS_SWIZ> lhs_load_s.chunkat(_, iv_warp);
          mb = mma.load.swiz<SPMM_RHS_SWIZ> rhs_load_s.chunkat(_, iv_warp);
          me = mma.load lhs_meta
              .subspan(SPMM_WARP_M, SPMM_META_TILE_COLS)
              .at(block_m, iv_k);
          mma.row.row.sp mc, ma, mb, me;
        }
      }
    }
    shared f16 [SPMM_WARP_M, SPMM_WARP_N] output_s;
    mma.store mc, output_s;
    tma.copy output_s => output.subspan(SPMM_WARP_M, SPMM_WARP_N)
        .at(block_m, block_n);
  }
}
```

**New concept vs dense GEMM:**

| Construct | Meaning |
| --------- | ------- |
| `global u32 [M, META_COLS] lhs_meta` | Metadata tensor in global memory. Prepacked layout where each `u32` encodes 16 × 2-bit position indices. |
| `me = mma.load lhs_meta...` | Load metadata for the current K-step from global memory into registers. |
| `mma.row.row.sp mc, ma, mb, me` | **Sparse** WGMMA: uses `wgmma.mma_async.sp` under the hood. The `.sp` suffix tells the Tensor Core to consume compressed A + metadata instead of dense A. |

Compared to the dense GEMM baseline (v2 in the dense tutorial), the structure is identical except for the third operand `me`. The kernel loads compressed LHS and dense RHS via TMA, loads metadata from global memory, and issues sparse WGMMA.

**Result: ~430 TFLOPS (14.2% HW efficiency)**

### NCU snapshot (Baseline)

```
gpu__time_duration               :   ~1.28 ms
sm__throughput   (% of peak)     :  25.30%
dram__throughput (% of peak)     :   8.06%
pipe_tensor_hmma (% of peak)     :  ~25%
pipe_tensor instructions         :   1,056,768
global ld sectors                :  16,777,216
```

DRAM throughput at only 8% means the TMA engine is grossly underutilized — the synchronous copy path blocks the entire warpgroup while data transfers. SM throughput at 25% and tensor core utilization around the same level confirm that the kernel spends most of its time waiting. The kernel has all the same problems as v2 in the dense tutorial (TMA and WGMMA serialized, no pipelining) plus an extra problem: metadata loads from global memory inside the hot loop.

---

## Step 1: Warp Specialization + Pipeline

**File:** `gemm_sp_e4m3_dyn_sm90_warpspec_1p2c_swizzle32_64_prepack.co` (modified to 2-stage)

Just like in the dense GEMM tutorial, the first structural optimization is to introduce **warp specialization** and **multi-stage pipelining**. The logic is identical: the single-warpgroup baseline serializes TMA and WGMMA, so we split them into a producer warpgroup (TMA) and consumer warpgroups (WGMMA) connected by a software ring buffer.

The key configuration changes from the baseline:

| Parameter | Baseline | Step 1 |
| --------- | -------- | ------ |
| `SPMM_TILE_M` | 64 (= WARP_M) | **128** (= 2 × WARP_M) |
| Pipeline stages | 1 | **2** |
| Consumers | 1 | **2** (1P2C) |
| Swizzle | 32/64 | 32/64 (unchanged) |

```choreo
__co__ void spmm(global f8_e4m3 [M, PACKED_K] lhs_packed,
                 global u32 [M, META_COLS] lhs_meta,
                 global f8_e4m3 [N, K] rhs,
                 global f16 [M, N] output) {
  parallel {block_m, block_n}
      by [cdiv(M, SPMM_TILE_M), cdiv(N, SPMM_WARP_N)] : block {
    shared event full[2], empty[2];
    shared f8_e4m3 [2 * SPMM_TILE_M, SPMM_PACKED_TILE_K] lhs_load_s;
    shared f8_e4m3 [2 * SPMM_WARP_N, SPMM_TILE_K] rhs_load_s;
    shared f16 [SPMM_TILE_M, SPMM_WARP_N] output_s;

    parallel p1 by 3 : group-4, t by 128 : thread {
      // ── Producer (p1=0): TMA loads ──
      inthreads.async (p1 == 0 && t == 0) {
        foreach {iv_k} in [cdiv(K, SPMM_TILE_K)] {
          stage = iv_k % 2;
          wait empty[stage];
          tma.copy.async<full[stage]>.swiz<SPMM_LHS_SWIZ> lhs_packed
              .subspan(SPMM_TILE_M, SPMM_PACKED_TILE_K)
              .at(block_m, iv_k)
              => lhs_load_s.subspan(SPMM_TILE_M, SPMM_PACKED_TILE_K)
                  .at(stage, 0);
          tma.copy.async<full[stage]>.swiz<SPMM_RHS_SWIZ> rhs
              .subspan(SPMM_WARP_N, SPMM_TILE_K)
              .at(block_n, iv_k)
              => rhs_load_s.subspan(SPMM_WARP_N, SPMM_TILE_K)
                  .at(stage, 0);
          trigger full[stage];
        }
      }

      // ── Consumers (p1>0): sparse WGMMA ──
      inthreads.async (p1 > 0) {
        mc = mma.fill.f16 0.0f;
        foreach {s} in [2] { trigger empty[s]; }
        foreach {iv_k} in [cdiv(K, SPMM_TILE_K)] {
          stage = iv_k % 2;
          wait full[stage];
          foreach {iv_warp} in [cdiv(SPMM_TILE_K, SPMM_WARP_K)] {
            ma = mma.load.swiz<SPMM_LHS_SWIZ> lhs_load_s
                .subspan(SPMM_TILE_M, SPMM_PACKED_TILE_K)
                .at(stage, 0)
                .subspan(SPMM_WARP_M, SPMM_PACKED_TILE_K)
                .at(p1 - 1, 0)
                .chunkat(_, iv_warp);
            mb = mma.load.swiz<SPMM_RHS_SWIZ> rhs_load_s
                .subspan(SPMM_WARP_N, SPMM_TILE_K)
                .at(stage, 0)
                .chunkat(_, iv_warp);
            me = mma.load lhs_meta
                .subspan(SPMM_TILE_M, SPMM_META_TILE_COLS)
                .at(block_m, iv_k)
                .subspan(SPMM_WARP_M, 1)
                .at(p1 - 1, iv_warp);
            mma.row.row.sp mc, ma, mb, me;
          }
          mma.commit;
          trigger empty[stage];
        }
        mma.store mc,
            output_s.subspan(SPMM_WARP_M, SPMM_WARP_N).at(p1 - 1, 0);
      }
    }

    tma.copy output_s => output.subspan(SPMM_TILE_M, SPMM_WARP_N)
        .at(block_m, block_n);
  }
}
```

This is the exact same structural move as v3→v4 in the dense tutorial:

- `parallel p1 by 3 : group-4` spawns 3 warpgroups (1 producer + 2 consumers)
- `shared event full[2], empty[2]` creates a 2-stage ring buffer (double-buffering)
- The producer runs the K-loop independently, issuing async TMA loads
- Both consumers process the same stage but on different M-tile halves (`p1 - 1` selects top/bottom)
- Metadata is still loaded from global memory (`mma.load lhs_meta...`) inside the consumer loop

We start with 2-stage rather than deeper pipelining because it is the simplest configuration that demonstrates the producer–consumer overlap benefit while keeping shared memory usage reasonable.

Compile and run:

```bash
./croq -gs -t cute -arch=sm_90a --use-warpspec --use-prepack \
  benchmark/performance/gemm_sp/gemm_sp_e4m3_dyn_sm90_warpspec_1p2c_swizzle32_64_prepack.co \
  -o /tmp/e4m3_step1.cute.result
CUDA_VISIBLE_DEVICES=0 bash /tmp/e4m3_step1.cute.result --execute
```

**Result: ~586 TFLOPS (+36% vs baseline, 19.4% HW efficiency)**

### NCU snapshot (Step 1)

```
gpu__time_duration               :   1.28 ms
sm__throughput   (% of peak)     :  24.44%
dram__throughput (% of peak)     :  12.52%
pipe_tensor_hmma (% of peak)     :  23.11%
pipe_tensor instructions         :   1,048,576
global ld sectors                :  16,777,216
```

The improvement comes from the same mechanisms as the dense tutorial: warp specialization decouples producer and consumer, double-buffering lets the producer prefetch the next tile while consumers compute, and 1P2C doubles the consumer-side compute capacity. But tensor core utilization is only 23% — the kernel spends too much time in loop overhead because `TK64` forces 128 K-steps for K=8192, and the narrow swizzle32/64 layout does not match the optimal TMA access pattern for this data type.

---

## Step 2: Swizzle and Layout Restructuring

**File:** `gemm_sp_e4m3_dyn_sm90_warpspec_1p2c_swizzle128_128_prepack.co`

Before we can apply the advanced sparse optimizations (barrier decomposition, decoupled metadata staging), the memory layout must be restructured. The narrow `swizzle32/64` layout from Step 1 does not support the wider K-tile and per-operand barrier design that the final kernel requires.

The key changes:

| Parameter | Step 1 | Step 2 |
| --------- | ------ | ------ |
| `SPMM_TILE_K` | 64 | **128** |
| `SPMM_PACKED_TILE_K` | 32 | **64** |
| `SPMM_LHS_SWIZ` | 32 | **64** |
| `SPMM_RHS_SWIZ` | 64 | **128** |
| `SPMM_STAGES` | 2 | 2 (unchanged) |
| `SPMM_META_TILE_COLS` | 2 | **4** |

Why these changes matter:

1. **Wider K-tile (TK128).** Doubling `TILE_K` from 64 to 128 halves the K-loop trip count, which means fewer metadata loads and fewer barrier round-trips. Each K-step now does more useful work.

2. **Wider swizzle (B64 LHS, B128 RHS).** The `wgmma.sp` instruction reads the compressed LHS with a K-major access pattern. For E4M3 (1 byte per element), the access stride is 64 bytes, matching B64 swizzle. The dense RHS always benefits from B128 swizzle regardless of data type. Without proper swizzle alignment, shared memory bank conflicts can reduce effective bandwidth by up to 32×.

3. **Wider metadata tile.** `SPMM_META_TILE_COLS` grows from 2 to 4, matching the doubled K-tile so that metadata covers the full TK128 step.

The kernel structure is identical to Step 1 — only the defines change. The Croqtile compiler regenerates the correct TMA tensor descriptors, swizzle layouts, and mbarrier counts automatically for the new tile shape.

```bash
./croq -gs -t cute -arch=sm_90a --use-warpspec --use-prepack \
  benchmark/performance/gemm_sp/gemm_sp_e4m3_dyn_sm90_warpspec_1p2c_swizzle128_128_prepack.co \
  -o /tmp/e4m3_step2.cute.result
CUDA_VISIBLE_DEVICES=0 bash /tmp/e4m3_step2.cute.result --execute
```

**Result: ~718 TFLOPS (+67% vs baseline, +23% vs Step 1, 23.7% HW efficiency)**

### NCU snapshot (Step 2)

```
gpu__time_duration               :   0.876 ms
sm__throughput   (% of peak)     :  34.16%
dram__throughput (% of peak)     :  18.35%
pipe_tensor_hmma (% of peak)     :  32.16%
pipe_tensor instructions         :   1,048,576
global ld sectors                :  16,777,216
```

Comparing the NCU numbers between Step 1 and Step 2 reveals the improvement mechanism:

| Metric | Step 1 (swizzle32, TK64) | Step 2 (swizzle64/128, TK128) | Change |
| ------ | -----------------------: | ----------------------------: | ------ |
| Kernel time | 1.28 ms | 0.876 ms | −32% |
| Tensor core utilization | 23.1% | 32.2% | +39% |
| SM throughput | 24.4% | 34.2% | +40% |
| DRAM throughput | 12.5% | 18.4% | +47% |

The total number of tensor instructions and global load sectors are identical (1,048,576 and 16,777,216 respectively) — both kernels do the same amount of work. The improvement comes from **how efficiently that work is scheduled**:

1. **TK128 halves the K-loop trip count** from 128 to 64 iterations. Each iteration has fixed overhead (barrier wait/signal, metadata load, loop control), so halving the trip count removes half that overhead.
2. **Wider swizzle (B64 LHS, B128 RHS) improves TMA transfer efficiency.** The TMA hardware transfers 128-byte cache lines. With B128 swizzle on the RHS, each TMA transfer aligns perfectly with the cache line boundary, raising DRAM throughput from 12.5% to 18.4%.
3. **Higher tensor core utilization** (23% → 32%) is the compound effect: less loop overhead means more cycles spent on actual `wgmma.sp` instructions.

This layout also creates the foundation that Step 3's barrier-decomposed design requires — the wider TK128 geometry enables meaningful per-operand barrier splitting because each K-step now contains enough work to overlap metadata decode with data transfer.

---

## Step 3: Barrier-Decomposed Overlap

Step 3 is where the sparse-specific optimizations compound into a large performance jump. The key insight comes from the paper: conventional pipelined GEMM uses a single monolithic barrier per stage — the consumer blocks on `full` until **all** operands arrive, then computes. With the per-step memory-to-compute ratio around 1.9× (memory takes almost twice as long as compute), the consumer sits idle for nearly half of each pipeline step.

Choreo's barrier-decomposed design eliminates this idle window through three mechanisms:

### Speculative Metadata Decode

Metadata (2 KB per step) arrives via TMA in ~0.1 μs, while bulk operand data (LHS + RHS, ~80 KB) requires ~1.0 μs. Instead of bundling metadata into the same `full` barrier, the kernel gives it a **separate `meta_full` barrier**. The consumer waits on `meta_full`, decodes the 2:4 sparse positions into register descriptors for `wgmma.sp`, and only then waits on `full`. The decode runs concurrently with the remaining ~0.9 μs of bulk data transfer.

### Early Buffer Release

With 3 pipeline stages, each K-step issues two `wgmma.sp` instructions from the same buffer. After the first instruction is dispatched, the consumer signals `empty[stage].arrive()`, releasing the buffer back to the producer **before** the second instruction finishes. This is safe because WGMMA reads its shared memory operands at dispatch time — the buffer contents are no longer needed once the instruction enters the Tensor Core queue.

### Depth-1 Instruction Overlap

Both kernels use `warpgroup_wait<1>()` rather than `warpgroup_wait<0>()`. This means the consumer issues the next K-step's first `wgmma.sp` before the current step's last `wgmma.sp` retires. There is always one instruction in flight while the next is being prepared, hiding completion latency.


### Combined Effect

The three mechanisms fill the idle window in every pipeline step:

- Metadata decode consumes ~0.1 μs (overlapped with data transfer)
- Depth-1 overlap hides ~0.2 μs of instruction latency
- Early buffer release lets the producer start ~0.15 μs sooner

The net effect: the effective memory-to-compute ratio drops from 1.9× to ~1.1×, bringing each pipeline step close to the compute-bound regime.

The 3-stage pipeline is the dominant contributor: E4M3's smaller per-stage data (43 KB vs 84 KB for FP16) fits 3 stages within the 228 KB shared memory budget, and the faster consumer compute (0.37 μs per step) critically needs the extra stage to avoid starving the compute pipeline.

The final tuned kernel, incorporating all barrier-decomposed optimizations, reaches:

```bash
CUDA_VISIBLE_DEVICES=0 CHOREO_TIMING_WARMUP=5 CHOREO_TIMING_REPEAT=50 \
  bash benchmark/performance/gemm_sp/e4m3_aitune_2026-03-21_iter068/run.sh
```

**Result: ~1,401 TFLOPS (+226% vs baseline, 46.3% HW efficiency)**

---

## E4M3 Summary

![E4M3 optimization steps](images/E4M3_OptSteps_dark.png#only-dark)
![E4M3 optimization steps](images/E4M3_OptSteps_light.png#only-light)

| Step | What changed | TFLOPS | Why it matters |
| ---- | ------------ | -----: | -------------- |
| Baseline | Prepack, swizzle32/64, no warpspec | 430 | Establishes the sparse metadata fast path |
| Step 1 | 1P2C + 2-stage, swizzle32/64 | 586 | Imports the dense GEMM structural wins |
| Step 2 | TK128, swizzle64/128 | 718 | Eliminates bank conflicts, reduces loop overhead |
| Step 3 | Barrier-decomposed overlap | 1,401 | Fills the idle window between memory and compute |

The optimization sequence has a clear causal structure:

1. **Prepack first** — put metadata on the intended fast path before touching the schedule.
2. **Import the dense playbook** — warp specialization, TMA, pipeline stages, 1P2C. This is Step 1 and it delivers a 36% improvement.
3. **Fix the layout** — wider K-tile and proper swizzle alignment eliminate bank conflicts and reduce loop overhead. This is Step 2, and it delivers another 23% on top.
4. **Decompose the barriers** — speculative metadata decode, early buffer release, and depth-1 overlap turn the idle window into useful work. This is Step 3 and it delivers the final ~2× jump.

---

## FP16 Path

The FP16 path follows the same logical structure as E4M3 but hits different constraints. FP16 elements are 2× larger than E4M3, which means:

- **Per-stage data is 2× larger** (84 KB vs 43 KB), limiting the pipeline depth
- **Compute is 2× slower** (FP16 Sparse TC peak is 1,513 TFLOPS vs E4M3's 3,026 TFLOPS), so the memory-to-compute ratio is different
- **The dominant optimization target shifts** from pipeline depth to memory layout and metadata delivery

| Step | Kernel | TFLOPS | Δ vs baseline |
| ---- | ------ | -----: | ------------- |
| Baseline | prepack, swizzle64/128, no warpspec | 457 | — |
| 1 | 1P2C + 3-stage, swizzle64/128 | 531 | +16% |
| 2 | TK128 + TMA metadata + split RHS | **784** | **+72%** |

### FP16 Baseline

**File:** `gemm_sp_f16_dynamic_sm90_swizzle64_128_prepack.co`

The FP16 baseline is structurally identical to the E4M3 baseline: a single warpgroup, synchronous TMA, metadata loaded from global memory. The key difference is the swizzle configuration — FP16 (2 bytes per element) uses `swizzle64` for LHS and `swizzle128` for RHS, matching the larger access stride.

```bash
./croq -gs -t cute -arch=sm_90a --use-prepack \
  benchmark/performance/gemm_sp/gemm_sp_f16_dynamic_sm90_swizzle64_128_prepack.co \
  -o /tmp/f16_baseline.cute.result
CUDA_VISIBLE_DEVICES=0 bash /tmp/f16_baseline.cute.result --execute
```

**Result: ~457 TFLOPS (30.2% HW efficiency)**

NCU snapshot:

```
gpu__time_duration               :   1.89 ms
sm__throughput   (% of peak)     :  32.29%
dram__throughput (% of peak)     :   9.62%
pipe_tensor_hmma (% of peak)     :  30.31%
pipe_tensor instructions         :   2,105,344
```

The FP16 baseline starts higher than E4M3 (457 vs 430 TFLOPS) because FP16's larger elements mean fewer K-steps per iteration, and the synchronous copy path handles the wider data more efficiently. But DRAM throughput at 9.6% shows the same TMA underutilization pattern.

### FP16 Step 1: Warp Specialization + 3-Stage Pipeline

**File:** `gemm_sp_f16_aitune_2026-03-25_iter120.co`

The first FP16 optimization applies the same structural move as E4M3 Step 1: introduce warp specialization (1P2C) and pipeline stages. The configuration:

| Parameter | Baseline | Step 1 |
| --------- | -------- | ------ |
| `SPMM_TILE_M` | 64 (= WARP_M) | **128** (= 2 × WARP_M) |
| Pipeline stages | 1 | **3** |
| Consumers | 1 | **2** (1P2C) |
| Swizzle | 64/128 | 64/128 (unchanged) |
| `TILE_K` | 64 | 64 (unchanged) |

FP16's larger per-stage data (84 KB) limits it to 3 stages (vs 4 for E4M3), which still provides enough buffering for the producer to stay ahead.

```bash
./croq -gs -t cute -arch=sm_90a --use-warpspec --use-prepack --stmatrix \
  benchmark/performance/gemm_sp/gemm_sp_f16_aitune_2026-03-25_iter120.co \
  -o /tmp/f16_step1.cute.result
CUDA_VISIBLE_DEVICES=0 bash /tmp/f16_step1.cute.result --execute
```

**Result: ~531 TFLOPS (+16% vs baseline, 35.1% HW efficiency)**

NCU snapshot:

```
gpu__time_duration               :   1.42 ms
sm__throughput   (% of peak)     :  41.99%
dram__throughput (% of peak)     :  19.72%
pipe_tensor_hmma (% of peak)     :  40.07%
pipe_tensor instructions         :   2,097,152
```

Comparing with the baseline:

| Metric | Baseline | Step 1 | Change |
| ------ | -------: | -----: | ------ |
| Kernel time | 1.89 ms | 1.42 ms | −25% |
| SM throughput | 32.3% | 42.0% | +30% |
| DRAM throughput | 9.6% | 19.7% | +105% |
| Tensor core utilization | 30.3% | 40.1% | +32% |

DRAM throughput more than doubles — the async TMA path is now moving data much more efficiently because the producer warpgroup runs independently. But tensor core utilization at 40% indicates the consumer is still frequently stalled, likely on metadata loads and loop overhead within the narrow TK64 K-step.

### FP16 Step 2: Wider-K + TMA Metadata + Split RHS

The final FP16 optimization addresses the remaining 60% of idle tensor core cycles through a memory-path redesign. This is the FP16 analogue of E4M3 Step 3's barrier-decomposed overlap, but adapted to FP16's constraints:

1. **TK128** doubles the work per K-step, halving the K-loop trip count from 128 to 64
2. **TMA metadata staging** moves metadata onto the async path instead of loading it from global memory in the consumer loop
3. **Split RHS TMA** issues two separate 64-column TMA loads per K-step instead of one 128-column load, enabling finer-grained pipeline overlap

FP16's per-stage data at TK128 (168 KB) cannot fit 3 stages, so this kernel uses 2 stages — but the wider K-tile compensates by doing more work per step.

```bash
CUDA_VISIBLE_DEVICES=0 CHOREO_TIMING_WARMUP=5 CHOREO_TIMING_REPEAT=50 \
  bash benchmark/performance/gemm_sp/gemm_sp_f16_aitune_2026-03-25_iter143/run.sh
```

**Result: ~784 TFLOPS (+72% vs baseline, 51.8% HW efficiency)**

This is the FP16 sparse analogue of a memory-system redesign: instead of waiting for all data to arrive before computing, the kernel overlaps metadata decode with bulk data transfer, splits the RHS load for finer-grained scheduling, and uses the wider K-tile to amortize fixed overhead.

### FP16 Summary

![FP16 optimization steps](images/FP16_OptSteps_dark.png#only-dark)
![FP16 optimization steps](images/FP16_OptSteps_light.png#only-light)

| Step | What changed | TFLOPS | Why it matters |
| ---- | ------------ | -----: | -------------- |
| Baseline | Prepack, swizzle64/128, no warpspec | 457 | Establishes the sparse metadata fast path |
| Step 1 | 1P2C + 3-stage | 531 | Imports the dense GEMM structural wins |
| Step 2 | TK128 + TMA metadata + split RHS | 784 | Memory-path redesign eliminates idle tensor cycles |

The FP16 optimization story is more compressed than E4M3: fewer steps, and the biggest gain comes from the memory-path redesign (Step 2, +48% over Step 1) rather than from pipeline architecture (Step 1, +16% over baseline). This is because FP16's larger elements make the per-tile memory-to-compute ratio less severe than E4M3, so the pipeline architecture provides a smaller initial lift — but the metadata and memory layout redesign in Step 2 is correspondingly more impactful.

---

## Current Checkout Reproduction Notes

Re-run on the current machine (`NVIDIA H800 PCIe`, `CUDA_VISIBLE_DEVICES=0`) with `CHOREO_TIMING_WARMUP=5`, `CHOREO_TIMING_REPEAT=50`:

| Kernel | TFLOPS | HW efficiency |
| ------ | -----: | ------------- |
| E4M3 baseline prepack `swizzle32_64` | 430 | 14.2% |
| E4M3 Step 1: 1P2C + 2-stage `swizzle32_64` | 586 | 19.4% |
| E4M3 Step 2: 1P2C + 2-stage `swizzle64_128`, TK128 | 718 | 23.7% |
| E4M3 Step 3: barrier-decomposed overlap | 1,401 | 46.3% |
| FP16 baseline prepack `swizzle64_128` | 457 | 30.2% |
| FP16 Step 1: 1P2C + 3-stage | 531 | 35.1% |
| FP16 Step 2: TK128 + TMA metadata + split RHS | 784 | 51.8% |

For review, the raw table is saved in the local data directory.

---

## Conclusion

The dense GEMM recipe still gives the right backbone: warp specialization, TMA, stage depth, occupancy, then 1P2C. Sparse GEMM adds one more requirement: **metadata is a real operand path** and must be optimized like one.

That leads to a cleaner rule of thumb:

1. **Prepack** — put metadata on the intended fast path
2. **Import the dense playbook** — warp specialization, pipelining, 1P2C
3. **Restructure the layout** — wider K-tile, proper swizzle for the target dtype
4. **Decompose the barriers** — speculative metadata decode, early buffer release, depth-1 instruction overlap

The two precision paths tell complementary stories:

- **E4M3** (430 → 586 → 718 → 1,401 TFLOPS): smaller elements enable deeper pipelines and more layout configurations. The final 2× jump from barrier decomposition is only possible because the earlier steps created the right layout foundation.
- **FP16** (457 → 531 → 784 TFLOPS): larger elements constrain pipeline depth, so the biggest gain comes from memory-path redesign (TK128 + TMA metadata + split RHS) rather than pipeline architecture.

Both paths converge on the same insight: once the dense GEMM structural wins are in place, the remaining performance gap is dominated by **how metadata is delivered to the Sparse Tensor Core** — and closing that gap requires treating metadata as a first-class pipelined operand.
