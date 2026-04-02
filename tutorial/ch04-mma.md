# Tensor Contraction: the `mma` Syntax

Chapter 3's tiled matmul computes the inner contraction the way a CPU would: a `foreach k` loop where each iteration multiplies two scalars and accumulates into one output element. That works. It is also the slowest way to use a modern accelerator, because it ignores the hardware unit designed specifically for this job.

The job is **2D tensor contraction** — the operation C += A × B where A, B, and C are small, fixed-shape matrix tiles. It is the inner kernel of every GEMM, every convolution expressed as an im2col matmul, every attention head's QK^T. The operation is so central that hardware vendors build dedicated units to execute it in one macro-instruction: NVIDIA calls them **tensor cores**, Google calls them **MXUs**, Intel has **AMX tiles**, and custom DSAs have their own variants. The tile sizes differ (16×16×16 for FP16 on NVIDIA, 128×128 systolic on TPU, 16×64 on AMX), but the mathematical shape is the same everywhere: take a tile of A, a tile of B, multiply, accumulate into C.

![2D tensor contraction: A[M,K] × B[K,N] → C[M,N], with different hardware implementations](../assets/images/ch04/fig1_tensor_contraction_dark.png#only-dark)
![2D tensor contraction: A[M,K] × B[K,N] → C[M,N], with different hardware implementations](../assets/images/ch04/fig1_tensor_contraction_light.png#only-light)

*The same mathematical operation — C += A × B on tile-shaped operands — maps to different hardware implementations on different accelerators.*

What makes this hard for programmers is not the math but the **register layout**. On a GPU tensor core, the tile is not stored contiguously in one thread's registers. It is **fragmented**: 32 threads in a warp each own scattered pieces of the tile, and the exact scatter pattern depends on the datatype, the architecture generation, and whether the operand is A, B, or C. Writing raw CUDA means declaring `wmma::fragment` objects, calling `load_matrix_sync` to distribute a shared-memory tile across lanes with the correct pattern, issuing `mma_sync`, and then calling `store_matrix_sync` to reassemble the output. Get the layout wrong — say, load a column-major tile into a row-major fragment — and the result is silently incorrect.

![GPU tensor core register layout: threads own scattered fragments of the tile](../assets/images/ch04/fig2_register_loading_dark.png#only-dark)
![GPU tensor core register layout: threads own scattered fragments of the tile](../assets/images/ch04/fig2_register_loading_light.png#only-light)

*Simplified view of how 32 threads in a warp own scattered register fragments of an MMA tile. The exact pattern is architecture-specific and deliberately opaque.*

Croktile's design sidesteps this complexity entirely. Instead of exposing architecture-specific fragment types, it provides **four abstract operations** that work on opaque register tiles: **fill**, **load**, **multiply**, and **store**. These operations describe the same 2D contraction workflow regardless of which hardware backend runs them. The compiler handles fragment layouts, lane mappings, and instruction selection for the target architecture — you describe *what* contraction to perform, not *how* registers are scattered.

![Croktile's four-step MMA syntax: fill, load, multiply, store](../assets/images/ch04/fig3_mma_syntax_dark.png#only-dark)
![Croktile's four-step MMA syntax: fill, load, multiply, store](../assets/images/ch04/fig3_mma_syntax_light.png#only-light)

*The four-step MMA syntax is an abstract interface — not hardwired to GPU tensor cores. Any DSA that supports 2D tile contraction can map to these operations.*

## The four-step MMA syntax

Every tensor-contraction kernel follows the same rhythm:

1. **`mma.fill 0.0`** — allocate an accumulator tile `mc` in registers and zero it.
2. **`mma.load`** — bring operand tiles from shared memory into opaque MMA registers `ma` and `mb`.
3. **`mma.row.row mc, ma, mb`** — issue the contraction: **C += A × B** into `mc`.
4. **`mma.store mc, dst`** — write `mc` from registers into shared memory.

You loop steps 2–3 over K (loading the next K-slice each iteration, accumulating into the same `mc`), then run step 4 once to flush the completed output tile. The names `mc`, `ma`, and `mb` are opaque register tiles — you do not declare per-lane layouts; the compiler derives them from the target and your layout choice (`row.row` here).

The `.row.row` suffix is a **layout contract** — it tells the hardware how to interpret the bits in `ma` and `mb`. Both operands are row-major. If B is stored column-major in shared memory, you write `mma.row.col mc, ma, mb` instead. The full set of layout combinations is `row.row`, `row.col`, `col.row`, and `col.col`; in practice, `row.row` and `row.col` cover most kernels. Choosing the wrong variant is a correctness bug, not a performance hint — the hardware interprets register bits differently for each layout.

## Scaling the cooperation scope

The four-step syntax stays the same regardless of how many threads cooperate on a single contraction. What changes is the **cooperation scope** — one warp, one warpgroup, two warpgroups — and that progression is the story of this section.

### One warp, one tile: the simplest MMA matmul

On Ampere (SM86), tensor-core MMA is scoped to a **single warp** (32 threads). In Croktile, that corresponds to `: group`. Here is a complete FP16 matmul kernel where every `MATMUL_*` constant is 16, so one block tile equals one MMA tile:

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, MATMUL_TILE_M), cdiv(N, MATMUL_TILE_N)] : block {
    shared f16 [MATMUL_TILE_M, MATMUL_TILE_N] output_s;

    parallel {warp_m, warp_n} by [cdiv(MATMUL_TILE_M, MATMUL_MMA_M), cdiv(MATMUL_TILE_N, MATMUL_MMA_N)] : group {
      mc = mma.fill 0.0;

      foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
        lhs_load_s = dma.copy lhs.subspan(MATMUL_TILE_M, MATMUL_TILE_K).at(block_m, iv_k) => shared;
        rhs_load_s = dma.copy rhs.subspan(MATMUL_TILE_N, MATMUL_TILE_K).at(block_n, iv_k) => shared;

        foreach iv_warp_k in [cdiv(MATMUL_TILE_K, MATMUL_MMA_K)] {
          ma = mma.load lhs_load_s.chunkat(warp_m, iv_warp_k);
          mb = mma.load rhs_load_s.chunkat(warp_n, iv_warp_k);
          mma.row.row mc, ma, mb;
        }
      }

      mma.store mc, output_s.subspan(MATMUL_MMA_M, MATMUL_MMA_N).at(warp_m, warp_n);
    }

    dma.copy output_s => output.subspan(MATMUL_TILE_M, MATMUL_TILE_N).at(block_m, block_n);
  }
}
```

**`__co__ void` and in-place output.** The kernel returns nothing; results go through `output`, matching the usual GPU pattern of writing through a global pointer.

**Block grid.** `cdiv(M, MATMUL_TILE_M)` is ceiling division — how many tiles along M, including partial tiles. `block_m` and `block_n` pick which output tile this block owns.

**Warp grid and `mma.fill`.** `parallel {warp_m, warp_n} ... : group` maps MMA tiles to warps. With all dimensions 16, extents are 1×1 — one warp covers the whole block tile. Wider block tiles would add warps, each with its own `mc`.

**K loop and DMA.** Each `iv_k` stage pulls A and B strips into shared memory via `dma.copy` with `subspan(...).at(...)`. Chapter 7 goes deeper on `subspan` versus `chunkat`.

**Operand loads.** `mma.load` moves the warp's tile from shared memory into `ma` / `mb`. `chunkat(warp_m, iv_warp_k)` selects the M×K slice for this warp and inner-K step.

**Store and epilogue.** After K completes, `mma.store` writes `mc` into the warp's sub-rectangle of `output_s`, then `dma.copy` sends the full block tile to global memory.

This kernel is simple because the cooperation scope is narrow: 32 threads, one warp, one MMA tile at a time. The four-step syntax reads linearly and the tile geometry is obvious. But what happens when the hardware offers a wider cooperation window?

### Widening the scope: warpgroups and WGMMA

Hopper (SM90) adds **Warpgroup Matrix Multiply-Accumulate (WGMMA)**: the same C += A × B contraction, but issued cooperatively by **four warps** (128 threads). The hardware instruction is wider, the tiles are bigger, and throughput improves — but the four-step syntax does not change. The only thing that changes in Croktile is the space specifier: `: group-4` instead of `: group`.

![Ampere vs Hopper MMA cooperation scope](../assets/images/ch04/fig4_sm86_vs_sm90_dark.png#only-dark)
![Ampere vs Hopper MMA cooperation scope](../assets/images/ch04/fig4_sm86_vs_sm90_light.png#only-light)

*SM86: one warp per MMA. SM90: four warps (`: group-4`) cooperate on WGMMA.*

```choreo
parallel {block_m, block_n} by [cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)] : block {
  shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
  shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;

  mc = mma.fill.f16 0.0f;

  foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
    dma.copy lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
    dma.copy rhs.chunkat(block_n, iv_k) => rhs_load_s;

    foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
      parallel p by 1 : group-4 {
        ma = mma.load lhs_load_s.chunkat(_, iv_warp);
        mb = mma.load rhs_load_s.chunkat(_, iv_warp);
        mma.row.row mc, ma, mb;
      }
    }
  }

  shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;
  mma.store mc, output_s;
  dma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
}
```

Read it side by side with the SM86 kernel. Fill, load, multiply, store — the rhythm is identical. The differences are mechanical:

**`mma.fill.f16 0.0f`.** Hopper often spells accumulator precision explicitly — `.f16`, `.f32`, etc. FP16 operands with FP32 accumulation is a common pattern for long K dimensions, avoiding numerical overflow in partial sums. SM86 commonly uses the shorter `mma.fill 0.0` and relies on inference.

**`parallel p by 1 : group-4`.** One warpgroup (four warps) executes the inner loads and MMA. The mnemonic `mma.row.row` matches Ampere, but the hardware issue is wider.

**`chunkat(_, iv_warp)`.** `_` means "do not tile that dimension" — keep the full M (or N) extent already resident in shared memory; only K is subdivided per MMA slice.

That is the whole point of the abstraction: the same four operations, the same layout contract, the same `chunkat` / `subspan` expressions. The compiler maps them to different hardware instructions depending on whether the target is SM86 or SM90. You think about *what* contraction to perform; the cooperation width is a deployment detail.

### Tiling further: two warpgroups sharing operands

Chapter 3 introduced `parallel p1 by 2 : group-4` — two warpgroups in one block. With MMA, both groups can share the same B tile while loading different rows of A. This is how large block tiles get split into multiple MMA tiles without duplicating the B operand in shared memory:

```choreo
parallel {block_m, block_n} by [cdiv(M, MATMUL_TILE_M), cdiv(N, MATMUL_WARP_N)] : block {
  shared f16 [MATMUL_TILE_M, MATMUL_TILE_K] lhs_load_s;
  shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
  shared f16 [MATMUL_TILE_M, MATMUL_WARP_N] output_s;

  mc = mma.fill.f32 0.0f;

  foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
    dma.copy lhs.subspan(MATMUL_TILE_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
    dma.copy rhs.chunkat(block_n, iv_k) => rhs_load_s;

    parallel p1 by 2 : group-4 {
      foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
        ma = mma.load lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(p1, 0).chunkat(_, iv_warp);
        mb = mma.load rhs_load_s.chunkat(_, iv_warp);
        mma.row.row mc, ma, mb;
      }
    }
  }

  parallel p1 by 2 : group-4 {
    mma.store mc, output_s.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(p1, 0);
  }
  dma.copy output_s => output.subspan(MATMUL_TILE_M, MATMUL_WARP_N).at(block_m, block_n);
}
```

**Splitting M with `p1`.** With `MATMUL_TILE_M = 128` and `MATMUL_WARP_M = 64`, the block spans 128 rows; `p1` selects the upper or lower 64-row strip. `lhs_load_s.subspan(MATMUL_WARP_M, ...).at(p1, 0)` gives each warpgroup its A rows; both use the same `rhs_load_s`.

**Mirrored store.** `mma.store` targets `output_s.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(p1, 0)` so each warpgroup writes its half of the staging buffer, then one `dma.copy` emits the combined tile.

The pattern scales: three warpgroups, four warpgroups, or any count that divides your block tile. The four-step syntax stays invariant; only the parallel decomposition of the tile changes.

| Aspect | One warp (`: group`) | One warpgroup (`: group-4`) | Two warpgroups |
|--------|---------------------|----------------------------|----------------|
| Threads | 32 | 128 | 256 |
| Accumulator | `mma.fill 0.0` | `mma.fill.f16 0.0f` | `mma.fill.f32 0.0f` |
| Tile split | One MMA tile per warp | One MMA tile per warpgroup | Block tile split across warpgroups |
| Operand sharing | N/A | N/A | B tile shared, A rows split by `p1` |

## Beyond dense FP16: what else the four steps express

The examples above use `mma.row.row` on dense FP16 tiles. The same four-step pattern extends to workloads the basic form cannot reach.

**Structured sparsity.** When half the elements of A follow a 2:4 zero pattern (Ampere and later), the hardware can skip the zero products and roughly double throughput — but it needs a **metadata operand** `me` that encodes which elements are nonzero:

```choreo
mma.row.row.sp mc, ma, mb, me;
```

The `.sp` suffix adds the metadata operand; everything else is the same fill-load-multiply-store rhythm. Any layout combination works: `mma.row.col.sp`, etc. You load `me` from a separate metadata tensor alongside A and B.

**Quantized operands with per-tile scaling.** FP8 operands (`f8_e4m3`, `f8_e5m2`) need per-tile dequantization so the accumulator stays numerically accurate. Croktile fuses the scaling into the contraction:

```choreo
mma.row.row.scale mc, ma, mb, sc_a, sc_b;
```

Each result element is scaled by `sc_a` and `sc_b` after the contraction — no separate dequant kernel needed. Alternatively, scaling can be a **standalone statement** when the scale source differs from the standard fused path:

```choreo
mma.row.row mc, ma, mb;
mma.scale mc, sc_a, sc_b;
```

The standalone `mma.scale` appears in some MoE and mixed-precision kernels.

**Swizzled loads and transposed stores.** When shared memory uses a swizzle pattern to avoid bank conflicts ([Chapter 7](ch07-advanced-movement.md)), the MMA load must use the matching swizzle mode: `mma.load.swiz<N>`. The `<N>` must agree between `tma.copy.swiz<N>` and `mma.load.swiz<N>` — a mismatch reads garbage. For output, `mma.store.transp mc, dst` writes the accumulator with rows and columns swapped, useful when the next stage expects column-major data.

**Pipeline synchronization.** In pipelined kernels where producer and consumer warpgroups overlap ([Chapter 6](ch06-synchronization.md)), `mma.commit` marks the boundary between "done reading this K-slab's operands" and "safe to reuse the shared-memory buffer." It is mandatory glue in event-driven pipelines.

These extensions all follow the same design principle: the four-step skeleton stays fixed, and the variant suffix communicates a specific contract to the hardware. The table below collects every variant for reference.

## New syntax

| Syntax | Meaning |
|--------|---------|
| `mc = mma.fill 0.0` | Initialize accumulator tile to zero |
| `mma.fill.f16 0.0f` / `mma.fill.f32 0.0f` | Accumulator with explicit precision |
| `ma = mma.load src.chunkat(...)` | Load operand tile from shared into MMA registers |
| `mma.load.swiz<N> src` | Load with swizzle mode (see [Ch 7](ch07-advanced-movement.md)) |
| `mma.row.row mc, ma, mb` | C += A × B (both row-major) |
| `mma.row.col mc, ma, mb` | C += A × B (A row-major, B col-major) |
| `mma.row.row.sp mc, ma, mb, me` | Sparse MMA with metadata operand |
| `mma.row.row.scale mc, ma, mb, sc_a, sc_b` | Fused MMA + per-tile dequantization |
| `mma.scale mc, sc_a, sc_b` | Standalone post-MMA scaling |
| `mma.store mc, dst` | Write accumulator to shared memory |
| `mma.store.transp mc, dst` | Write accumulator transposed |
| `mma.commit` | Pipeline stage fence for WGMMA (see [Ch 6](ch06-synchronization.md)) |
| `cdiv(a, b)` | Ceiling division |
| `__co__ void fn(...)` | Kernel that writes results in-place |

## Chapter summary

| Topic | Takeaway |
|-------|----------|
| 2D tensor contraction | C += A × B on tile-shaped operands — the universal inner kernel |
| Hardware diversity | GPU tensor cores, TPU MXU, Intel AMX, custom DSAs all implement this; tile sizes and register layouts differ |
| Four-step abstraction | **fill → load → multiply → store**; the compiler handles fragment layouts for each target |
| Scaling cooperation | `: group` (one warp) → `: group-4` (one warpgroup) → multiple warpgroups — syntax stays invariant |
| Layout contract | `mma.row.row`, `mma.row.col`, etc. — must match data in shared memory |
| Sparse and quantized | `.sp` adds a metadata operand; `.scale` fuses per-tile dequantization |
| Swizzle and pipeline | `mma.load.swiz<N>` matches swizzled shared layouts; `mma.commit` fences pipeline stages |

The contraction is fast, but loads and compute still take turns — while tensor cores multiply, the memory system idles. The [next chapter](ch05-branch-control.md) introduces **role specialization** so different thread groups can play different roles: one group loading data while another computes, overlapping memory and math in time.
