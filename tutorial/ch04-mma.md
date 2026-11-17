# Tensor Contraction: the `mma` Syntax

Chapter 3's tiled matmul computes the inner contraction the way a CPU would: a `foreach k` loop where each iteration multiplies two scalars and accumulates into one output element. That works. It is also the slowest way to use a modern accelerator, because it ignores the unit designed specifically for this job.

The job is **2D tensor contraction** — the operation C += A × B where A, B, and C are small, fixed-shape matrix tiles. It is the inner kernel of every GEMM, every convolution expressed as an im2col matmul, every attention head's QK^T. The operation is central enough that accelerators execute it as one macro-instruction: take a tile of A, a tile of B, multiply, accumulate into C. The tile sizes differ between implementations, but the mathematical shape is the same everywhere.

![2D tensor contraction: A[M,K] times B[K,N] into C[M,N]](../assets/images/ch04/fig1_tensor_contraction_dark.png#only-dark)
![2D tensor contraction: A[M,K] times B[K,N] into C[M,N]](../assets/images/ch04/fig1_tensor_contraction_light.png#only-light)

*The same mathematical operation — C += A × B on tile-shaped operands — maps onto different implementations.*

What makes this hard for programmers is not the math but the **operand layout**. The tile is not stored contiguously in one thread's storage; it is **fragmented** across the cooperating lanes, and the exact scatter pattern depends on the datatype and on whether the operand is A, B, or C. Get the layout wrong — say, load a column-major tile into a row-major fragment — and the result is silently incorrect.

![Operand fragment layout: lanes own scattered fragments of the tile](../assets/images/ch04/fig2_register_loading_dark.png#only-dark)
![Operand fragment layout: lanes own scattered fragments of the tile](../assets/images/ch04/fig2_register_loading_light.png#only-light)

*Simplified view of how the cooperating lanes own scattered fragments of an MMA tile. The exact pattern is implementation-specific and deliberately opaque.*

Croqtile's design sidesteps this complexity entirely. Instead of exposing implementation-specific fragment types, it provides **four abstract operations** that work on opaque tiles: **fill**, **load**, **multiply**, and **store**. These operations describe the same 2D contraction workflow regardless of which backend runs them. The compiler handles fragment layouts, lane mappings, and instruction selection — you describe *what* contraction to perform, not *how* the operands are scattered.

![Croqtile's four-step MMA syntax: fill, load, multiply, store](../assets/images/ch04/fig3_mma_syntax_dark.png#only-dark)
![Croqtile's four-step MMA syntax: fill, load, multiply, store](../assets/images/ch04/fig3_mma_syntax_light.png#only-light)

*The four-step MMA syntax is an abstract interface — not hardwired to any one contraction unit. Any backend that supports 2D tile contraction can map to these operations.*

## The four-step MMA syntax

Every tensor-contraction kernel follows the same rhythm:

1. **`mma.fill 0.0`** — allocate an accumulator tile `mc` in registers and zero it.
2. **`mma.load`** — bring operand tiles from shared storage into opaque MMA tiles `ma` and `mb`.
3. **`mma.row.row mc, ma, mb`** — issue the contraction: **C += A × B** into `mc`.
4. **`mma.store mc, dst`** — write `mc` out into shared storage.

You loop steps 2–3 over K (loading the next K-slice each iteration, accumulating into the same `mc`), then run step 4 once to flush the completed output tile. The names `mc`, `ma`, and `mb` are opaque register tiles — you do not declare per-lane layouts; the compiler derives them from the target and your layout choice (`row.row` here).

The `.row.row` suffix is a **layout contract** — it states how the elements of `ma` and `mb` are arranged. Both operands are row-major. If B is stored column-major in shared storage, you write `mma.row.col mc, ma, mb` instead. The full set of layout combinations is `row.row`, `row.col`, `col.row`, and `col.col`; in practice, `row.row` and `row.col` cover most kernels. Choosing the wrong variant is a correctness bug, not a performance hint — each layout is interpreted differently.

## Scaling the cooperation scope

The four-step syntax stays the same regardless of how many instances cooperate on a single contraction. What changes is the **cooperation scope** — how many lanes form one cooperating group — and that is what this section varies.

### One group, one tile: the simplest MMA matmul

The smallest example assigns one output tile to a cooperative group, written `: group`. It accepts left and right operands plus an output buffer supplied by the caller. Follow one output tile through the code: initialize its accumulator, visit every K slice, store the accumulated tile in shared storage, and copy it to its output position. The `: group` level names a fixed cooperating group; a target that has no such level rejects it, so read this listing against a target that provides one.

```choreo
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16
#define WARP_M 16
#define WARP_N 16
#define WARP_K 16
#define MMA_M 16
#define MMA_N 16
#define MMA_K 16

__co__ void matmul_wmma(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, TILE_M), cdiv(N, TILE_N)] : block {
    shared f16 [TILE_M, TILE_N] output_s;

    parallel {warp_m, warp_n} by [cdiv(TILE_M, MMA_M), cdiv(TILE_N, MMA_N)] : group {
      mc = mma.fill 0.0;

      foreach {iv_k} in [cdiv(K, TILE_K)] {
        lhs_load_s = dma.copy lhs.subspan(TILE_M, TILE_K).at(block_m, iv_k) => shared;
        rhs_load_s = dma.copy rhs.subspan(TILE_N, TILE_K).at(block_n, iv_k) => shared;

        foreach iv_warp_k in [cdiv(TILE_K, MMA_K)] {
          ma = mma.load lhs_load_s.chunkat(warp_m, iv_warp_k);
          mb = mma.load rhs_load_s.chunkat(warp_n, iv_warp_k);
          mma.row.row mc, ma, mb;
        }
      }

      mma.store mc, output_s.subspan(MMA_M, MMA_N).at(warp_m, warp_n);
    }

    dma.copy output_s => output.subspan(TILE_M, TILE_N).at(block_m, block_n);
  }
}
```

**`__co__ void` and in-place output.** The kernel returns nothing; results go through `output`, matching the usual pattern of writing back in place.

**Block grid.** `cdiv(M, TILE_M)` is ceiling division — how many tiles along M, including partial tiles. `block_m` and `block_n` pick which output tile this block owns.

**Group grid and `mma.fill`.** `parallel {warp_m, warp_n} ... : group` maps MMA tiles to cooperating groups. With all dimensions 16, extents are 1×1 — one group covers the whole block tile. Wider block tiles would add groups, each with its own `mc`.

**K loop and DMA.** Each `iv_k` stage pulls A and B strips into shared storage via `dma.copy` with `subspan(...).at(...)`. Chapter 7 goes deeper on `subspan` versus `chunkat`.

!!! note "Quick `subspan` syntax reminder"
    `subspan(h, w)` declares a fixed-size rectangular view shape, and `.at(i, j)` selects which view instance you want from the parent tensor. In other words, `lhs.subspan(TILE_M, TILE_K).at(block_m, iv_k)` means "take the `[TILE_M, TILE_K]` tile at `(block_m, iv_k)`." Use `subspan` when you want an explicit tile shape; use `chunkat` when you want to slice by equal chunks of an existing dimension.

**Operand loads.** `mma.load` moves the group's tile from shared storage into `ma` / `mb`. `chunkat(warp_m, iv_warp_k)` selects the M×K slice for this group and inner-K step.

**Store and epilogue.** After K completes, `mma.store` writes `mc` into the group's sub-rectangle of `output_s`, then `dma.copy` sends the full block tile to global storage.

This kernel is simple because the cooperation scope is narrow: one group, one MMA tile at a time. The four-step syntax reads linearly and the tile geometry is obvious. A wider cooperation scope changes the count of groups, not the operations.

## Beyond dense FP16: what else the four steps express

The examples above use `mma.row.row` on dense FP16 tiles. The same four-step pattern extends to workloads the basic form cannot reach.

**Structured sparsity.** When half the elements of A follow a fixed sparse zero pattern, the backend can skip the zero products — but it needs a **metadata operand** `me` that encodes which elements are nonzero:

```choreo
mma.row.row.sp mc, ma, mb, me;
```

The `.sp` suffix adds the metadata operand; everything else is the same fill-load-multiply-store rhythm. Any layout combination works: `mma.row.col.sp`, etc. You load `me` from a separate metadata tensor alongside A and B.

**Quantized operands with per-tile scaling.** FP8 operands (`f8_e4m3`, `f8_e5m2`) need per-tile dequantization so the accumulator stays numerically accurate. Croqtile fuses the scaling into the contraction:

```choreo
mma.row.row.scale mc, ma, mb, sc_a, sc_b;
```

Each result element is scaled by `sc_a` and `sc_b` after the contraction — no separate dequant kernel needed. Alternatively, scaling can be a **standalone statement** when the scale source differs from the standard fused path:

```choreo
mma.row.row mc, ma, mb;
mma.scale mc, sc_a, sc_b;
```

The standalone `mma.scale` appears in some MoE and mixed-precision kernels.

**Swizzled loads and transposed stores.** When a staging buffer uses a swizzle pattern ([Chapter 7](ch07-advanced-movement.md)), the MMA load must use the matching swizzle mode: `mma.load.swiz<N>`. The `<N>` must agree between `dma.copy.swiz<N>` and `mma.load.swiz<N>` — a mismatch reads garbage. For output, `mma.store.transp mc, dst` writes the accumulator with rows and columns swapped, useful when the next stage expects column-major data.

**Pipeline synchronization.** In pipelined kernels where a producer and a consumer region overlap ([Chapter 6](ch06-synchronization.md)), `mma.commit` marks the boundary between "done reading this K-slab's operands" and "safe to reuse the staged buffer." It is mandatory glue in event-driven pipelines.

These extensions all follow the same design principle: the four-step skeleton stays fixed, and the variant suffix communicates a specific contract to the backend. The table below collects every variant for reference.

## New syntax

| Syntax | Meaning |
|--------|---------|
| `mc = mma.fill 0.0` | Initialize accumulator tile to zero |
| `mma.fill.f16 0.0f` / `mma.fill.f32 0.0f` | Accumulator with explicit precision |
| `ma = mma.load src.chunkat(...)` | Load operand tile from shared into MMA registers |
| `mma.load.swiz<N> src` | Load with swizzle mode (see Chapter 7) |
| `mma.row.row mc, ma, mb` | C += A × B (both row-major) |
| `mma.row.col mc, ma, mb` | C += A × B (A row-major, B col-major) |
| `mma.row.row.sp mc, ma, mb, me` | Sparse MMA with metadata operand |
| `mma.row.row.scale mc, ma, mb, sc_a, sc_b` | Fused MMA + per-tile dequantization |
| `mma.scale mc, sc_a, sc_b` | Standalone post-MMA scaling |
| `mma.store mc, dst` | Write accumulator to shared storage |
| `mma.store.transp mc, dst` | Write accumulator transposed |
| `cdiv(a, b)` | Ceiling division |
| `__co__ void fn(...)` | Kernel that writes results in-place |

## Chapter summary

| Topic | Takeaway |
|-------|----------|
| 2D tensor contraction | C += A × B on tile-shaped operands — the universal inner kernel |
| Backend diversity | Different backends implement the contraction differently; tile sizes and operand layouts differ |
| Four-step abstraction | **fill → load → multiply → store**; the compiler handles fragment layouts for each target |
| Scaling cooperation | More cooperating groups per tile — the four operations stay invariant |
| Layout contract | `mma.row.row`, `mma.row.col`, etc. — must match the staged data |
| Sparse and quantized | `.sp` adds a metadata operand; `.scale` fuses per-tile dequantization |
| Swizzle and pipeline | `mma.load.swiz<N>` matches swizzled shared layouts; `mma.commit` fences pipeline stages |

The contraction is fast, but loads and compute still take turns — while the contraction unit works, the memory path idles. The [next chapter](ch05-branch-control.md) introduces **role specialization** so different regions can play different roles: one group loading data while another computes, overlapping memory and math in time.
