# Advanced Data Movement: Swizzle and Irregular Access

Chapter 2 introduced `dma.copy` as Croqtile's general data movement primitive — a way to move rectangular tiles between storage levels using a simple `src => dst` arrow syntax. This chapter covers the two things that chapter left for later: the **swizzled layouts** a contraction consumer needs, and the **irregular-access** utilities that reach tiles the regular tiling cannot describe.

Both keep the same statement form. A swizzle is a modifier on the copy; an irregular access is a different way to name the source region.

## Swizzle and layout conflicts

A swizzle changes how the elements of a staged tile are arranged in shared storage. The logical matrix shape stays the same, but the reader must use the matching address mapping. Croqtile exposes this choice on the copy operation:

```choreo
dma.copy.swiz<128> src => dst;
```

The copy places the tile using swizzle pattern `N`. The MMA read path must use the same `swiz<N>` so addresses match the staged layout:

```choreo
ma = mma.load.swiz<128> lhs_load_s.chunkat(_, iv_warp);
```

Swizzle is a property of the copy, not of a particular engine: `dma.copy.swiz<N>` fixes the shared-storage layout, and the consumer has to agree with it.

**Swizzle width.** The template argument is the swizzle width in bytes, and the examples use `swiz<128>`. A wider swizzle defeats wider conflict patterns but requires tile extents that line up with that width.

**Matching rule.** The `<N>` on the copy must match `mma.load.swiz<N>`. If you load with plain `mma.load` from `swiz<128>` data, addresses disagree and you read garbage. The compiler does not enforce the pairing — it is a correctness invariant you maintain. (As introduced in [Chapter 4](ch04-mma.md#new-syntax), `mma.load.swiz<N>` is part of the MMA load family.)

![Layout conflicts without swizzle vs a swizzle spreading accesses](../assets/images/ch07/fig2_swizzle_dark.png#only-dark)
![Layout conflicts without swizzle vs a swizzle spreading accesses](../assets/images/ch07/fig2_swizzle_light.png#only-light)

## Why the restricted interface works: expressiveness vs performance

In a language that exposes raw addressing, a programmer implementing data movement has enormous freedom: arbitrary pointer arithmetic, variable-stride access, hand-computed conflict avoidance, custom swizzle formulas. That flexibility is a double-edged sword. The space of possible data movement patterns is vast, but the subset that actually performs well is narrow — it requires coalesced reads, conflict-free staged access, and correct swizzle alignment. Getting any of these wrong silently degrades throughput by an order of magnitude.

Croqtile takes the opposite approach: it restricts the expressible patterns to those that are **guaranteed to be in the performance sweet spot**. When you write `dma.copy`, the compiler handles coalescing, conflict-free layout, and swizzle alignment. There is no way to accidentally write a strided, uncoalesced read or a conflicted staged layout — the syntax simply does not allow it.

![Expressiveness vs performance: the unconstrained range includes many slow patterns; Croqtile's restricted range maps entirely to the performance sweet spot](../assets/images/ch07/fig8_expressiveness_dark.png#only-dark)
![Expressiveness vs performance: the unconstrained range includes many slow patterns; Croqtile's restricted range maps entirely to the performance sweet spot](../assets/images/ch07/fig8_expressiveness_light.png#only-light)

The trade is explicit: you give up the ability to write arbitrary data movement, and in return, every data movement you write is fast.

## Handling irregular access

The tiling primitives introduced so far — `chunkat`, `subspan(...).at(...)` — assume tensors that divide evenly into tiles. This works for textbook GEMM where M, N, K are multiples of the tile dimensions. Production kernels are rarely so tidy. Expert batches start at dynamic offsets. Convolution windows overlap. The last K-tile is almost never an exact multiple of TILE_K. Some inputs arrive as flat buffers that need reshaping before MMA can consume them.

Croqtile provides four tools for these cases. Each one modifies how the compiler interprets a tensor's addressing without touching the pipeline structure — DMA, MMA, events and swizzle all continue to work unchanged.

### Arbitrary-offset windows: `view` and `from`

`chunkat` divides a tensor into a regular grid of equal-sized tiles. But what if the slice you need does not start on a tile boundary? Consider a mixture-of-experts (MoE) kernel: each expert processes a different number of tokens, so the starting row for each expert's operand is determined at runtime. You cannot precompute a fixed tile grid.

`view(M, N).from(row, col)` defines an `M x N` rectangle starting at any `(row, col)` — no alignment requirement:

```choreo
patch = matrix.view(16, 16).from(37, 50);
```

This is a `[16, 16]` slice starting at row 37, column 50. The origin can be any runtime value.

![chunkat (aligned grid) vs view/from (arbitrary offset window)](../assets/images/ch07/fig4_view_from_dark.png#only-dark)
![chunkat (aligned grid) vs view/from (arbitrary offset window)](../assets/images/ch07/fig4_view_from_light.png#only-light)

The power of `view` / `from` is composability: once you carve out a window, all downstream operations — `subspan`, `chunkat`, `dma.copy` — apply to the window as if it were a standalone tensor. The pipeline stays unchanged; only the origin shifts:

```choreo
expert_lhs = lhs.view(expert_M, K).from(expert_offset, 0);
dma.copy expert_lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k) => shared;
```

In an MoE stack, each expert's token batch starts at a dynamic row `expert_offset`. The `view` / `from` call rewires the operand at the top of the pipeline; everything below — DMA, staging, MMA loop — runs the same code for every expert.

**When to use it.** Prefer `chunkat` when the tensor divides evenly into tiles at compile time. Use `view` / `from` when the window's origin or extent is determined at runtime, or when the geometry does not align to the tile grid.

### Strided tiles: `.subspan`, `.step`, and `.at`

`subspan(M, K).at(i, j)` selects the tile at logical tile indices `(i, j)` with extent `[M, K]`. By default, tiles are packed contiguously: tile `(1, 0)` starts immediately after tile `(0, 0)`. But some layouts space tiles apart — either because there is padding between rows, or because a stencil kernel needs overlapping windows.

Adding `.step(sM, sK)` overrides the stride: tiles are spaced `sM` rows and `sK` columns apart instead of being packed:

```choreo
matrix.subspan(16, 16).step(32, 32).at(i, j);
```

![Packed tiling vs strided tiling with .step](../assets/images/ch07/fig5_subspan_step_dark.png#only-dark)
![Packed tiling vs strided tiling with .step](../assets/images/ch07/fig5_subspan_step_light.png#only-light)

Tile `(0,0)` starts at `(0,0)`, but tile `(1,0)` starts at row 32 (not row 16), leaving a 16-row gap. Omitting `.step` uses a step equal to the tile size — the packed case.

**When you need this.** Skipping padding or guard bands between rows. Overlapping stencil windows where the step is smaller than the tile extent (e.g., a `16 x 16` tile sliding by 8 rows). Matching an outer memory layout that is not dense tile-major — for instance, when tiles are interleaved with metadata.

### Zero-padding: `.zfill`

Every tiled kernel faces the same edge case: what happens when M or K is not a multiple of the tile size? The last tile along that axis is partial — some of its elements lie beyond the tensor boundary. Reading those addresses is undefined behaviour.

The brute-force fix is to add a special code path for the boundary tile. But this breaks the uniformity of the MMA loop and adds branches that the compiler cannot optimize away. `.zfill` solves this elegantly:

```choreo
dma.copy.swiz<128> lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k).zfill
  => lhs_load_s;
```

`.zfill` applies to the source side of a copy: out-of-range elements are written as zero in the destination tile. For a GEMM, zeros contribute nothing to the accumulation (0 × anything = 0), so the MMA loop stays completely uniform — same code for interior tiles and boundary tiles — while remaining mathematically correct. No branches, no special cases, no performance penalty.

![.zfill: zero-padding partial tiles at the tensor boundary](../assets/images/ch07/fig6_zfill_dark.png#only-dark)
![.zfill: zero-padding partial tiles at the tensor boundary](../assets/images/ch07/fig6_zfill_light.png#only-light)

### Layout reinterpretation: `span_as`

Some kernels load data as a flat 1D strip but need to feed it to MMA as a 2D matrix. A naive approach would copy the data into a reshaped buffer — wasting staging space and traffic. `span_as` avoids this by reinterpreting the existing buffer's shape in place:

```choreo
flat_buffer.span_as([rows, cols])
```

Element count is preserved; only the logical rank changes. No data is moved.

```choreo
strip_load = dma.copy data.chunkat(tile) => shared;
tile_2d = strip_load.data.span_as([tile_m, tile_k]);
ma = mma.load tile_2d.chunkat(_, iv_warp);
```

The loaded 1D strip is exposed as a 2D matrix for `chunkat` and MMA operand loading, without an extra copy or additional staging storage. `rows * cols` must equal the span length of the underlying storage, or the compiler rejects the program — this invariant is checked at compile time, not at runtime.

![span_as: zero-copy shape reinterpretation from 1D to 2D](../assets/images/ch07/fig7_span_as_dark.png#only-dark)
![span_as: zero-copy shape reinterpretation from 1D to 2D](../assets/images/ch07/fig7_span_as_light.png#only-light)

## Chapter summary

| Concept                 | Syntax                                  | Role                                                                        |
|-------------------------|-----------------------------------------|-----------------------------------------------------------------------------|
| Tile transfer (Ch. 2, 6) | `dma.copy` / `dma.copy.swiz<N>`        | Thread-cooperative tile transfer                                            |
| Swizzle                 | `.swiz<N>` on copy + `mma.load.swiz<N>` | Swizzled shared-storage layout shared by copy and consumer                  |
| Expressiveness trade    | —                                       | Croqtile restricts patterns to guarantee coalesced, conflict-free transfers |
| Arbitrary windows       | `view(M,N).from(r,c)`                   | Ragged or runtime-positioned slices                                         |
| Strided tiling          | `.subspan().step().at()`                | Non-packed layouts, overlapping stencils                                    |
| Partial tiles           | `.zfill`                                | Zero-fill out-of-bounds elements                                            |
| Shape reinterpretation  | `span_as([dims])`                       | Zero-copy reshape for staging buffers                                       |

## New syntax

| Syntax                                 | Meaning                                                  |
|----------------------------------------|----------------------------------------------------------|
| `dma.copy.swiz<N> src => dst`          | Copy with swizzle width `N` in bytes                     |
| `mma.load.swiz<N> src`                 | MMA operand load consistent with swizzle `N`             |
| `.view(M, N).from(r, c)`         | Arbitrary-offset `M x N` window                          |
| `.subspan(M, K).step(sM, sK).at(i, j)` | Strided tile selection                                   |
| `.zfill`                               | Zero-fill out-of-bounds elements on copy source          |
| `span_as(dims)`                        | Reinterpret linear storage as shaped tensor              |

The [next chapter](ch08-cpp-interop.md) steps past pure Croqtile into **C++ interop**: `__device__` functions, **register hints**, **preprocessor guards**, and **inline PTX** when you need to drop down to the metal.
