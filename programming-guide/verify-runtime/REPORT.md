# Runtime verification of the guide's complete code blocks

Target: `choreo -t topscc -arch=gcu300`, executed on the local GCU device.
Compiler: `/home/wsj/choreo/build/choreo`.

## How to reproduce

```sh
./verdict.sh   # target-check every published block, write verdict.txt
./run.sh       # build and run every harness, numeric compare
```

`blocks/` is regenerated from `build/units/*.tex` after each `build_guide.py`
run; it holds the listings exactly as the PDF prints them.

## Coverage

`build/units/*.tex` contains 199 `croqtile` listings. Filtering to listings that
contain `__co__` and no `...` yields **40 "complete" blocks**.

| Verdict | Count | Meaning |
|---|---|---|
| `ok` | 12 | no diagnostic from the front end |
| `FAIL` | 24 | rejected by the target with a diagnostic |
| `CRASH` | 4 | compiler abort or segfault |

No published block compiles and runs unmodified on GCU. `verdict.txt` has the
per-block reason.

### The 12 that pass are mostly skeletons

`F10-S11`, `F15-S03`, `F26-S01` (x2), `F33-S03`, `F34-S07`, `F35-S03` have
comment-only bodies. `F11-S02` only prints. `F14-S08`, `F38-S01`, `F40-S03` and
`F40-S06` call kernels the block never defines.

## Failure categories

| Category | Blocks | Diagnostic |
|---|---|---|
| direct global element access | F02-S04, F11-S03, F31-S10, F36-S02, F36-S03 | `global data access 'a[p]' is not allowed` |
| parallel-by count over the target limit | F03-S03, F05-S13, F05-S14, F08-S02, F33-S04, F34-S08 | `total thread/block parallel-by count must not exceed N` |
| thread-indexed DMA | F04-S03, F24-S06-1, F29-S03, F29-S04, F29-S05 | `a thread-level DMA with a thread-indexed source must index the thread-level parallel variable on the destination` |
| MMA operand in shared storage | F08-S10 | `MMA load on GCU requires source in LOCAL (L1) storage` |
| `: group` level absent on this target | F06-S04 | `group level is not supported by the target architecture` |
| teaching examples that must fail | F03-S04, F15-S05, F15-S21, F24-S03, F24-S04, F24-S06-2, F19-S06 | expected |

### Compiler crashes

| Block | Symptom |
|---|---|
| F07-S08 | `Assertion (vty->Dims() != 0) && IsValidRank(vty->Dims())` |
| F07-S09 | `no type for an init expression` |
| F10-S04 | abort, no message |
| F19-S06 | segfault (139) |

These are compiler defects, not guide defects, but a reader pasting the block
sees a crash instead of a diagnostic.

## Code fixes applied to the guide

Round 80 in `tools/build_guide.py` (plus matching `editorial/narrative-polish.json`
prose edits). Each fix is target-independent — a reader on any backend gets
correct code.

| Block | Problem | Fix |
|---|---|---|
| F05-S13 | `+=` into a declared span that is never cleared | add a zeroing loop in the same parallel instance |
| F05-S14 | same, across the K loop | zero at thread level before the `tile_k` loop |
| F08-S02 | same, before the double-buffer prologue | zero at the top of the thread body |
| F07-S08 | `void` kernel with an output parameter never returns its result | `auto`, declare and return `output` |
| F07-S09 | same | same |
| F08-S10 | same | same |
| F06-S04 | `MMA_M` / `MMA_N` / `MMA_K` used but never defined | `#define` them alongside the other tile constants |
| F24-S04 | `M` is a symbolic extent, not a name in scope | read it as `input.span(0)` |
| F15-S21 | `#include "croqtile.h"` and `croqtile::` — neither exists | `choreo.h` and `croq::` |
| F40-S03 | `out_store_s` read after the loop that assigns it | seed with `dma.any` before the loop |
| F40-S03 | async copies never awaited | `wait lhs_load_s, rhs_load_s;` before the dot product, `wait out_store;` before returning |
| F15-S21, F19-S05, F23-S02, F25-S04 | runtime named `croqtile` throughout | renamed to `choreo` (see below) |

Two of the fixes turned compiler crashes into ordinary diagnostics (F06-S04,
F40-S03), which is how the remaining problems became visible. F40-S03 now clears
the front end entirely.

**Still open in the listings:** F07-S08 deliberately omits the producer/consumer
handoff and says so in prose; F14-S08, F38-S01 and F40-S06 call kernels the
guide never defines.

## Runtime naming

The library ships as `choreo` (aliased `croq`). The guide had spelled the
header, the namespace, the assertion helper and a factory helper `croqtile`,
none of which exist under that name. Round 81 renames them everywhere: code
listings, prose, the editorial rule files, both translation trees and the
glossary.

| Guide spelling | Actual spelling |
|---|---|
| `croqtile.h` | `choreo.h` |
| `croqtile::` | `choreo::` |
| `croqtile_assert` | `choreo_assert` |
| `croqtile::make_spanned` | `choreo::make_spanview` |
| `croqtile::span_view<2, T>` | `choreo::spanned_view<T, 2>` |

The `croqtile` spelling that remains in the guide is the language name itself
and the `language=croqtile` listing tag, both intentional.

## Harnesses

Seven blocks were ported to GCU-legal form and given a host `main` that fills
random inputs, calls the kernel and compares against a CPU reference.

| Harness | Source block | Result |
|---|---|---|
| `01-identity` | F02-S04 | `bad=0` |
| `02-ele-add` | F03-S03 | `bad=0` |
| `03-tiled-add-2d` | F04-S03 | `bad=0` |
| `04-chunkat-roundtrip` | F29-S05 | `bad=0` |
| `05-mma-matmul` | F06-S04 | `bad=0 maxerr=3.9e-3` |
| `06-dma-matmul` | F05-S14 | `bad=0` |
| `07-mma-load-form` | F06-S04 (`mma.load` spelling) | `bad=0 maxerr=3.9e-3` |

The ports change only what the target forces: stage global reads through
`local`/`shared` with `dma.copy`, and replace wide `parallel-by` grids with
`foreach` inside one thread. Tile grids, `chunkat` arithmetic, accumulation
order and the `mma.load` spelling are unchanged.

## MMA finding

MMA results compared against a CPU reference over a range of operand shapes:

| M | N | K | Result |
|---|---|---|---|
| 64 | 64 | 128 | exact |
| 128 | 128 | 128 | exact |
| 128 | 128 | 64 | exact |
| 64 | 64 | 256 | exact |
| 64 | 64 | 64 | **wrong** |
| 64 | 64 | 192 | **wrong** |
| 32 | 32 | 32 | **wrong** |
| 32 | 32 | 128 | **wrong** |
| 96 | 96 | 128 | rejected (`96` is not a supported extent) |

The operand tiles must cover whole micro-tiles; a partially covered tile
silently produces wrong numbers rather than being rejected.

Both MMA read paths verify identically: passing the local buffer straight to
`mma.row.row`, and the guide's `ma = mma.load lhs_l` spelling.
