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

## Full listing audit

Every `croqtile` listing in `build/units/*.tex` is compiled standalone against
the front end (`choreo -t cc -es`). There are 222 listings; 199 are CroqTile and
23 are host C++, which the CroqTile front end does not apply to.

| Result | Count |
|---|---|
| compiles clean | 186 |
| flagged | 13 |

A listing counts as flagged only when the compiler emits a diagnostic in its
`file.co:line.col: error:` form. Matching the word "error" anywhere in the
output is wrong: the compiler echoes the offending source line, so a listing
whose comment reads `// error: ...` looks like a failure when it compiles.

Nothing is excused merely for being labelled a fragment. A listing is only
allowed to fail when the failure is an unterminated construct, which happens
exactly once (F03-S04). Everything else is either a placeholder the author wrote
on purpose, a target limitation, a compiler defect, or a defect in the listing.

| Cause | Listings | Verdict |
|---|---|---|
| unterminated `__co__` signature | F03-S04 | intentional fragment |
| ellipsis placeholder `...` | F14-S03, F14-S04, F18-S06, F19-S03, F25-S03, F25-S04 (x2), F31-S08 | intentional abbreviation |
| deliberate error example | F24-S03 | intentional |
| bare statement outside any `__co__` body, so the front end never sees it | F16-S03, F17-S02, F17-S05, F17-S06 | fragment; each does report the documented error once placed inside a function |
| `: group` unsupported on x86_64 | F06-S04 | needs a GPU target; the text now says so |
| undefined macro plus a third parallel level | F10-S04 | open, see below |

### Compiler defects found

Three listings crashed the CPU backend rather than failing it. Both crashes
reduce to a few lines and are fixed in the compiler, not in the guide:

- A bounded integer derived by arithmetic (`st = k % STAGES`) used as an array
  subscript hit `unsupported CC type: int->[0,2]`; the backend mapped the
  bounded tuple but not the bounded integer.
- A future seeded with `dma.any` and later reassigned in a loop, with `swap`
  between the two, hit `MapHostSymbol: symbol existed`; the placeholder and the
  copy that replaces it name the same buffer, which is not a conflict.

After the fix, F08-S02, F08-S10 and F38-S01 all compile clean.

### Fixed in this pass

| Listing | Defect |
|---|---|
| F19-S04 | declares `tile_m` twice and uses an undeclared `tile_n` — the same text as F19-S05, which had already been fixed |
| F24-S04 | passes `.data` of an mdspan parameter and of a declared buffer; `.data` belongs to a future, so both calls pass the buffer itself |
| F17-S02 | calls a device function outside any `parallel-by` block |
| F04-S03, F29-S03, F29-S04, F29-S05 | move one tile per instance without saying so, so the compiler reads the instances as lanes of one thread and every lane copies the whole tile into the same buffer; annotated `: block` |
| F07-S08, F07-S09 | use `WARP_M`, `WARP_N`, `TILE_K`, `WARP_K` (and `NUM_GROUPS`) without defining them |
| F25-S04 | a stray comma ends a statement |
| F10-S04 | uses `N_TOKENS`, `N_EXPERTS` and `N_THREAD` without defining them |
| F18-S03 | claimed `s32 a;` is rejected; it compiles to a control integer, and the diagnostic the comment quotes no longer exists in the compiler |
| F24-S04 | said the scalar integer type "only includes `int`"; `s32` and `u32` are accepted too |
| F24-S06 | said a runtime template argument is an error; the compiler now reports a warning |
| F23-S02 | said a macro with parameters "would not work"; parameterized macros expand, including inside a shape expression |
| F15-S20 | said one `__co__` function may not call another; the compiler warns that it cannot find a device function, and the program still builds |
| F32-S02 | said a `parallel-by` variable cannot be used in `foreach`; it can bound a loop, it just cannot be the iterated variable |
| F29-S05 | said a DMA requires source and destination of the same shape; the compiler only rejects a source dimension larger than the destination, and a smaller source is accepted |
| F19-S06 | the comment promised an error the code did not produce: `f32 [M]` compiles once `M` is known. The listing now shows a derived buffer next to a rejected anonymous one, so the comment matches what the compiler reports |
| F40-S03 | staged whole matrices in shared storage (8 MB); the outer tiling is now `[4, 4, 8]`, so each staging buffer is one tile |

### Open

- **F10-S04** nests three parallel levels, where the CPU target supports two.
  The macros it used to leave undefined are now defined; what remains is a
  target limitation, not a defect in the listing.
- **F40-S03** declares a `[512, 1024]` s32 staging buffer, which is far past the
  shared-memory budget, so the compiler rejects it before any other check. The
  prose calls the listing a structural illustration; fixing it means rescaling
  the tile bounds, which changes the numbers the surrounding text quotes.

### A second target

`choreo -t gpu -es` is not usable: the NVPTX path runs CoIR rather than AST
codegen. `-t cute -es` does work, so the same audit was repeated there:
175 compile clean against 24 flagged. The extra failures are CUDA-target
capabilities the CPU target does not model — an unsupported MMA shape, local
buffers past the GPU's limit, and async copies the GPU target does not allow
in those directions. None of them contradicts the CPU-target result.

## Coverage

`build/units/*.tex` contains 199 `croqtile` listings. Filtering to listings that
contain `__co__` and no `...` yields **40 "complete" blocks**.

| Verdict | Count | Meaning |
|---|---|---|
| `ok` | 13 | no diagnostic from the front end |
| `FAIL` | 23 | rejected by the target with a diagnostic |
| `CRASH` | 4 | compiler abort or segfault |

No published block compiles and runs unmodified on GCU. `verdict.txt` has the
per-block reason.

### The 13 that pass are mostly skeletons

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
| teaching examples that must fail | F03-S04, F15-S21, F24-S03, F24-S04, F24-S06-2, F19-S06 | expected |

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
| F15-S21, F19-S05, F23-S02, F25-S04 | runtime named `croqtile` throughout | renamed to `croq` (see below) |
| F19-S05 | `int tile_m = 32, tile_m = 8, tile_k = 16;` declares `tile_m` twice and leaves `tile_n` undefined | `tile_m`, `tile_n`, `tile_k` |
| F03-S03, F05-S16 | `croq::croq_assert` — the helper is a free function | `croq_assert` |
| F15-S05 | `__co__ int foo(int b);` — a `__co__` function must carry a body; only `__device__` helpers may be declared without one | `__co__ int foo(int b) { return b; }` |
| F23-S02 | left operand declared `[M, N]` but the caller passes an `[M, K]` buffer, so the shape check fails at entry | `f32 [M, K] lhs` |
| F24-S06 | the block computed a `[16, 4]` tile, wrote it nowhere, and indexed only the row blocks — just the diagonal of the tile grid was covered | loop over `{p, q}`, write the tile back with `dma.copy buffer => output.chunkat(p, q)`, and mark the movement `: block` |
| F19-S05 | the listing claims to quote compiler output but used the guide's spelling, a non-existent assert helper, and by-value parameters | `choreo::spanned_view<choreo::s32, 2> &`, `choreo::runtime_check`, aligned by hand |

Two of the fixes turned compiler crashes into ordinary diagnostics (F06-S04,
F40-S03), which is how the remaining problems became visible. F40-S03 now clears
the front end entirely.

**Still open in the listings:** F07-S08 deliberately omits the producer/consumer
handoff and says so in prose; F14-S08, F38-S01 and F40-S06 call kernels the
guide never defines.

## Runtime naming

The library ships as `choreo`, aliased as `croq`; the guide uses the short
alias. The guide had spelled the header, the namespace, the assertion helper
and a factory helper `croqtile`, none of which exist under that name. Round 81
renames them everywhere: code listings, prose, the editorial rule files, both
translation trees and the glossary.

| Guide spelling | Actual spelling |
|---|---|
| `croqtile.h` | `choreo.h` (the header file has no alias) |
| `croqtile::` | `croq::` |
| `croqtile_assert` | `croq_assert` (a free function — no namespace qualifier) |
| `croqtile::make_spanned` | `croq::make_spanview` |
| `croqtile::span_view<2, T>` | `croq::spanned_view<T, 2>` |

Verified against the shipped runtime: `croq::s32`, `croq::make_spanview<2>`,
`croq::spanned_view<T, 2>`, `croq::runtime_check` and the global `croq_assert`
all compile; `croqtile::` does not resolve at all. The old
`croq::croq_assert` spelling used by two listings was also wrong, since the
assertion helper is not a member of the namespace.

The `croqtile` spelling that remains in the guide is the language name itself
and the `language=croqtile` listing tag, both intentional.

Note: code the compiler itself emits says `choreo::` (for example
`choreo::runtime_check(...)`). The one listing that quotes transpiled output,
F19-S05, therefore keeps the compiler's spelling rather than the guide's; every
listing that shows code a reader would write uses `croq::`.

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
