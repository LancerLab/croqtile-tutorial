# Runtime verification of the guide's complete code blocks

Target: `choreo -t topscc -arch=gcu300`, executed on the local GCU device.
Compiler: `/home/wsj/choreo/build/choreo`.

## How to reproduce

```sh
./verdict.sh          # target-check every published block, write verdict.txt
./run.sh              # build and run every harness, numeric compare
```

## Coverage

`build/units/*.tex` contains 199 `croqtile` listings. Filtering to listings that
contain `__co__` and no `...` yields **40 "complete" blocks**. They were extracted
verbatim into `blocks/`.

| Verdict | Count | Meaning |
|---|---|---|
| `ok` | 11 | no diagnostic from the front end |
| `FAIL` | 23 | rejected by the target with a diagnostic |
| `CRASH` | 6 | compiler abort or segfault |

So **no published "complete" block compiles and runs unmodified on GCU**.
`verdict.txt` has the per-block reason.

### The 11 that pass are mostly skeletons

`F10-S11`, `F15-S03`, `F26-S01` (x2), `F33-S03`, `F34-S07`, `F35-S03` have
comment-only bodies or empty bodies. `F11-S02` only prints. `F14-S08`,
`F38-S01` and `F40-S06` call external kernels (`call kernel(...)`,
`call dot_general_fp32(...)`, `call rgb2gray_kernel_fp32_fp32(...)`) that the
block never defines.

## Failure categories

| Category | Blocks | Diagnostic |
|---|---|---|
| direct global element access | F02-S04, F11-S03, F31-S10, F36-S02, F36-S03 | `global data access 'a[p]' is not allowed` |
| parallel-by count over the target limit | F03-S03, F05-S13, F05-S14, F08-S02, F33-S04, F34-S08 | `total thread/block parallel-by count must not exceed N` |
| thread-indexed DMA | F04-S03, F24-S06-1, F29-S03, F29-S04, F29-S05 | `a thread-level DMA with a thread-indexed source must index the thread-level parallel variable on the destination` |
| MMA operand in shared storage | F08-S10 | `MMA load on GCU requires source in LOCAL (L1) storage` |
| undefined symbols | F06-S04, F07-S08, F07-S09, F10-S04, F19-S06, F24-S06-2 | `symbol 'WARP_M' is used before declaration` etc. |
| teaching examples that must fail | F03-S04, F15-S05, F15-S21, F24-S03, F24-S04 | expected |

### Compiler crashes

| Block | Symptom |
|---|---|
| F06-S04 | `Assertion (vty->Dims() != 0) && IsValidRank(vty->Dims())` |
| F07-S08 | same assertion |
| F07-S09 | `no type for an init expression` |
| F10-S04 | abort, no message |
| F19-S06 | segfault (139) |
| F40-S03 | `unable to find symbol 'out_store_s'` (also seen to pass once — nondeterministic) |

These are compiler defects, not guide defects, but a reader pasting the block
sees a crash instead of a diagnostic.

## Harnesses

Seven blocks were ported to GCU-legal form and given a host `main` that fills
random inputs, calls the kernel, and compares against a CPU reference.

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

MMA results were compared against a CPU reference over a range of operand shapes:

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
silently produces wrong numbers rather than being rejected. The `16 x 16`
tiles used by the guide's first MMA example fall in the wrong band.

Both MMA read paths verify identically: passing the local buffer straight to
`mma.row.row`, and the guide's `ma = mma.load lhs_l` spelling.
