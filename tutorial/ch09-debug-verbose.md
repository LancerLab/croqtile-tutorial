# Debug and Verbose: Printing, RTTI, and GDB

GPU kernels are **opaque systems**. Thousands of threads execute concurrently, shared memory is invisible from the host, and when results are wrong, there is no stack trace pointing to the bad line. This opacity is not unique to GPUs — any system where the programmer cannot directly observe intermediate state faces the same challenge: distributed systems (where is the lost message?), embedded firmware (which interrupt handler corrupted the register?), optimizing compilers (which pass broke the semantics?).

The universal debugging discipline is the same everywhere: **systematically narrow the search space**, from cheap checks to expensive ones. Start with static analysis and compile-time assertions (free, catches whole classes of bugs). Move to targeted runtime probes (cheap, localizes the problem). Resort to interactive debuggers only when the cheaper tools have narrowed the suspect list. The cost of each level increases, so you want to catch as many bugs as possible at the cheapest level.

Croktile provides tools at each level: **compile-time shape printing** (`print!`, `println!`) to verify tile dimensions without launching a kernel, **guarded device `println`** for runtime inspection of specific threads, **`choreo_assert`** for invariant checks, **debug RTTI** for `cuda-gdb`, and a clear **order of suspicion** that guides where to look first.

So far you have walked the full Croktile stack — element-wise addition, data movement, parallelism, tensor cores, warp specialization, pipelines, TMA, and C++ escape hatches. Each step added performance and complexity. This chapter is about what to do when the results are wrong.

## `print!` and `println!`: compile-time inspection

The bang variants (`print!`, `println!`) run at **compile time**, not at runtime. They print to the compiler’s output and are useful for inspecting shapes, extents, and type information **without launching a kernel**:

```choreo
__co__ void check_shapes(f32 [3, 2] b) {
  print!("shape of b: ");
  println!("b.span = ", b.span);
}
```

When you compile this file, the compiler emits:

```
shape of b: b.span = {3, 2}
```

No GPU needed. String literals are concatenated (`"wor" "ld!"` becomes `"world!"`), which is handy for building diagnostic messages from fragments. Use `print!` / `println!` to verify that `chunkat` and `subspan` produce the tile sizes you expect before you spend time on a full kernel launch.

## `print` and `println`: runtime device output

**Runtime printing.** `print` writes its arguments to standard output; `println` does the same but appends a newline. Both work inside `__co__` functions and emit device-side `printf` calls in the generated CUDA:

```choreo
__co__ void inspect(s32 [4, 8] data) {
  foreach i in [data.span] {
    println("element ", i, " = ", data.at(i));
  }
}
```

Each `println` call takes a comma-separated mix of string literals and expressions. Strings are printed verbatim; expressions are evaluated and printed as their runtime value. The output order across threads is **nondeterministic** — GPU `printf` buffers are flushed asynchronously, so lines from different threads interleave unpredictably.

**Guarding output.** For a specific check — say, whether tile index `(3, 5)` computes the right partial sum — guard the print with a condition:

```choreo
parallel {px, py} by [8, 16] : block
  parallel {qx, qy} by [16, 16] : thread {
    // ... compute ...
    if (px == 0 && py == 0 && qx == 3 && qy == 5) {
      println("partial sum = ", accum);
    }
  }
```

Without the guard you get one line per thread — thousands of lines for a large grid, most of which you do not care about.

## Debugging MMA-heavy kernels

When the wrong answer comes from a **tensor-core path**, treat it as its own category. **Suspect layout first** — row versus column major, and whether the RHS is logically `[N, K]` versus `[K, N]` in memory — **then indexing** (which `block_m`, `block_n`, and K slice you attached with `.at` / `chunkat`), **then async ordering** if you introduced asynchronous copies or split producer and consumer warps. A common mistake is **mislabeling `mma.row.row`** when the staged data is actually column-major, or using **`chunkat` indices that do not align** with the MMA tile geometry. If you use swizzled loads (`tma.copy.swiz` / `mma.load.swiz`), a mismatch against plain row expectations shows up as a **regular pattern** in the error (for example every sixteenth element correct, the rest shifted) — the systematic workflow below calls that out explicitly.

## Debug RTTI and `cuda-gdb`

When `print` / `println` is not enough — you need to step through execution, inspect register state, or examine complex data structures — Croktile supports **debug RTTI** (Runtime Type Information) that makes its types visible to `cuda-gdb`.

**Build flags.** Compile with `-g -O0` to enable debug symbols and disable optimizations:

```bash
croktile kernel.co -g -O0 -o kernel_debug
```

The generated code includes RTTI structs for Croktile types:

| Croktile type | GDB type | Fields |
|--------------|----------|--------|
| Shaped tensor (`s32 [M, N]`) | `choreo::rtti::spanned<int, 2>` | `.span.data[]` (dimensions), `.stride.data[]` (strides), `.data` (pointer) |
| Index tuple | `choreo::rtti::bounded_ituple<N>` | `.data[]` (values), `.ub[]` (upper bounds) |
| Integer tuple | `choreo::rtti::ituple<N>` | `.data[]` (values) |
| Multi-dim span | `choreo::rtti::mdspan<N>` | `.data[]` (extents) |

**Example session.** A GDB session on a debug-compiled kernel:

```bash
cuda-gdb -q ./kernel_debug
(gdb) break my_kernel
(gdb) run
(gdb) ptype __dbg_lhs
type = struct choreo::rtti::spanned<int, 2>
(gdb) print __dbg_lhs.span.data[0]
$1 = 32
(gdb) print __dbg_lhs.span.data[1]
$2 = 64
(gdb) print __dbg_lhs.data != 0
$3 = true
```

The `__dbg_` prefix on variable names is generated by the compiler — it makes Croktile variables visible to GDB alongside the generated C++ intermediates. You can inspect tensor dimensions, strides, and data pointers, and step through the generated code to see which iteration of a `foreach` loop produces a wrong value.

## Putting it together: a systematic workflow

The following order is deliberate: **cheap compile-time checks first**, then **narrowed runtime prints**, then **pipeline semantics**, **layout**, and finally **the debugger** for pointer-level bugs.

![Debugging workflow: shapes → one tile → sync → layout → GDB](../assets/images/ch09/fig1_debug_workflow_dark.png#only-dark)
![Debugging workflow: shapes → one tile → sync → layout → GDB](../assets/images/ch09/fig1_debug_workflow_light.png#only-light)

**1. Check shapes.** Use `println!` at compile time to verify that all `chunkat`, `subspan`, and `span` expressions produce the tile sizes you expect. Shape bugs are the most common category — a mismatched extent silently reads or writes wrong memory regions.

**2. Check one tile.** Add a guarded `println` inside the K loop that fires only for block `(0, 0)` and one thread. Print the accumulator value after each K iteration. Compare with a hand-computed reference for that specific output element.

**3. Check synchronization.** If the value is wrong only at large K, suspect event ordering. Print `iv_k` and `stage` in both producer and consumer to verify they visit the same sequence. A common bug: the consumer’s `trigger empty[stage]` fires too early, allowing the producer to overwrite a buffer the consumer is still reading.

**4. Check layout.** If the result is wrong in a pattern (e.g., every sixteenth element is correct, the rest are shifted), suspect a row-major vs column-major mismatch in `mma.row.row` or a swizzle mode inconsistency between `tma.copy.swiz<N>` and `mma.load.swiz<N>`. Cross-check with the MMA subsection above.

**5. Use GDB for pointer bugs.** If `println` shows a value that makes no sense (NaN, huge integer, zero when nonzero is expected), compile with `-g -O0` and step through in `cuda-gdb`. Inspect the tensor pointer (`__dbg_x.data`) and check whether it points to valid memory.

**Performance before benchmarks.** `print` / `println` are expensive: each call serializes through a global `printf` buffer and destroys throughput measurements. Debug RTTI structs add register pressure and inhibit optimization. Remove all prints and drop `-g -O0` before benchmarking. A practical compromise is to keep prints behind a `#ifdef DEBUG_PRINT` macro (Chapter 8) so you can toggle them from the command line without editing the source:

```choreo
#ifdef DEBUG_PRINT
  println("tile_k=", iv_k, " accum=", mc);
#endif
```

Compile with `croktile kernel.co -DDEBUG_PRINT` to enable, or without the flag for production runs.

## Summary

| Construct / tool | Role |
|------------------|------|
| `print!` / `println!` | Compile-time shape and type inspection; no kernel launch |
| `print` / `println` | Device-side `printf`; use guards to avoid log floods |
| MMA debugging | Layout → indexing → async ordering; watch `mma.*` vs actual memory order |
| `-g -O0`, `cuda-gdb`, RTTI | Inspect `__dbg_*` tensors, pointers, and control flow |
| `#ifdef DEBUG_PRINT` | Optional printf without permanent overhead |

You started from **element-wise addition** in Chapter 1, learned **data movement** (Chapter 2), **parallelism** (Chapter 3), **tensor cores** (Chapter 4), **warp specialization** (Chapter 5), **pipelines** (Chapter 6), **TMA** (Chapter 7), **C++ interop** (Chapter 8), and **debugging** (Chapter 9). The next productive step is to open a benchmark kernel from the `croktile/benchmark/` directory, map its regions to the chapters here, change one constant, rebuild, and measure. Small, deliberate edits beat large rewrites — the `#error` guards will tell you when a configuration stops being legal long before you chase mysterious wrong answers on the GPU.
