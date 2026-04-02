# C++ Interop: Inline Code and the Preprocessor

Every high-level language needs an escape hatch to the layer below. Python has `ctypes` and C extensions. Rust has `unsafe` blocks and `extern "C"`. Java has JNI. The reason is always the same: no matter how expressive the abstraction, some hardware feature, some legacy library, some performance-critical intrinsic lives outside the language's domain. The escape hatch lets you reach it without abandoning the rest of the system.

Croktile's escape hatch is **`__cpp__`**: inject verbatim C++ or PTX into the generated code. The preprocessor — `#define`, `#if`, `#ifdef`, `#error` — serves the complementary role of **compile-time configuration**, the same role `#define` plays in C/C++ codebases: sharing constants between the DSL body and host code, guarding platform-specific paths, and catching configuration mistakes before they become runtime bugs.

So far the tutorial has built a stack of abstractions: tiles, parallelism, MMA, pipelines, events. That is the Croktile you want to live in. This chapter is about the times you need to step outside — a single PTX instruction the compiler does not emit, a register-budget hint for warp-specialized roles, or a compile-time constant that must match in both the `__co__` body and host-side C++.

![Croktile kernel body with an embedded __cpp__ island for verbatim PTX/C++](../assets/images/ch08/fig1_escape_hatch_dark.png#only-dark)
![Croktile kernel body with an embedded __cpp__ island for verbatim PTX/C++](../assets/images/ch08/fig1_escape_hatch_light.png#only-light)

## `__cpp__`: Verbatim C++ Injection

**What it does.** `__cpp__` takes a string literal and pastes it, character for character, into the generated CUDA or C++ file. Whatever you place there must be valid at the splice point: braces, semicolons, types, and scope must agree with the surrounding codegen. The Croktile compiler does not parse or rewrite the contents, and Croktile symbols are not magically visible inside the string — only names that actually appear in the generated output.

**Two forms.**

- **`__cpp__("...")`** — Ordinary string; best for short one-liners and simple guards.
- **`__cpp__(R"(...)")`** — Raw string literal; use this for `asm volatile("...")` and other fragments where escaping every `"` would be painful.

### Register hints: `setmaxnreg`

The canonical `__cpp__` use case is **register redistribution** in warp-specialized pipelines ([Chapter 5](ch05-branch-control.md)). The producer warpgroup is register-light — it mostly issues TMA loads — while the consumer is register-heavy — it holds MMA accumulators. NVIDIA’s PTX `setmaxnreg` instruction moves the register budget between those roles:

```choreo
parallel p1 by 2 : group-4 {
  inthreads.async (p1 == 0) {
    __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");)");
    // producer: register-light, decrease to 40
    foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
      // ... TMA loads ...
    }
  }

  inthreads.async (p1 == 1) {
    __cpp__(R"(asm volatile("setmaxnreg.inc.sync.aligned.u32 216;");)");
    // consumer: register-heavy, increase to 216
    mc = mma.fill.f16 0.0f;
    // ... WGMMA compute ...
  }
}
```

**`setmaxnreg.dec` / `setmaxnreg.inc`** — The exact counts (here 40 and 216) are tuned per kernel. The `.dec` and `.inc` forms cooperate with the hardware allocator so both roles can coexist without unnecessary spilling.

**Placement** — Put the hint at the top of each `inthreads.async` branch, before the heavy loop, so the allocator sees the budget before the body runs.

### Early returns and guards

MoE-style GEMM kernels often process variable-sized expert segments; some launches have zero width for a given expert. A one-line `__cpp__` injection can bail out before the rest of the kernel runs:

```choreo
__cpp__("if (seg_end - seg_start <= 0) return;\n\n");
```

**Name discipline** — The identifiers (`seg_end`, `seg_start`) must match what the surrounding generated code declares. If you rename a Croktile parameter and codegen changes its spelling, a stale `__cpp__` string fails at compile time — which is preferable to silently wrong results.

### Macros inside `__cpp__` strings

A common mistake is to assume the preprocessor will help inside the injected string:

```choreo
#define PRODUCER_MAXNREG 40

inthreads.async (p1 == 0) {
  __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 PRODUCER_MAXNREG;");)");
}
```

**Why it fails** — The preprocessor does not expand macros inside string literals. The generated asm would still contain the identifier `PRODUCER_MAXNREG`, not `40`, and PTX would reject it.

**What teams do instead** — Type numeric literals inside `__cpp__` strings, and use `#if` / `#error` *outside* the strings to enforce consistency with your tile configuration:

```choreo
#define PRODUCER_MAXNREG 40
#if PRODUCER_MAXNREG > 50
#error "Producer maxnreg too high for this tile config"
#endif

inthreads.async (p1 == 0) {
  __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");)");
}
```

## The Preprocessor

Croktile’s preprocessor runs before the main compiler pass. **`#define`**, **`#if` / `#elif` / `#else` / `#endif`**, **`#ifdef` / `#ifndef`**, and **`#error`** behave like their C counterparts. Macros expand in both `__co__` regions and host C++ in the same `.co` file, so one definition can keep tile geometry and host-side checks aligned.

**Directive reference.**

| Directive | Role |
|-----------|------|
| `#define NAME value` | Object-like macro: textual replacement |
| `#if` / `#elif` / `#else` / `#endif` | Conditional inclusion |
| `#ifdef` / `#ifndef` | Shorthand for whether a macro is defined |
| `#error message` | Force a compile-time failure with a message |

**Limitation** — Function-like macros (`#define MAX(a, b) ...`) are not supported. Use `constexpr` helpers in ordinary C++ when you need parameterized expressions.

### Tile geometry as macros

Production matmul sources usually centralize every tile dimension at the top of the file:

```choreo
#define MATMUL_WARP_M 64
#define MATMUL_WARP_N 128
#define MATMUL_TILE_K 64
#define MATMUL_WARP_K 16
#define MATMUL_STAGES 4
```

**Shared contract** — The same names appear in `parallel ... by [cdiv(M, MATMUL_WARP_M), ...]`, shared-memory declarations, `foreach` bounds, and host-side verification. Change one `#define`, and every use site updates together.

### Compile-time assertions with `#error`

Libraries often pair `#if` with `#error` so illegal combinations fail at compile time with a clear message:

```choreo
#if MATMUL_SWIZ != (2 * MATMUL_TILE_K)
#error "MATMUL_SWIZ must equal 2 * MATMUL_TILE_K for f16 kernel"
#endif

#if MATMUL_WARP_M != 64
#error "MATMUL_WARP_M must be 64 for f16 WGMMA constraints"
#endif
```

**Living documentation** — When someone changes swizzle width or warp tile incompatibly, the build stops immediately instead of emitting an illegal kernel. Treat these guards as documentation of hardware constraints, not one-off checks.

### Conditional code paths

You can select entire regions from a macro:

```choreo
#define PATH0

__co__ foo() {
  // ...
  #ifdef PATH0
    // path 0 code
  #else
    // path 1 code
  #endif
}
```

**Compile-time elimination** — The preprocessor keeps one branch and discards the other before Croktile parses the `__co__` body. This is not a runtime `if`; the discarded branch never appears in the generated program.

**Command-line defines** — `croktile kernel.co -DMATMUL_TILE_K=128` defines or overrides macros without editing the source — useful for benchmark sweeps over tile sizes without duplicating files.

## Reading a Production `.co` File

When you open a benchmark kernel such as `blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c.co`, read **top down**:

1. **Macros and `#error` guards** — The contract: allowed tile sizes, swizzle rules, architecture flags.
2. **Host setup** — Buffers, launch configuration, timing; ordinary C++.
3. **The `__co__` function** — Orchestration: `parallel`, `foreach`, TMA/MMA, `inthreads.async`, events. Map each region back to earlier chapters.
4. **`__cpp__` islands** — Usually a handful of lines. Pause on each and ask what the hardware receives that the DSL does not spell.

That order keeps you from diving into a warp-specialized loop before you know which constants you are allowed to change.

## Chapter Summary

| Topic | Takeaway |
|-------|----------|
| When to escape | PTX instructions, register hints, host/device constants, guards — use the DSL everywhere it fits; drop down sparingly. |
| `__cpp__` | Verbatim paste into generated CUDA; raw strings for `asm`; names must match generated C++. |
| `setmaxnreg` | Canonical example: dec on producer, inc on consumer; tune per kernel; place before heavy loops. |
| Guards / macros in strings | Early `return` via `__cpp__`; macros do **not** expand inside string literals — use literals + `#error` outside. |
| Preprocessor | `#define` for tile geometry; `#if` / `#error` for constraints; `#ifdef` for variants; `-D` for sweeps. |
| Reading `.co` files | Macros first, then host, then `__co__`, then `__cpp__` islands. |

**New syntax**

| Syntax | Meaning |
|--------|---------|
| `__cpp__("...")` | Inject verbatim C++ (ordinary string) |
| `__cpp__(R"(...)")` | Inject verbatim C++ (raw string literal) |
| `#define NAME value` | Object-like macro |
| `#if expr` / `#elif` / `#else` / `#endif` | Conditional compilation |
| `#ifdef NAME` / `#ifndef NAME` | Test whether a macro is defined |
| `#error "message"` | Compile-time assertion failure |

The escape hatches close the loop: Croktile keeps everyday kernels readable and structured, while `__cpp__` and the preprocessor handle the hardware-specific details that sit below the abstraction. Prefer a single PTX hint, a single guard, or a single pragma over scattering raw islands through the program — let the `__co__` function own the story.

The [next chapter](ch09-debug-verbose.md) turns to the other side of the workflow: what to do when the kernel compiles but the output is wrong — debugging, verbose modes, and systematic narrowing.
