# C++ Interop: Device Functions, Inline Code, and the Preprocessor

Every high-level language needs an escape hatch to the layer below. Python has `ctypes` and C extensions. Rust has `unsafe` blocks and `extern "C"`. Java has JNI. The reason is always the same: no matter how expressive the abstraction, some hardware feature, some legacy library, some performance-critical intrinsic lives outside the language's domain.

Croqtile's interoperability story has three parts:

1. **`__device__` functions** — device functions that coexist with `__co__` functions in the same `.co` file. The compiler passes them through unchanged.
2. **`__cpp__`** — inject verbatim C++ into the generated code for target-specific instructions the DSL does not emit.
3. **The preprocessor** — `#define`, `#if`, `#ifdef`, `#error` for compile-time configuration, the same role `#define` plays in C/C++ codebases.

The preceding sections introduced tiles, parallelism, MMA, pipelines, and events. That is the Croqtile you want to live in. This chapter is about the times you need to step outside.

## `__device__` vs ordinary C++ inline

Both `__device__` functions and ordinary C++ functions are passed through unchanged, but they land in different compilation paths — and the distinction matters.

A `__device__` function compiles to **device machine code**. It can use target intrinsics (`__shfl_xor_sync`, `__frsqrt_rn`, `atomicAdd`), access `__shared__` storage, and run inside a `__global__` kernel. The `__co__` body calls it with the `call` keyword, and the generated kernel invokes it directly on the device.

An ordinary C++ function (no `__device__` or `__co__` annotation) compiles to **CPU code**. It runs on the host — setting up buffers, launching kernels, printing results. You cannot call a host function from inside a `__co__` body; the compiler would reject it because the generated kernel runs on the device.

The boundary is strict: `__device__` lives on the device side of the split, host C++ lives on the host side. The `__co__` function is the bridge — the compiler transforms it into a device kernel and generates a host-side wrapper that launches it.

## `__device__` functions

When an algorithm needs a helper function that operates at the lane or thread level — a custom reduction, a sorting network, a special math function — writing it as a device function is often the natural choice. Croqtile does not need to manage these; they are passed through to the target compiler.

A `__co__` function can call a `__device__` function using the `call` keyword:

```choreo
#define N_TOKENS 1024
#define N_EXPERTS 8
#define N_THREAD 128
template <int K>
__device__ void warp_topk(f32* vals, s32* idxs);

template <typename T>
__device__ __forceinline__ T SHFL_XOR(T var, int lane_mask, int width) {
  return __shfl_xor_sync(uint32_t(-1), var, lane_mask, width);
}

__co__ void moe_topk(f32 [N_TOKENS, N_EXPERTS] scores,
                     s32 [N_TOKENS, K]& topk_ids,
                     f32 [N_TOKENS, K]& topk_scores,
                     int N_BLOCK) {
  parallel n by N_BLOCK : block {
    foreach m in [ |scores.span| / N_THREAD / N_BLOCK ] {
      shared_scores = dma.copy scores.chunkat(n#m, _) => shared;

      parallel gid by [N_THREAD / 32] : group
        parallel tid by 32 : thread {
          flat_scores = shared_scores.span_as([|shared_scores.span|]);
          score = flat_scores.at(gid # tid);

          local s32 [K] frag_idx{-1};
          local f32 [K] frag_val{-1.0f};
          frag_idx.at(0) = tid;
          frag_val.at(0) = score;

          call warp_topk<8>(frag_val, frag_idx);

          inthreads (tid == 0)
            foreach k in K {
              topk_ids.at(n#m, gid#k) = frag_idx.at(k);
              topk_scores.at(n#m, gid#k) = frag_val.at(k);
            }
        }
    }
  }
}
```

**`__device__ void warp_topk<K>`** — A templated device function implementing group-level top-K selection using shuffle instructions. It operates on raw pointers (`f32*`, `s32*`), not Croqtile spans.

**`__device__ T SHFL_XOR`** — A lane-shuffle intrinsic wrapper. The `__forceinline__` and `__shfl_xor_sync` are device code — Croqtile passes them through unchanged.

**`call warp_topk<8>(...)`** — The `call` keyword invokes a `__device__` function from within a `__co__` body. Arguments are passed by pointer; the compiler handles the address translation between Croqtile spans and raw pointers.

**`inthreads (tid == 0)`** — Only lane 0 of each group writes back results. Note this uses `inthreads` without `.async` — a sequential thread filter, not a concurrent region.

This pattern — Croqtile handles the parallelism, tiling, and memory orchestration; `__device__` functions handle the per-group algorithm — is common in production kernels like MoE (mixture-of-experts) top-K, custom reductions, and specialized math.

## `__cpp__`: verbatim C++ injection

`__cpp__` takes a string literal and pastes it, character for character, into the generated source file. Whatever you place there must be valid at the splice point. The Croqtile compiler does not parse or rewrite the contents.

**Two forms:**

- **`__cpp__("...")`** — Ordinary string; best for short one-liners.
- **`__cpp__(R"(...)")`** — Raw string literal; use for `asm volatile` where escaping every `"` would be painful.

## The preprocessor

Croqtile's preprocessor runs before the main compiler pass. Macros expand in both `__co__` regions and host C++ in the same `.co` file, so one definition keeps tile geometry and host-side checks aligned.

| Directive                            | Role                                        |
|--------------------------------------|---------------------------------------------|
| `#define NAME value`                 | Object-like macro: textual replacement      |
| `#if` / `#elif` / `#else` / `#endif` | Conditional inclusion                       |
| `#ifdef` / `#ifndef`                 | Shorthand for whether a macro is defined    |
| `#error message`                     | Force a compile-time failure with a message |

Function-like macros (`#define MAX(a, b) ...`) are not supported. Use `constexpr` helpers in ordinary C++.

### Tile geometry as macros

Production matmul sources centralize tile dimensions at the top:

```choreo
#define MATMUL_WARP_M 64
#define MATMUL_WARP_N 128
#define MATMUL_TILE_K 64
#define MATMUL_WARP_K 16
#define MATMUL_STAGES 4
```

The same names appear in `parallel`, shared-storage declarations, `foreach` bounds, and host-side verification.

### Compile-time assertions with `#error`

```choreo
#if MATMUL_SWIZ != (2 * MATMUL_TILE_K)
#error "MATMUL_SWIZ must equal 2 * MATMUL_TILE_K for f16 kernel"
#endif

#if MATMUL_WARP_M != 64
#error "MATMUL_WARP_M must be 64 for the f16 operand geometry"
#endif
```

Treat these guards as documentation of the tile geometry the kernel was written for.

### Conditional code paths

```choreo
#define PATH0

__co__ auto foo() {
  #ifdef PATH0
    // path 0 code
  #else
    // path 1 code
  #endif
}
```

The preprocessor keeps one branch and discards the other before Croqtile parses the `__co__` body. Command-line defines: `croq kernel.co -DMATMUL_TILE_K=128`.

## Chapter summary

| Syntax                      | What it does                                                                | Runs on      |
|-----------------------------|-----------------------------------------------------------------------------|--------------|
| `__device__ fn()`           | Device function; passed through unchanged to the generated source           | Device       |
| `call fn(args)`             | Invoke a `__device__` function from a `__co__` body                         | Device       |
| `__cpp__("...")`            | Inject verbatim C++ at the splice point (ordinary string)                   | Target       |
| `__cpp__(R"(...)")`         | Inject verbatim C++ (raw string; use for `asm volatile`)                    | Target       |
| `#define NAME value`        | Object-like macro; shared across `__co__` and host regions                  | Preprocessor |
| `#if` / `#ifdef` / `#error` | Conditional compilation and compile-time assertions                         | Preprocessor |
| `-DNAME=value`              | Define macros from the command line for tile sweeps                         | Preprocessor |

The [next chapter](ch09-debug-verbose.md) turns to the other side of the workflow: what to do when the kernel compiles but the output is wrong — debugging, verbose modes, and systematic narrowing.
