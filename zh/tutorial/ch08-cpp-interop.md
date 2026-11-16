# C++ 互操作：`__device__` 函数、内联代码与预处理器

每一种高级语言都需要一条通往更底层机制的逃生通道。Python 有 `ctypes` 与 C 扩展，Rust 有 `unsafe` 块与 `extern "C"`，Java 有 JNI。原因总是一样的：无论抽象多么富有表现力，总有某些硬件特性、某些遗留库、某些性能攸关的内建能力处在该语言域之外。

鳄霸（Croqtile）的互操作叙事包含三个部分：

1. **`__device__` 函数** — 设备函数，与 `__co__` 函数共存于同一 `.co` 文件中。编译器对它们原样透传。
2. **`__cpp__`** — 将逐字 C++ 或 PTX 注入到生成代码中，用于 DSL 无法发出的硬件相关指令。
3. **预处理器** — `#define`、`#if`、`#ifdef`、`#error`，用于编译期配置，与 C/C++ 代码库中 `#define` 所扮演的角色相同。

前面几章已堆起一整座抽象栈：分块、并行、MMA、流水线、事件。那是你希望长期栖居的 Croqtile 世界。本章讨论的是你需要暂时跨出去的那些时刻。

## `__device__` 与普通 C++ inline 的区别

`__device__` 函数与普通 C++ 函数都被原样透传，但它们落入不同的编译路径——这一区别至关重要。

`__device__` 函数编译为**设备机器码**。它可以使用目标内建（`__shfl_xor_sync`、`__frsqrt_rn`、`atomicAdd`），访问 `__shared__` 存储，在 `__global__` kernel 内部运行。`__co__` 主体通过 `call` 关键字调用它，生成的 `__global__` kernel 在设备端直接调用。

普通 C++ 函数（无 `__device__` 或 `__co__` 注解）编译为 **CPU 代码**。它在 host 上运行——建立缓冲区、启动 kernel、打印结果。你不能在 `__co__` 主体内调用 host 函数；编译器会拒绝，因为生成的 kernel 运行在设备上。

边界是严格的：`__device__` 位于切分的设备侧，host C++ 位于主机侧。`__co__` 函数是桥梁——编译器将其变换为 `__global__` kernel（device 侧）并生成 host 侧的启动封装。

## `__device__` 函数

当算法需要在线程或 lane 粒度上运作的辅助函数——例如自定义归约、排序网络、特殊数学函数——把它写成设备函数往往是自然之选。Croqtile 无需管理这些；它们原样交给目标编译器。

`__co__` 函数可使用 `call` 关键字调用 `__device__` 函数：

```choreo
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

      parallel gid by [N_THREAD / 32] : group {
        parallel tid by 32 : thread {
          score = shared_scores.data.span_as(|shared_scores.data.span|).at(gid # tid);

          local s32 [K] frag_idx{-1};
          local f32 [K] frag_val{-1.0f};
          frag_idx.at(0) = tid;
          frag_val.at(0) = score;

          call warp_topk<8>(frag_val, frag_idx);

          inthreads (tid == 0) {
            foreach k in K {
              topk_ids.at(n#m, gid#k) = frag_idx.at(k);
              topk_scores.at(n#m, gid#k) = frag_val.at(k);
            }
          }
        }
      }
    }
  }
}
```

**`__device__ void warp_topk<K>`** — 使用 shuffle 指令实现组内 top-K 选择的模板设备函数。它操作原始指针（`f32*`、`s32*`），而非 Croqtile spans。

**`__device__ T SHFL_XOR`** — lane shuffle 内建封装。`__forceinline__` 与 `__shfl_xor_sync` 是设备代码——Croqtile 原样透传。

**`call warp_topk<8>(...)`** — `call` 关键字在 `__co__` 主体内调用 `__device__` 函数。参数按指针传递；编译器负责在 Croqtile span 与原始指针之间完成地址转换。

**`inthreads (tid == 0)`** — 每个组中只有 lane 0 写回结果。注意此处使用不带 `.async` 的 `inthreads` —— 为顺序线程过滤，而非并发区域。

这一模式——Croqtile 负责并行、分块与存储编排；`__device__` 函数负责每组内的算法——在 MoE（mixture-of-experts）top-K、自定义归约与专用数学等生产级 kernel 中十分常见。

## `__cpp__`：逐字 C++ 注入

`__cpp__` 接收一个字符串字面量，并逐字符粘贴到生成的源码文件中。置于该处的任何内容在拼接点都必须合法。鳄霸编译器既不解析也不改写其内容。

**两种形式：**

- **`__cpp__("...")`** — 普通字符串；最适合短单行。
- **`__cpp__(R"(...)")`** — 原始字符串字面量；在需要 `asm volatile` 且为每个 `"` 转义会很痛苦时使用。

### 提前返回与守卫

MoE 风格 kernel 常处理可变长度的 expert 段；某些 launch 宽度为零：

```choreo
__cpp__("if (seg_end - seg_start <= 0) return;\n\n");
```

标识符须与周围生成代码所声明的一致。

## 预处理器

鳄霸的预处理器在主编译遍之前运行。宏在同一 `.co` 文件的 `__co__` 区域与 host C++ 中都会展开，因此一份定义即可使 tile 几何与 host 侧检查保持一致。

| 指令 | 作用 |
|-----------|------|
| `#define NAME value` | 类对象宏：文本替换 |
| `#if` / `#elif` / `#else` / `#endif` | 条件包含 |
| `#ifdef` / `#ifndef` | 宏是否已定义的简写 |
| `#error message` | 以消息强制编译期失败 |

不支持函数式宏（`#define MAX(a, b) ...`）。请在普通 C++ 中使用 `constexpr` 辅助函数。

### 以宏表示 tile 几何

生产级 matmul 源码通常在文件顶部集中定义 tile 尺寸：

```choreo
#define MATMUL_WARP_M 64
#define MATMUL_WARP_N 128
#define MATMUL_TILE_K 64
#define MATMUL_WARP_K 16
#define MATMUL_STAGES 4
```

相同名字出现在 `parallel`、共享存储声明、`foreach` 边界以及主机侧校验中。

### 使用 `#error` 的编译期断言

```choreo
#if MATMUL_SWIZ != (2 * MATMUL_TILE_K)
#error "MATMUL_SWIZ must equal 2 * MATMUL_TILE_K for f16 kernel"
#endif

#if MATMUL_WARP_M != 64
#error "MATMUL_WARP_M must be 64 for the f16 operand geometry"
#endif
```

把这些守卫看作对核函数所要求的 tile 几何的文档化说明。

### 条件代码路径

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

预处理器在鳄霸解析 `__co__` 主体之前保留一分支并丢弃另一分支。命令行定义：`croq kernel.co -DMATMUL_TILE_K=128`。

## 本章小结

| 语法 | 功能 | 运行位置 |
|--------|------|---------|
| `__device__ fn()` | 设备函数；逐字透传到生成的源码 | 设备 |
| `call fn(args)` | 从 `__co__` 主体内调用 `__device__` 函数 | 设备 |
| `__cpp__("...")` | 在拼接点注入逐字 C++（普通字符串） | 目标 |
| `__cpp__(R"(...)")` | 注入逐字 C++（原始字符串；用于 `asm volatile`） | 目标 |
| `#define NAME value` | 类对象宏；在 `__co__` 与 host 区域间共享 | 预处理器 |
| `#if` / `#ifdef` / `#error` | 条件编译与编译期断言 | 预处理器 |
| `-DNAME=value` | 命令行定义宏，用于 tile 尺寸扫描 | 预处理器 |

[下一章](ch09-debug-verbose.md)转向工作流的另一面：kernel 已能编译但结果错误时该如何处理 —— 调试、verbose 模式与系统化缩小问题范围。
