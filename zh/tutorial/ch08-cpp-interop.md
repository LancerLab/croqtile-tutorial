# C++ 互操作：内联代码与预处理器

每种高级语言都需要一个通向底层的逃生通道。Python 有 `ctypes` 和 C 扩展。Rust 有 `unsafe` 块和 `extern "C"`。Java 有 JNI。原因始终相同：无论抽象层多富有表达力，总有某些硬件特性、某个遗留库、某条性能关键的内建指令超出该语言的领域。逃生通道让你在不放弃系统其余部分的前提下触及它们。

鳄霸（Croktile）的逃生通道是 **`__cpp__`**：向生成的代码中逐字注入 C++ 或 PTX。预处理器——`#define`、`#if`、`#ifdef`、`#error`——则承担**编译期配置**的角色，功能与 C/C++ 代码库中的 `#define` 一致：在 DSL 函数体与宿主代码间共享常量、保护平台相关路径、在配置错误变为运行时 bug 之前捕获它们。

至此，本教程已搭建起完整的抽象栈：tile、并行、MMA、流水线、事件。这就是你希望长期使用的鳄霸。本章讲述需要走出这一层的时刻——编译器不会发出的某条 PTX 指令、使 warp 特化角色干净共存的寄存器预算提示，或在 `__co__` 函数体与宿主 C++ 两侧都必须一致的编译期常量。

![鳄霸内核体中嵌入用于逐字 PTX/C++ 的 __cpp__ 孤岛](../assets/images/ch08/fig1_escape_hatch_dark.png#only-dark)
![鳄霸内核体中嵌入用于逐字 PTX/C++ 的 __cpp__ 孤岛](../assets/images/ch08/fig1_escape_hatch_light.png#only-light)

## `__cpp__`：逐字 C++ 注入

**作用。** `__cpp__` 接收一个字符串字面量，并将其**逐字符**粘贴到生成的 CUDA 或 C++ 文件中。置于该处的内容必须在拼接点合法：花括号、分号、类型与作用域须与周围代码生成一致。鳄霸编译器不会解析或改写其内容，鳄霸符号也不会在字符串内自动可见——只有实际出现在生成输出中的名称才有效。

**两种形式。**

- **`__cpp__("...")`** ——普通字符串；适合简短单行与简单守卫。
- **`__cpp__(R"(...)")`** ——原始字符串字面量；用于 `asm volatile("...")` 等若对每个 `"` 转义会很繁琐的片段。

### 寄存器提示：`setmaxnreg`

典型的 `__cpp__` 用例是在 warp 特化流水线（[第 5 章](ch05-branch-control.md)）中进行**寄存器重分配**。生产者 warpgroup 寄存器占用较轻——主要发出 TMA 加载——而消费者较重——持有 MMA 累加器。NVIDIA PTX 的 `setmaxnreg` 指令在这些角色之间移动寄存器预算：

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

**`setmaxnreg.dec` / `setmaxnreg.inc`** ——具体数值（此处为 40 与 216）按内核调优。`.dec` 与 `.inc` 形式与硬件分配器协同，使两种角色能共存而不必无谓地 spill。

**放置位置** ——将提示放在每个 `inthreads.async` 分支顶部、重循环之前，以便分配器在主体执行前看到预算。

### 提前返回与守卫

MoE 风格 GEMM 内核常处理可变大小的专家片段；某些启动对给定专家宽度为零。一行 `__cpp__` 注入即可在其余内核逻辑运行前退出：

```choreo
__cpp__("if (seg_end - seg_start <= 0) return;\n\n");
```

**命名纪律** ——标识符（`seg_end`、`seg_start`）须与周围生成代码中的声明一致。若重命名鳄霸参数导致代码生成改名，过时的 `__cpp__` 字符串会在编译时报错——这优于悄无声息的错误结果。

### `__cpp__` 字符串内的宏

常见误区是以为预处理器会在注入字符串内起作用：

```choreo
#define PRODUCER_MAXNREG 40

inthreads.async (p1 == 0) {
  __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 PRODUCER_MAXNREG;");)");
}
```

**失败原因** ——预处理器不会在字符串字面量内展开宏。生成的汇编仍会包含标识符 `PRODUCER_MAXNREG` 而非 `40`，PTX 会拒绝。

**团队常见做法** ——在 `__cpp__` 字符串内直接写数值字面量，并在字符串**之外**用 `#if` / `#error` 与 tile 配置强制一致：

```choreo
#define PRODUCER_MAXNREG 40
#if PRODUCER_MAXNREG > 50
#error "Producer maxnreg too high for this tile config"
#endif

inthreads.async (p1 == 0) {
  __cpp__(R"(asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");)");
}
```

## 预处理器

鳄霸预处理器在主编译阶段之前运行。**`#define`**、**`#if` / `#elif` / `#else` / `#endif`**、**`#ifdef` / `#ifndef`** 与 **`#error`** 的行为与其 C 语言对应物类似。宏在同一 `.co` 文件的 `__co__` 区域与宿主 C++ 中均会展开，因而一份定义可同时约束 tile 几何与宿主侧检查。

**指令参考。**

| 指令 | 作用 |
|-----------|------|
| `#define NAME value` | 对象式宏：文本替换 |
| `#if` / `#elif` / `#else` / `#endif` | 条件包含 |
| `#ifdef` / `#ifndef` | 宏是否已定义的简写 |
| `#error message` | 以消息强制编译期失败 |

**限制** ——不支持函数式宏（`#define MAX(a, b) ...`）。需要参数化表达式时，请在普通 C++ 中使用 `constexpr` 辅助函数。

### 以宏表示 tile 几何

生产级矩阵乘法源码通常在文件顶部集中定义所有 tile 维度：

```choreo
#define MATMUL_WARP_M 64
#define MATMUL_WARP_N 128
#define MATMUL_TILE_K 64
#define MATMUL_WARP_K 16
#define MATMUL_STAGES 4
```

**共享契约** ——同一组名称出现在 `parallel ... by [cdiv(M, MATMUL_WARP_M), ...]`、共享内存声明、`foreach` 边界与宿主侧校验中。修改一处 `#define`，所有使用点一并更新。

### 以 `#error` 做编译期断言

库代码常将 `#if` 与 `#error` 配对，使非法组合在编译期以明确消息失败：

```choreo
#if MATMUL_SWIZ != (2 * MATMUL_TILE_K)
#error "MATMUL_SWIZ must equal 2 * MATMUL_TILE_K for f16 kernel"
#endif

#if MATMUL_WARP_M != 64
#error "MATMUL_WARP_M must be 64 for f16 WGMMA constraints"
#endif
```

**活文档** ——当有人不兼容地修改 swizzle 宽度或 warp tile 时，构建会立即中止，而非生成非法内核。应将这些守卫视为硬件约束的文档化，而非一次性检查。

### 条件代码路径

可根据宏选择整个代码区域：

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

**编译期消除** ——预处理器在鳄霸解析 `__co__` 函数体之前保留一分支并丢弃另一分支。这不是运行时的 `if`；被丢弃分支不会出现在生成程序中。

**命令行定义** ——`croktile kernel.co -DMATMUL_TILE_K=128` 可在不编辑源码的情况下定义或覆盖宏——便于在基准测试中扫描 tile 大小而无需复制文件。

## 阅读生产级 `.co` 文件

打开诸如 `blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c.co` 的基准内核时，请**自上而下**阅读：

1. **宏与 `#error` 守卫** ——契约：允许的 tile 大小、swizzle 规则、架构标志。
2. **宿主设置** ——缓冲区、启动配置、计时；普通 C++。
3. **`__co__` 函数** ——编排：`parallel`、`foreach`、TMA/MMA、`inthreads.async`、事件。将每个区域对应回前面章节。
4. **`__cpp__` 孤岛** ——通常仅数行。在每处停顿并追问：硬件收到了 DSL 未写明的什么。

按此顺序可避免在尚未弄清允许修改哪些常量之前就扎进 warp 特化循环。

## 本章小结

| 主题 | 要点 |
|-------|----------|
| 何时下沉 | PTX 指令、寄存器提示、宿主/设备常量、守卫——DSL 能覆盖处尽量用 DSL；下沉须克制。 |
| `__cpp__` | 逐字粘贴进生成 CUDA；`asm` 用原始字符串；名称须与生成 C++ 一致。 |
| `setmaxnreg` | 典型示例：生产者 dec、消费者 inc；按内核调优；置于重循环之前。 |
| 守卫 / 字符串内宏 | 经 `__cpp__` 提前 `return`；宏**不会**在字符串字面量内展开——用字面量并在字符串外用 `#error`。 |
| 预处理器 | `#define` 表示 tile 几何；`#if` / `#error` 表示约束；`#ifdef` 表示变体；`-D` 用于扫描。 |
| 阅读 `.co` 文件 | 先宏，再宿主，再 `__co__`，最后 `__cpp__` 孤岛。 |

**新语法**

| 语法 | 含义 |
|--------|---------|
| `__cpp__("...")` | 注入逐字 C++（普通字符串） |
| `__cpp__(R"(...)")` | 注入逐字 C++（原始字符串字面量） |
| `#define NAME value` | 对象式宏 |
| `#if expr` / `#elif` / `#else` / `#endif` | 条件编译 |
| `#ifdef NAME` / `#ifndef NAME` | 测试宏是否已定义 |
| `#error "message"` | 编译期断言失败 |

逃生通道形成闭环：鳄霸使日常内核保持可读与结构化，而 `__cpp__` 与预处理器处理位于抽象之下的硬件相关细节。宁可一条 PTX 提示、一条守卫或一条 pragma，也不要在程序中到处散落原始孤岛——让 `__co__` 函数承载主线叙事。

[下一章](ch09-debug-verbose.md)转向工作流的另一面：内核能编译但结果错误时该怎么办——调试、详细模式与系统化收窄。
