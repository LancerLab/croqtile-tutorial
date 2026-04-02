# 调试与详细输出：打印、RTTI 与 GDB

GPU 内核是**不透明系统**。成千上万的线程并发执行，共享内存从宿主端不可见，结果错误时也没有堆栈跟踪指向出错的行。这种不透明性并非 GPU 独有——任何程序员无法直接观察中间状态的系统都面临同样的挑战：分布式系统（丢失的消息在哪里？）、嵌入式固件（哪个中断处理程序破坏了寄存器？）、优化编译器（哪个 pass 破坏了语义？）。

通用的调试准则在各处相同：**从廉价检查到昂贵检查，系统地缩小搜索空间**。先用静态分析与编译期断言（零成本，可捕获整类 bug）；再用定向的运行时探针（成本较低，可定位问题）；仅在更便宜的工具已缩小嫌疑范围后才求助于交互式调试器。每一层的成本递增，因此你希望在最便宜的层面尽可能多地捕获 bug。

鳄霸（Croktile）在每个层面都提供工具：**编译期形状打印**（`print!`、`println!`）在不启动内核的前提下验证 tile 维度，**带守卫的设备端 `println`** 用于特定线程的运行时检查，**`choreo_assert`** 用于不变量检查，**供 `cuda-gdb` 使用的调试 RTTI**，以及清晰的**怀疑顺序**指引排查方向。

至此你已走过完整的鳄霸栈——逐元素加法、数据搬运、并行、张量核心、warp 特化、流水线、TMA 与 C++ 逃生通道。每一步都增加了性能与复杂度。本章讲述结果出错时该怎么办。

## `print!` 与 `println!`：编译期检查

带感叹号的形式（`print!`、`println!`）在**编译期**运行，而非运行时。它们打印到编译器输出，适用于在不启动内核的情况下检查形状、范围与类型信息：

```choreo
__co__ void check_shapes(f32 [3, 2] b) {
  print!("shape of b: ");
  println!("b.span = ", b.span);
}
```

编译此文件时，编译器会输出：

```
shape of b: b.span = {3, 2}
```

无需 GPU。字符串字面量会拼接（`"wor" "ld!"` 变为 `"world!"`），便于由片段拼装诊断信息。在投入完整内核启动前，用 `print!` / `println!` 验证 `chunkat` 与 `subspan` 是否产生你预期的 tile 大小。

## `print` 与 `println`：运行时设备输出

**运行时打印。** `print` 将参数写到标准输出；`println` 相同但追加换行。二者均可在 `__co__` 函数内使用，并在生成 CUDA 中发出设备端 `printf` 调用：

```choreo
__co__ void inspect(s32 [4, 8] data) {
  foreach i in [data.span] {
    println("element ", i, " = ", data.at(i));
  }
}
```

每次 `println` 调用接受逗号分隔的字符串字面量与表达式混合。字符串原样打印；表达式求值后打印其运行时值。跨线程的**输出顺序非确定性**——GPU `printf` 缓冲区异步刷新，不同线程的行会不可预测地交错。

**守卫输出。** 针对特定检查——例如 tile 索引 `(3, 5)` 是否算对部分和——用条件守卫打印：

```choreo
parallel {px, py} by [8, 16] : block
  parallel {qx, qy} by [16, 16] : thread {
    // ... compute ...
    if (px == 0 && py == 0 && qx == 3 && qy == 5) {
      println("partial sum = ", accum);
    }
  }
```

若无守卫，每个线程一行——大网格下成千上万行，大多无关。

## 调试 MMA 密集型内核

若错误答案来自**张量核心路径**，应单独归类处理。**先怀疑布局**——行主序与列主序，以及 RHS 在内存中逻辑上是 `[N, K]` 还是 `[K, N]`——**再怀疑索引**（`block_m`、`block_n` 以及用 `.at` / `chunkat` 附着的 K 切片），若引入了异步拷贝或拆分生产者与消费者 warp，则**再怀疑异步次序**。常见错误包括：分阶段数据实为列主序却**误标 `mma.row.row`**，或使用的 **`chunkat` 索引与 MMA tile 几何不对齐**。若使用 swizzle 加载（`tma.copy.swiz` / `mma.load.swiz`），与朴素行主期望不一致会表现为误差中的**规则模式**（例如每第 16 个元素正确，其余偏移）——下文系统化流程会明确点出这一点。

## 调试 RTTI 与 `cuda-gdb`

当 `print` / `println` 不足——需要单步执行、检查寄存器状态或检视复杂数据结构时——鳄霸支持**调试 RTTI**（运行时类型信息），使其类型对 `cuda-gdb` 可见。

**编译选项。** 使用 `-g -O0` 启用调试符号并关闭优化：

```bash
croktile kernel.co -g -O0 -o kernel_debug
```

生成代码包含鳄霸类型的 RTTI 结构：

| 鳄霸类型 | GDB 类型 | 字段 |
|--------------|----------|--------|
| 带形状张量（`s32 [M, N]`） | `choreo::rtti::spanned<int, 2>` | `.span.data[]`（维度）、`.stride.data[]`（步长）、`.data`（指针） |
| 索引元组 | `choreo::rtti::bounded_ituple<N>` | `.data[]`（取值）、`.ub[]`（上界） |
| 整数元组 | `choreo::rtti::ituple<N>` | `.data[]`（取值） |
| 多维 span | `choreo::rtti::mdspan<N>` | `.data[]`（范围） |

**示例会话。** 对调试编译内核的一次 GDB 会话：

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

变量名上的 `__dbg_` 前缀由编译器生成——使鳄霸变量与生成 C++ 中间代码一起在 GDB 中可见。可检查张量维度、步长与数据指针，并在生成代码中单步查看 `foreach` 循环的哪次迭代产生错误值。

## 综合：系统化工作流

以下顺序是有意为之：**先做廉价编译期检查**，再**收窄运行时打印**，接着**流水线语义**、**布局**，最后针对指针级 bug 使用**调试器**。

![调试工作流：形状 → 单 tile → 同步 → 布局 → GDB](../assets/images/ch09/fig1_debug_workflow_dark.png#only-dark)
![调试工作流：形状 → 单 tile → 同步 → 布局 → GDB](../assets/images/ch09/fig1_debug_workflow_light.png#only-light)

**1. 检查形状。** 编译期用 `println!` 验证所有 `chunkat`、`subspan` 与 `span` 表达式是否产生预期 tile 大小。形状 bug 最常见——范围不一致会静默读写错误内存区域。

**2. 检查单个 tile。** 在 K 循环内添加带守卫的 `println`，仅对块 `(0, 0)` 与单个线程触发。每次 K 迭代后打印累加器值。与该输出元素的手算参考值比对。

**3. 检查同步。** 若仅在大 K 时值错误，怀疑事件次序。在生产者与消费者两侧打印 `iv_k` 与 `stage`，验证二者访问同一序列。常见 bug：消费者的 `trigger empty[stage]` 过早触发，使生产者在消费者仍读取时覆盖缓冲区。

**4. 检查布局。** 若结果呈模式性错误（例如每第 16 个元素正确，其余偏移），怀疑 `mma.row.row` 的行/列主序不匹配，或 `tma.copy.swiz<N>` 与 `mma.load.swiz<N>` 的 swizzle 模式不一致。与上文 MMA 小节交叉核对。

**5. 指针 bug 用 GDB。** 若 `println` 显示毫无道理的值（NaN、巨大整数、期望非零却为零），以 `-g -O0` 编译并在 `cuda-gdb` 中单步。检查张量指针（`__dbg_x.data`）是否指向有效内存。

**基准测试前的性能。** `print` / `println` 代价高：每次调用经全局 `printf` 缓冲区串行化并破坏吞吐测量。调试 RTTI 结构增加寄存器压力并抑制优化。基准测试前应移除所有打印并去掉 `-g -O0`。实用折衷是将打印置于 `#ifdef DEBUG_PRINT` 宏之后（第 8 章），以便从命令行开关而无需改源码：

```choreo
#ifdef DEBUG_PRINT
  println("tile_k=", iv_k, " accum=", mc);
#endif
```

使用 `croktile kernel.co -DDEBUG_PRINT` 编译以启用；不带该标志则为生产运行。

## 小结

| 构造 / 工具 | 作用 |
|------------------|------|
| `print!` / `println!` | 编译期形状与类型检查；无需启动内核 |
| `print` / `println` | 设备端 `printf`；用守卫避免日志洪水 |
| MMA 调试 | 布局 → 索引 → 异步次序；对照 `mma.*` 与实际内存序 |
| `-g -O0`、`cuda-gdb`、RTTI | 检查 `__dbg_*` 张量、指针与控制流 |
| `#ifdef DEBUG_PRINT` | 可选 printf，无永久开销 |

你从第 1 章的**逐元素加法**出发，依次学习了**数据搬运**（第 2 章）、**并行**（第 3 章）、**张量核心**（第 4 章）、**warp 特化**（第 5 章）、**流水线**（第 6 章）、**TMA**（第 7 章）、**C++ 互操作**（第 8 章）与**调试**（第 9 章）。下一步可打开 `croktile/benchmark/` 目录中的基准内核，将其各区域映射到本章，修改一个常量，重建并测量。小而刻意的修改优于大重写——在 GPU 上追逐神秘错误答案之前，`#error` 守卫会早早告诉你配置何时不再合法。
