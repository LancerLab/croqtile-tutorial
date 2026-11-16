# 并行性：鳄霸如何表达并行工作

前两章刻意保持简单：`parallel {i, j} by [4, 8]` 创建了 32 个实例，每个处理一个元素，我们从未追问这些实例究竟在哪里执行。对于逐元素加法和理解数据搬运来说，这已经足够——编译器会选择合理的默认值。但执行引擎并不是一堆完全相同处理器的平面集合，它把工作组织成彼此协作的层级，而把工作映射到正确层级，正是玩具演示与真正实用内核之间的关键差距。

本章将并行性作为鳄霸（Croqtile）的核心概念正式引入：它的含义是什么，鳄霸如何将*逻辑*结构（哪些任务并发执行）与*物理*映射（由哪些硬件单元来执行）分离开来，以及你如何通过**空间标注符（space specifiers）**来控制这种映射。

## 我们如何思考并行任务

在接触任何鳄霸语法之前，先来谈谈"并行"到底意味着什么。

假设你有八个独立的任务——八个 tile 要做加法，八行数据要处理，随便什么。你可以一次性描述这八个任务，称之为"并行"。但这个声明本身并没有说明有多少个任务*真正同时*在运行。在单核上，它们一个接一个地执行；在四核机器上，也许四个同时运行；若有八个执行槽位，八个都可能真正同时推进。

![虚拟并行性：相同任务，不同硬件调度](../assets/images/ch03/fig1_virtual_parallelism_dark.png#only-dark)
![虚拟并行性：相同任务，不同硬件调度](../assets/images/ch03/fig1_virtual_parallelism_light.png#only-light)

*同样八个任务分别按串行、4 路与 8 路调度。任务完全相同——只有执行槽位数量不同。*

关键在于：并行性是一个**虚拟**概念。当你写下"这八个任务是独立的"时，你是在对程序的*逻辑*做声明，而非对硬件做声明。硬件根据自身拥有的执行单元数量和调度器的分配策略，来决定有多少个任务真正同时运行。

这个区别很重要，因为很多并行语言把两者混为一谈。在启动语法同时充当工作描述的语言里，一条语句同时声明了逻辑结构（多少个实例）和物理映射（多少个块、每块多少线程）。想改变分块方式，就不得不修改启动配置。逻辑与物理纠缠在一起。

鳄霸把它们解开了。

## 鳄霸的双层并行模型

鳄霸提供两个独立的控制维度：

1. **`parallel`** — 声明迭代是独立的，*可以*并发执行。这是一个逻辑层面的声明。
2. **空间标注符**（`: block`、`: thread`、`: group`）— 告诉编译器每个并行轴落在哪个层级。这是一个物理层面的声明。

还有 `foreach`，它是 `parallel` 的对立面：迭代按顺序依次执行。当后续迭代依赖先前迭代的结果时使用 `foreach`——例如累加循环、K 维归约、先加载 tile *k* 再用它计算的流水线。

```choreo
parallel {px, py} by [8, 16] : block    // 并发——映射到线程块
  foreach {tile_k} in [16]               // 顺序——K 维循环
    parallel {qx, qy} by [16, 16] : thread  // 并发——映射到线程
      foreach k in [16]                      // 顺序——内层累加
        output += lhs * rhs;
```

`parallel` 行声明"这些迭代是独立的"。`: block` 和 `: thread` 标注声明"将外层放在不同的线程块上，将内层放在同一线程块内的不同线程上"。`foreach` 行声明"按顺序执行"。没有歧义，没有混淆。

![鳄霸中的逻辑并行与物理并行](../assets/images/ch03/fig2_logical_vs_physical_dark.png#only-dark)
![鳄霸中的逻辑并行与物理并行](../assets/images/ch03/fig2_logical_vs_physical_light.png#only-light)

*左：程序员编写的逻辑嵌套（parallel = 并发，foreach = 顺序）。右：空间标注符所选择的层级结构。*

如果你省略空间标注符——只写 `parallel p by 8`——编译器会选择默认映射。第 1 章和第 2 章就是这样做的，对简单情况完全够用。显式标注是你在性能关键场景中掌控映射的方式。

`parallel` 和 `foreach` 均支持多维索引（`{a, b}` 语法）、组合运算符（`#`）和范围运算符（`#p`）。第 1 章和第 2 章的所有内容仍然适用。

### `: block` — 块

```choreo
parallel {px, py} by [8, 16] : block
```

创建 8 × 16 = 128 个线程块网格。每个块运行在一个流式多处理器（SM）上，拥有独立的共享内存分配。块之间在执行期间无法通信——它们是真正独立的。

将 `: block` 用于输出的最外层 tile 划分。如果输出为 `[128, 256]`，每个块处理 `[16, 16]` 的 tile，则需要 `128/16 × 256/16 = 8 × 16 = 128` 个块。

### `: thread` — 块内线程

```choreo
parallel {qx, qy} by [16, 16] : thread
```

在所属块内创建 16 × 16 = 256 个线程。每个线程拥有独立的寄存器，但与同一块内的所有其他线程共享片上存储。

将 `: thread` 用于最细粒度的并行——tile 内的逐元素工作。

### 嵌套标注符

实际的内核会嵌套使用这些层级。常见模式：

```choreo
parallel {px, py} by [8, 16] : block {
  parallel {qx, qy} by [16, 16] : thread {
    // 128 个块 × 256 个线程 = 共 32,768 个线程
  }
}
```

外层 `parallel` 创建块网格，内层 `parallel` 在每个块内创建线程。花括号 `{px, py}` 引入多维索引——是两个嵌套 `parallel` 行的简写。

## 类型系统旁注：`mdspan` 和 `ituple`

在构建矩阵乘法之前，鳄霸类型系统的两个特性值得一提。你已经隐式地使用过它们了；现在正好给它们命名。

**`mdspan`** — 鳄霸中的每个张量都将形状作为类型的一部分。当你写 `s32 [128, 256] lhs` 时，形状 `[128, 256]` 不是运行时值——它是编译期属性。编译器利用它来验证每一个 `.at()`、`chunkat` 和 `dma.copy` 在维度上的一致性。如果你对一个 2D 张量使用 `lhs.at(i, j, k)`，编译器会直接拒绝。表达式 `output.at(px#qx, py#qy)` 会被验证生成的索引在输出的边界范围内。

`span(i)` 语法提取某一个维度：`lhs.span(0)` 是 128，`rhs.span(1)` 是 256。简写 `lhs.span` 复制整个形状。

**`ituple`** — `parallel {px, py} by [8, 16]` 中的 `{px, py}` 语法引入了一个 `ituple`（编译期整数元组）。组合运算符 `px#qx` 将两个 ituple 元素组合为全局索引。范围运算符 `#p` 提取并行变量的范围。这些都在编译期解析——没有运行时开销。

完整细节请参阅 [mdspan 参考](../documentation/shape-in-choreo.md) 和 [ituple 参考](../documentation/integer-ituple.md)。目前的关键要点是：鳄霸在编译期捕获形状不匹配，而非在运行时。

## 矩阵乘法：综合运用

有了 `parallel`、`foreach`、`dma.copy`、`chunkat`、`#` 和空间标注符，分块矩阵乘法就在眼前了。这是教程中第一个把此前所有构造组合起来的程序。

计划：将 `[128, 256]` 矩阵乘以 `[256, 256]` 矩阵，得到 `[128, 256]` 的结果。

### 标量矩阵乘法（无 DMA）

先只用 `parallel` 和 `.at()`——全局内存，无显式数据搬运：

```choreo
__co__ s32 [128, 256] matmul(s32 [128, 256] lhs, s32 [256, 256] rhs) {
  s32 [lhs.span(0), rhs.span(1)] output;

  parallel p by 16, q by 64 {
    foreach index = {m, n, k} in [128 / #p, 256 / #q, 256]
      output.at(p#m, q#n) += lhs.at(p#m, k) * rhs.at(k, q#n);
  }

  return output;
}
```

![标量矩阵乘法输出分块网格](../assets/images/ch03/fig5_scalar_matmul_dark.png#only-dark)
![标量矩阵乘法输出分块网格](../assets/images/ch03/fig5_scalar_matmul_light.png#only-light)

*[128, 256] 输出被划分为 16 × 64 的 tile 网格。每个 (p, q) 对负责一个 tile。*

这段代码很紧凑——逐一解释各个部分。

**从操作数维度推导输出形状。** `s32 [lhs.span(0), rhs.span(1)] output` 从输入构建输出形状：`lhs.span(0)` 是 128（左矩阵的行数），`rhs.span(1)` 是 256（右矩阵的列数）。

**多轴并行。** `parallel p by 16, q by 64` 用逗号声明两个并行索引。创建 16 × 64 = 1024 路并行网格。每个 `(p, q)` 对拥有输出矩阵的一个 tile。

**命名元组解构。** `foreach index = {m, n, k} in [128 / #p, 256 / #q, 256]` 引入三个嵌套循环索引——`m`、`n`、`k`——绑定到名为 `index` 的元组。迭代次数分别为：

- `128 / #p = 128 / 16 = 8` 行/tile
- `256 / #q = 256 / 64 = 4` 列/tile
- `256` 为完整的缩减维度

**组合全局索引。** `p#m` 将 tile 索引与 tile 内偏移组合。`p` 选择 16 个行 tile 中的哪一个，`m` 在该 tile 内从 0 到 7，因此 `p#m` 跨所有 tile 覆盖 0..127。

**算术运算。** `output.at(p#m, q#n) += lhs.at(p#m, k) * rhs.at(k, q#n)` 是教科书式的点积：对每个输出元素，沿 K 维求积之和。这里每个 `.at()` 都从全局内存读取。

### DMA 矩阵乘法：共享存储中的 Tile

标量版本能工作，但每次乘法都从全局内存读取。将 K-tile 加载到共享内存，让块内所有线程复用它们：

```choreo
__co__ s32 [128, 256] matmul(s32 [128, 256] lhs, s32 [256, 256] rhs) {
  s32 [lhs.span(0), rhs.span(1)] output;

  parallel {px, py} by [8, 16] : block {
    foreach {tile_k} in [16] {
      lhs_load = dma.copy lhs.chunkat(px, tile_k) => shared;
      rhs_load = dma.copy rhs.chunkat(tile_k, py) => shared;

      parallel {qx, qy} by [16, 16] : thread {
        foreach k in [256 / #tile_k]
          output.at(px#qx, py#qy) += lhs_load.data.at(qx, k) * rhs_load.data.at(k, qy);
      }
    }
  }

  return output;
}
```

![DMA 矩阵乘法：线程块网格与共享内存及 K 循环](../assets/images/ch03/fig6_dma_matmul_dark.png#only-dark)
![DMA 矩阵乘法：线程块网格与共享内存及 K 循环](../assets/images/ch03/fig6_dma_matmul_light.png#only-light)

*一个块的详细视图：K 循环通过 `dma.copy` 将 lhs 和 rhs tile 加载到共享内存，然后 256 个线程从共享副本计算部分积。*

相比标量版本的变化：

1. 外层 `parallel` 现在使用 `: block` 和花括号索引 `{px, py}`。8 × 16 的块网格覆盖输出。

2. `foreach` 遍历 `tile_k`，沿 K 维分 16 步。每步通过 `dma.copy ... => shared` 将 `lhs` 和 `rhs` 的一个条带复制到 `shared` 内存中。

3. 内层 `parallel {qx, qy} by [16, 16] : thread` 在每个块内创建 256 个线程。每个线程负责块 tile 内的一个输出元素。

4. 算术运算读取 `lhs_load.data` 和 `rhs_load.data`——即共享内存中的副本——而非全局的 `lhs` 和 `rhs`。

组合索引在两层上工作：`px#qx` 将块索引 `px` 与线程索引 `qx` 组合为全局行号；`py#qy` 对列做同样的操作。

**维度计算。** `[8, 16]` 个块意味着每个块拥有 `128/8 = 16` 行和 `256/16 = 16` 列。内层 `[16, 16]` 个线程将其细分为每线程一个元素。沿 K 维，16 个 tile 各含 `256/16 = 16` 个元素，覆盖整个缩减维度。

### 宿主代码

宿主端使用与前面章节相同的 `make_spandata` / `.view()` 模式：

```choreo
int main() {
  auto lhs = choreo::make_spandata<choreo::s32>(128, 256);
  auto rhs = choreo::make_spandata<choreo::s32>(256, 256);
  lhs.fill_random(-10, 10);
  rhs.fill_random(-10, 10);

  auto res = matmul(lhs.view(), rhs.view());

  for (size_t i = 0; i < res.shape()[0]; ++i)
    for (size_t j = 0; j < res.shape()[1]; ++j) {
      int ref = 0;
      for (size_t k = 0; k < lhs.shape()[1]; ++k)
        ref += lhs[i][k] * rhs[k][j];
      choreo::choreo_assert(ref == res[i][j], "values are not equal.");
    }

  std::cout << "Test Passed\n" << std::endl;
}
```

这里没有新内容。宿主程序保持无聊，好让鳄霸函数成为主角。

## 追踪一个输出元素

选取输出中的全局位置 `(row=37, col=50)`。它属于哪个块和哪个线程？

块将 128 行分为 8 个 tile，每个 16 行：`px = 37 / 16 = 2`，偏移 `qx = 37 % 16 = 5`。列将 256 分为 16 个 tile，每个 16 列：`py = 50 / 16 = 3`，偏移 `qy = 50 % 16 = 2`。因此位于块 `(2, 3)`，线程 `(5, 2)`。

对于该线程，K 循环执行 16 次迭代。在 `tile_k = 0` 时，`dma.copy` 将 `lhs` 的第 32..47 行、K 列 0..15 加载到共享内存 `lhs_load`，将 `rhs` 的 K 行 0..15、列 48..63 加载到共享内存 `rhs_load`。内层 `foreach k in [16]` 对 k = 0..15 累加 `lhs_load.data.at(5, k) * rhs_load.data.at(k, 2)` 到 `output.at(37, 50)`。然后 `tile_k = 1` 加载下一个 K 条带，以此类推。全部 16 次迭代完成后，`output.at(37, 50)` 持有完整的点积——与标量参考值完全一致。

## 新语法总结

| 语法 | 含义 |
|------|------|
| `parallel p by N` | N 路并行，索引 `p` 从 0 到 N-1 并发执行 |
| `parallel {px, py} by [M, N]` | 多维并行（笛卡尔积） |
| `parallel p by N : block` | 映射到块 |
| `parallel q by N : thread` | 映射到块内线程 |
| `parallel w by N : group` | 映射到组 |
| `=> shared` | DMA 目标：块级共享存储 |
| `foreach index = {m, n, k} in [a, b, c]` | 在 `foreach` 中使用命名元组解构 |
| `parallel p by A, q by B` | 逗号分隔的并行轴 |
| `lhs.span(0)` | 提取张量形状的某一个维度 |
| `p#m` | 将外层 tile 索引 `p` 与内层偏移 `m` 组合 |

本章构建的分块 DMA 矩阵乘法是每个高性能核函数的结构骨架。下一章用 **`mma`** 操作替换内层循环中的标量 `.at()` 算术——一次处理一个分块，而不是一个在单条指令中处理一个 16×16×16 的 tile。
