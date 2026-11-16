# 张量缩并：`mma` 语法

第 3 章的分块矩阵乘法以内缩并的方式模拟 CPU：通过 `foreach k` 循环，每次迭代将两个标量相乘并累加到某一输出元素。这种做法可行，却也是使用现代加速器最慢的方式，因为它忽略了专为此类任务设计的硬件单元。

该任务即**二维张量收缩**——运算 C += A × B，其中 A、B、C 均为形状固定的小型矩阵分块。它是所有 GEMM、所有以 im2col 矩阵乘表示的卷积、以及每个注意力头中 QK^T 的内层核函数。这类运算足够核心，因此加速器把它作为一条宏指令执行：取一分块 A、一分块 B，相乘，累加到 C。各实现的分块尺寸不同，但数学形态相同。

![2D tensor contraction: A[M,K] times B[K,N] into C[M,N]](../assets/images/ch04/fig1_tensor_contraction_dark.png#only-dark)
![2D tensor contraction: A[M,K] times B[K,N] into C[M,N]](../assets/images/ch04/fig1_tensor_contraction_light.png#only-light)

*同一数学运算——在分块形操作数上执行 C += A × B——映射到不同的实现。*

对程序员而言，困难不在于数学而在于**操作数布局**。分块并非连续存放在某个线程的存储中，而是**碎片化**分布到协作的各个 lane 上，具体散布方式取决于数据类型，以及操作数是 A、B 还是 C。布局一旦弄错——例如把列主序分块当作行主序片段装载——结果会静默出错。

![Operand fragment layout: lanes own scattered fragments of the tile](../assets/images/ch04/fig2_register_loading_dark.png#only-dark)
![Operand fragment layout: lanes own scattered fragments of the tile](../assets/images/ch04/fig2_register_loading_light.png#only-light)

*协作 lanes 如何持有 MMA 分块分散片段的简化示意。具体模式因实现而异且有意不透明。*

Croqtile 的设计完全绕开上述复杂性。它不暴露实现相关的 fragment 类型，而是提供作用于不透明分块的**四种抽象运算**：**fill**、**load**、**multiply**、**store**。无论哪个后端执行，这些运算都描述同一套二维收缩流程。编译器负责片段布局、lane 映射与指令选择——你描述的是*要做什么*收缩，而不是*操作数如何*散布。

![Croqtile's four-step MMA syntax: fill, load, multiply, store](../assets/images/ch04/fig3_mma_syntax_dark.png#only-dark)
![Croqtile's four-step MMA syntax: fill, load, multiply, store](../assets/images/ch04/fig3_mma_syntax_light.png#only-light)

*四步 MMA 语法是抽象接口——并不绑定于任何单一收缩单元。任何支持二维分块收缩的后端都可以映射到这些运算。*

## 四步 MMA 语法

每个张量缩并内核遵循同一节奏：

1. **`mma.fill 0.0`** —— 在寄存器中分配累加器 tile `mc` 并置零。
2. **`mma.load`** —— 将操作数 tile 从共享内存载入不透明的 MMA 寄存器 `ma` 与 `mb`。
3. **`mma.row.row mc, ma, mb`** —— 发出缩并：**C += A × B** 写入 `mc`。
4. **`mma.store mc, dst`** —— 将 `mc` 从寄存器写回共享内存。

对 K 循环执行第 2–3 步（每次迭代载入下一 K 切片，向同一 `mc` 累加），然后执行一次第 4 步以刷出完成的输出 tile。名称 `mc`、`ma`、`mb` 均为不透明寄存器 tile——无需逐 lane 声明布局；编译器依据目标与布局选择（此处为 `row.row`）推导之。

`.row.row` 后缀为**布局契约**——它告知硬件如何解释 `ma` 与 `mb` 中的比特。两操作数均为行主序。若 B 在共享内存中为列主存储，则应写 `mma.row.col mc, ma, mb`。完整布局组合为 `row.row`、`row.col`、`col.row`、`col.col`；实践中 `row.row` 与 `row.col` 覆盖多数内核。选错变体属于正确性错误，而非性能提示——硬件对不同布局解释寄存器比特的方式不同。

## 扩展协作范围

无论多少个实例协作完成单次收缩，四步语法都保持不变。变化的是**协作范围**——多少个 lane 组成一个协作组——本节正是围绕这一点展开。

### 一组一分块：最简单的 MMA 矩阵乘

在某些后端上，收缩的作用域是一个固定的小组。在 Croqtile 中对应 `: group`。以下为一完整 FP16 矩阵乘内核，其中各 `MATMUL_*` 常数均为 16，故一个 block tile 即等于一个 MMA tile：

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, MATMUL_TILE_M), cdiv(N, MATMUL_TILE_N)] : block {
    shared f16 [MATMUL_TILE_M, MATMUL_TILE_N] output_s;

    parallel {warp_m, warp_n} by [cdiv(MATMUL_TILE_M, MATMUL_MMA_M), cdiv(MATMUL_TILE_N, MATMUL_MMA_N)] : group {
      mc = mma.fill 0.0;

      foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
        lhs_load_s = dma.copy lhs.subspan(MATMUL_TILE_M, MATMUL_TILE_K).at(block_m, iv_k) => shared;
        rhs_load_s = dma.copy rhs.subspan(MATMUL_TILE_N, MATMUL_TILE_K).at(block_n, iv_k) => shared;

        foreach iv_warp_k in [cdiv(MATMUL_TILE_K, MATMUL_MMA_K)] {
          ma = mma.load lhs_load_s.chunkat(warp_m, iv_warp_k);
          mb = mma.load rhs_load_s.chunkat(warp_n, iv_warp_k);
          mma.row.row mc, ma, mb;
        }
      }

      mma.store mc, output_s.subspan(MATMUL_MMA_M, MATMUL_MMA_N).at(warp_m, warp_n);
    }

    dma.copy output_s => output.subspan(MATMUL_TILE_M, MATMUL_TILE_N).at(block_m, block_n);
  }
}
```

**`__co__ void` 与原地输出。** 内核无返回值；结果经 `output` 写出，与通过指针原地写入的常见做法一致。

**Block 网格。** `cdiv(M, MATMUL_TILE_M)` 为向上取整除法——沿 M 方向含不完整 tile 在内的 tile 数目。`block_m` 与 `block_n` 选定本 block 负责的输出 tile。

**组网格与 `mma.fill`。** `parallel {warp_m, warp_n} ... : group` 把 MMA 分块映射到协作组。各维都为 16 时，extent 为 1×1——一个组覆盖整个块分块。更宽的块分块会增加组，每组有自己的 `mc`。

**K 循环与 DMA。** 每个 `iv_k` 阶段通过 `dma.copy` 与 `subspan(...).at(...)` 将 A、B 条带拉入共享内存。第 7 章进一步讨论 `subspan` 与 `chunkat` 的对比。

**操作数装载。** `mma.load` 把该组的分块从共享存储移入 `ma` / `mb`。`chunkat(warp_m, iv_warp_k)` 为本组和当前内层 K 步选出 M×K 切片。

**写回与收尾。** K 走完后，`mma.store` 把 `mc` 写入 `output_s` 中属于该组的子矩形，再由 `dma.copy` 把整个块分块送往全局存储。

这个核函数简单，是因为协作范围窄：一个组，一次一个 MMA 分块。四步语法读起来是线性的，分块几何也一目了然。更宽的协作范围改变的是组的数量，而不是这些操作。

## 超越稠密 FP16：四步还能表达什么

上文示例在稠密 FP16 tile 上使用 `mma.row.row`。同一四步模式可延伸至基本形式无法覆盖的工作负载。

**结构化稀疏。** 当 A 中一半元素遵循固定的稀疏零模式时，后端可以跳过零乘积——但它需要一个**元数据操作数** `me` 来标明哪些元素非零：

```choreo
mma.row.row.sp mc, ma, mb, me;
```

`.sp` 后缀增加元数据操作数；其余仍为同一 fill-load-multiply-store 节奏。任意布局组合均可：`mma.row.col.sp` 等。需将 `me` 与 A、B 一并自独立元数据张量载入。

**带每 tile 缩放的量化操作数。** FP8 操作数（`f8_e4m3`、`f8_e5m2`）需要每 tile 反量化以使累加器保持数值准确。鳄霸将缩放融合进缩并：

```choreo
mma.row.row.scale mc, ma, mb, sc_a, sc_b;
```

每个结果元素在缩并后由 `sc_a` 与 `sc_b` 缩放——无需单独的反量化内核。若缩放来源与标准融合路径不同，缩放亦可为**独立语句**：

```choreo
mma.row.row mc, ma, mb;
mma.scale mc, sc_a, sc_b;
```

独立 `mma.scale` 见于部分 MoE 与混精度内核。

**打乱载入与转置存储。** 当共享内存采用 swizzle 模式以避免 bank 冲突（[第 7 章](ch07-advanced-movement.md)）时，MMA 载入须使用匹配的 swizzle 模式：`mma.load.swiz<N>`。`<N>` 须在 `dma.copy.swiz<N>` 与 `mma.load.swiz<N>` 之间一致——不一致则读出垃圾数据。对输出，`mma.store.transp mc, dst` 以行列互换方式写入累加器，适用于下一阶段期望列主数据的情形。

**流水线同步。** 在生产者和消费者区域重叠的流水线核函数中（[第 6 章](ch06-synchronization.md)），`mma.commit` 标记「本 K 切片的操作数已读完」与「可以安全复用暂存缓冲区」之间的边界。它是事件驱动流水线中不可省略的粘合。

上述扩展均遵循同一设计原则：四步骨架固定，变体后缀向硬件传达特定契约。下表汇总各变体以备查阅。

## 新语法

| 语法 | 含义 |
|--------|---------|
| `mc = mma.fill 0.0` | 将累加器 tile 初始化为零 |
| `mma.fill.f16 0.0f` / `mma.fill.f32 0.0f` | 显式精度的累加器 |
| `ma = mma.load src.chunkat(...)` | 从共享内存将操作数 tile 载入 MMA 寄存器 |
| `mma.load.swiz<N> src` | 按 swizzle 模式载入（见[第 7 章](ch07-advanced-movement.md)） |
| `mma.row.row mc, ma, mb` | C += A × B（二者均为行主序） |
| `mma.row.col mc, ma, mb` | C += A × B（A 行主序，B 列主序） |
| `mma.row.row.sp mc, ma, mb, me` | 带元数据操作数的稀疏 MMA |
| `mma.row.row.scale mc, ma, mb, sc_a, sc_b` | 融合 MMA 与每 tile 反量化 |
| `mma.scale mc, sc_a, sc_b` | 独立的 MMA 后缩放 |
| `mma.store mc, dst` | 将累加器写入共享内存 |
| `mma.store.transp mc, dst` | 转置写入累加器 |
| `mma.commit` | 流水线阶段栅栏（见[第 6 章](ch06-synchronization.md)） |
| `cdiv(a, b)` | 向上取整除法 |
| `__co__ void fn(...)` | 原地写入结果的内核 |

## 本章小结

| 主题 | 要点 |
|-------|----------|
| 二维张量缩并 | 在 tile 形操作数上 C += A × B——通用的内层内核 |
| 后端多样性 | 不同后端以不同方式实现该收缩；tile 尺寸与操作数布局各异 |
| 四步抽象 | **fill → load → multiply → store**；编译器为各目标处理 fragment 布局 |
| 协作规模扩展 | 每个分块使用更多协作组——四个运算保持不变 |
| 布局契约 | `mma.row.row`、`mma.row.col` 等——须与共享内存中的数据一致 |
| 稀疏与量化 | `.sp` 增加元数据操作数；`.scale` 融合每 tile 反量化 |
| Swizzle 与流水线 | `mma.load.swiz<N>` 与打乱后的共享布局匹配；`mma.commit` 划分流水线阶段 |

缩并本身很快，但载入与计算仍须轮流执行——张量核心做乘法时，内存子系统往往空闲。[下一章](ch05-branch-control.md)引入**角色特化**，使不同线程组承担不同职责：一组载入数据的同时另一组计算，从而在时间上重叠内存访问与算术。
