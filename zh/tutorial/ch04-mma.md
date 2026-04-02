# 张量缩并：`mma` 语法

第 3 章的分块矩阵乘法以内缩并的方式模拟 CPU：通过 `foreach k` 循环，每次迭代将两个标量相乘并累加到某一输出元素。这种做法可行，却也是使用现代加速器最慢的方式，因为它忽略了专为此类任务设计的硬件单元。

该任务即**二维张量缩并**——运算 C += A × B，其中 A、B、C 均为形状固定的小型矩阵 tile。它是所有 GEMM、所有以 im2col 矩阵乘表示的卷积、以及每个注意力头中 QK^T 的内核。该运算如此核心，硬件厂商会构建专用单元以单条宏指令执行：NVIDIA 称之为 **tensor core**，Google 称之为 **MXU**，Intel 提供 **AMX tile**，定制 DSA 亦有各自变体。tile 尺寸各异（NVIDIA 上 FP16 为 16×16×16，TPU 上为 128×128 脉动阵列，AMX 上为 16×64），但数学形态各处相同：取 A 的一块、B 的一块，相乘，累加到 C。

![2D tensor contraction: A[M,K] × B[K,N] → C[M,N], with different hardware implementations](../assets/images/ch04/fig1_tensor_contraction_dark.png#only-dark)
![2D tensor contraction: A[M,K] × B[K,N] → C[M,N], with different hardware implementations](../assets/images/ch04/fig1_tensor_contraction_light.png#only-light)

*同一数学运算——在 tile 形操作数上执行 C += A × B——在不同加速器上映射为不同的硬件实现。*

对程序员而言，困难不在于数学而在于**寄存器布局**。在 GPU tensor core 上，tile 并非连续存放在单一线程的寄存器中，而是**碎片化**的：warp 内 32 条线程各自持有 tile 的分散片段，具体散布模式取决于数据类型、架构代际，以及操作数为 A、B 抑或 C。编写原始 CUDA 意味着声明 `wmma::fragment` 对象，调用 `load_matrix_sync` 以正确模式将共享内存中的 tile 分布到各 lane，发出 `mma_sync`，再调用 `store_matrix_sync` 重组输出。布局一旦出错——例如将列主 tile 载入行主 fragment——结果会在无声中错误。

![GPU tensor core register layout: threads own scattered fragments of the tile](../assets/images/ch04/fig2_register_loading_dark.png#only-dark)
![GPU tensor core register layout: threads own scattered fragments of the tile](../assets/images/ch04/fig2_register_loading_light.png#only-light)

*warp 中 32 条线程如何持有 MMA tile 分散寄存器片段的简化示意。具体模式因架构而异且有意不透明。*

鳄霸（Croktile）的设计完全绕开上述复杂性。它不暴露架构相关的 fragment 类型，而是提供作用于不透明寄存器 tile 的**四种抽象运算**：**fill**、**load**、**multiply**、**store**。无论由何种硬件后端执行，这些运算都描述同一套二维缩并工作流。编译器为目标架构处理 fragment 布局、lane 映射与指令选择——你描述的是*做何种*缩并，而非*寄存器如何*散布。

![Croktile's four-step MMA syntax: fill, load, multiply, store](../assets/images/ch04/fig3_mma_syntax_dark.png#only-dark)
![Croktile's four-step MMA syntax: fill, load, multiply, store](../assets/images/ch04/fig3_mma_syntax_light.png#only-light)

*四步 MMA 语法是抽象接口——并非硬绑定于 GPU tensor core。任何支持二维 tile 缩并的 DSA 均可映射到这些运算。*

## 四步 MMA 语法

每个张量缩并内核遵循同一节奏：

1. **`mma.fill 0.0`** —— 在寄存器中分配累加器 tile `mc` 并置零。
2. **`mma.load`** —— 将操作数 tile 从共享内存载入不透明的 MMA 寄存器 `ma` 与 `mb`。
3. **`mma.row.row mc, ma, mb`** —— 发出缩并：**C += A × B** 写入 `mc`。
4. **`mma.store mc, dst`** —— 将 `mc` 从寄存器写回共享内存。

对 K 循环执行第 2–3 步（每次迭代载入下一 K 切片，向同一 `mc` 累加），然后执行一次第 4 步以刷出完成的输出 tile。名称 `mc`、`ma`、`mb` 均为不透明寄存器 tile——无需逐 lane 声明布局；编译器依据目标与布局选择（此处为 `row.row`）推导之。

`.row.row` 后缀为**布局契约**——它告知硬件如何解释 `ma` 与 `mb` 中的比特。两操作数均为行主序。若 B 在共享内存中为列主存储，则应写 `mma.row.col mc, ma, mb`。完整布局组合为 `row.row`、`row.col`、`col.row`、`col.col`；实践中 `row.row` 与 `row.col` 覆盖多数内核。选错变体属于正确性错误，而非性能提示——硬件对不同布局解释寄存器比特的方式不同。

## 扩展协作范围

无论多少线程协作完成单次缩并，四步语法保持不变。变化的是**协作范围**——一个 warp、一个 warpgroup、两个 warpgroup——本节的叙述即围绕这一递进展开。

### 一 warp、一 tile：最简单的 MMA 矩阵乘

在 Ampere（SM86）上，tensor-core MMA 的作用域为**单个 warp**（32 条线程）。在鳄霸中对应 `: group`。以下为一完整 FP16 矩阵乘内核，其中各 `MATMUL_*` 常数均为 16，故一个 block tile 即等于一个 MMA tile：

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

**`__co__ void` 与原地输出。** 内核无返回值；结果经 `output` 写出，与常见 GPU 经全局指针原地写入的模式一致。

**Block 网格。** `cdiv(M, MATMUL_TILE_M)` 为向上取整除法——沿 M 方向含不完整 tile 在内的 tile 数目。`block_m` 与 `block_n` 选定本 block 负责的输出 tile。

**Warp 网格与 `mma.fill`。** `parallel {warp_m, warp_n} ... : group` 将 MMA tile 映射到各 warp。当各维均为 16 时，范围为 1×1——单 warp 覆盖整个 block tile。更宽的 block tile 会增加 warp 数，各自持有独立的 `mc`。

**K 循环与 DMA。** 每个 `iv_k` 阶段通过 `dma.copy` 与 `subspan(...).at(...)` 将 A、B 条带拉入共享内存。第 7 章进一步讨论 `subspan` 与 `chunkat` 的对比。

**操作数载入。** `mma.load` 将该 warp 的 tile 从共享内存移入 `ma` / `mb`。`chunkat(warp_m, iv_warp_k)` 选取本 warp 及内层 K 步对应的 M×K 切片。

**存储与收尾。** K 循环结束后，`mma.store` 将 `mc` 写入 `output_s` 中该 warp 的子矩形，随后 `dma.copy` 将整个 block tile 送至全局内存。

该内核简单，因协作范围狭窄：32 条线程、一个 warp、每次一个 MMA tile。四步语法线性可读，tile 几何亦清晰。若硬件提供更宽的协作窗口，情形又如何？

### 扩展范围：warpgroup 与 WGMMA

Hopper（SM90）引入 **Warpgroup Matrix Multiply-Accumulate（WGMMA）**：仍为同一 C += A × B 缩并，但由**四个 warp**（128 条线程）协同发出。硬件指令更宽，tile 更大，吞吐提升——而四步语法不变。鳄霸中唯一变化是空间说明符：`: group-4` 取代 `: group`。

![Ampere vs Hopper MMA cooperation scope](../assets/images/ch04/fig4_sm86_vs_sm90_dark.png#only-dark)
![Ampere vs Hopper MMA cooperation scope](../assets/images/ch04/fig4_sm86_vs_sm90_light.png#only-light)

*SM86：每个 MMA 一个 warp。SM90：四个 warp（`: group-4`）协作执行 WGMMA。*

```choreo
parallel {block_m, block_n} by [cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)] : block {
  shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
  shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;

  mc = mma.fill.f16 0.0f;

  foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
    dma.copy lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
    dma.copy rhs.chunkat(block_n, iv_k) => rhs_load_s;

    foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
      parallel p by 1 : group-4 {
        ma = mma.load lhs_load_s.chunkat(_, iv_warp);
        mb = mma.load rhs_load_s.chunkat(_, iv_warp);
        mma.row.row mc, ma, mb;
      }
    }
  }

  shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;
  mma.store mc, output_s;
  dma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
}
```

与 SM86 内核对照阅读。fill、load、multiply、store——节奏完全一致。差异在于机械细节：

**`mma.fill.f16 0.0f`。** Hopper 常显式写出累加器精度——`.f16`、`.f32` 等。FP16 操作数配合 FP32 累加是长 K 维情形的常见模式，可避免部分和数值溢出。SM86 常用更短的 `mma.fill 0.0` 并依赖推断。

**`parallel p by 1 : group-4`。** 一个 warpgroup（四个 warp）执行内层载入与 MMA。助记符 `mma.row.row` 与 Ampere 一致，但硬件发射更宽。

**`chunkat(_, iv_warp)`。** `_` 表示“不对该维做分块”——共享内存中已驻留完整 M（或 N）范围；仅按 MMA 切片对 K 细分。

此即抽象之要义：同一四种运算、同一布局契约、同一 `chunkat` / `subspan` 表达式。编译器据目标为 SM86 或 SM90 映射到不同硬件指令。你思考的是*做何种*缩并；协作宽度属于部署细节。

### 进一步分块：两 warpgroup 共享操作数

第 3 章已介绍 `parallel p1 by 2 : group-4`——即一个 block 内两个 warpgroup。配合 MMA 时，两组可共享同一 B tile，同时载入 A 的不同行。由此大块 tile 可拆为多个 MMA tile，而无需在共享内存中重复 B 操作数：

```choreo
parallel {block_m, block_n} by [cdiv(M, MATMUL_TILE_M), cdiv(N, MATMUL_WARP_N)] : block {
  shared f16 [MATMUL_TILE_M, MATMUL_TILE_K] lhs_load_s;
  shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
  shared f16 [MATMUL_TILE_M, MATMUL_WARP_N] output_s;

  mc = mma.fill.f32 0.0f;

  foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
    dma.copy lhs.subspan(MATMUL_TILE_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
    dma.copy rhs.chunkat(block_n, iv_k) => rhs_load_s;

    parallel p1 by 2 : group-4 {
      foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
        ma = mma.load lhs_load_s.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(p1, 0).chunkat(_, iv_warp);
        mb = mma.load rhs_load_s.chunkat(_, iv_warp);
        mma.row.row mc, ma, mb;
      }
    }
  }

  parallel p1 by 2 : group-4 {
    mma.store mc, output_s.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(p1, 0);
  }
  dma.copy output_s => output.subspan(MATMUL_TILE_M, MATMUL_WARP_N).at(block_m, block_n);
}
```

**以 `p1` 划分 M。** 当 `MATMUL_TILE_M = 128` 且 `MATMUL_WARP_M = 64` 时，block 跨 128 行；`p1` 选取上或下 64 行条带。`lhs_load_s.subspan(MATMUL_WARP_M, ...).at(p1, 0)` 为各 warpgroup 提供其 A 行；二者共用同一 `rhs_load_s`。

**镜像式存储。** `mma.store` 目标为 `output_s.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(p1, 0)`，使各 warpgroup 写入其半幅暂存缓冲，随后一次 `dma.copy` 发出合并后的 tile。

该模式可扩展：三个 warpgroup、四个 warpgroup，或任意能整除 block tile 的数目。四步语法保持不变；仅 tile 的并行分解变化。

| 方面 | 单 warp（`: group`） | 单 warpgroup（`: group-4`） | 双 warpgroup |
|--------|---------------------|----------------------------|----------------|
| 线程数 | 32 | 128 | 256 |
| 累加器 | `mma.fill 0.0` | `mma.fill.f16 0.0f` | `mma.fill.f32 0.0f` |
| Tile 划分 | 每 warp 一个 MMA tile | 每 warpgroup 一个 MMA tile | Block tile 跨 warpgroup 划分 |
| 操作数共享 | 不适用 | 不适用 | 共享 B tile，A 行由 `p1` 划分 |

## 超越稠密 FP16：四步还能表达什么

上文示例在稠密 FP16 tile 上使用 `mma.row.row`。同一四步模式可延伸至基本形式无法覆盖的工作负载。

**结构化稀疏性。** 当 A 中半数元素服从 2:4 零模式（Ampere 及以后）时，硬件可跳过零乘积并大致将吞吐翻倍——但需要**元数据操作数** `me` 编码哪些元素非零：

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

**打乱载入与转置存储。** 当共享内存采用 swizzle 模式以避免 bank 冲突（[第 7 章](ch07-advanced-movement.md)）时，MMA 载入须使用匹配的 swizzle 模式：`mma.load.swiz<N>`。`<N>` 须在 `tma.copy.swiz<N>` 与 `mma.load.swiz<N>` 之间一致——不一致则读出垃圾数据。对输出，`mma.store.transp mc, dst` 以行列互换方式写入累加器，适用于下一阶段期望列主数据的情形。

**流水线同步。** 在生产者与消费者 warpgroup 重叠的流水线内核中（[第 6 章](ch06-synchronization.md)），`mma.commit` 标示“已完成读取本 K 条带操作数”与“可安全复用共享内存缓冲”之间的边界。在事件驱动流水线中为必需胶合。

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
| `mma.commit` | WGMMA 的流水线阶段栅栏（见[第 6 章](ch06-synchronization.md)） |
| `cdiv(a, b)` | 向上取整除法 |
| `__co__ void fn(...)` | 原地写入结果的内核 |

## 本章小结

| 主题 | 要点 |
|-------|----------|
| 二维张量缩并 | 在 tile 形操作数上 C += A × B——通用的内层内核 |
| 硬件多样性 | GPU tensor core、TPU MXU、Intel AMX、定制 DSA 均实现该运算；tile 尺寸与寄存器布局各异 |
| 四步抽象 | **fill → load → multiply → store**；编译器为各目标处理 fragment 布局 |
| 协作规模扩展 | `: group`（单 warp）→ `: group-4`（单 warpgroup）→ 多 warpgroup——语法不变 |
| 布局契约 | `mma.row.row`、`mma.row.col` 等——须与共享内存中的数据一致 |
| 稀疏与量化 | `.sp` 增加元数据操作数；`.scale` 融合每 tile 反量化 |
| Swizzle 与流水线 | `mma.load.swiz<N>` 与打乱后的共享布局匹配；`mma.commit` 划分流水线阶段 |

缩并本身很快，但载入与计算仍须轮流执行——张量核心做乘法时，内存子系统往往空闲。[下一章](ch05-branch-control.md)引入**角色特化**，使不同线程组承担不同职责：一组载入数据的同时另一组计算，从而在时间上重叠内存访问与算术。
