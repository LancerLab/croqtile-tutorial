# 高级数据搬运：Swizzle 与不规则访问

第 2 章把 `dma.copy` 介绍为 Croqtile 的通用数据搬运原语——用一条简单的 `src => dst` 箭头在存储层级之间移动矩形分块。本章补齐那一章留下的两件事：收缩消费者需要的**重排布局**，以及常规划分无法表达的**非规则访问**工具。

两者都沿用同一种语句形式。重排是搬运上的修饰符；非规则访问只是换一种方式命名源区域。

## Swizzle 与布局冲突

重排改变暂存分块元素在共享存储中的排列方式。逻辑矩阵形状不变，但读取方必须使用匹配的地址映射。Croqtile 在搬运操作上暴露这个选择：

```choreo
dma.copy.swiz<128> src => dst;
```

搬运按重排模式 `N` 放置分块。MMA 读取路径必须使用相同的 `swiz<N>`，地址才能与暂存布局对上：

```choreo
ma = mma.load.swiz<128> lhs_load_s.chunkat(_, iv_warp);
```

重排是搬运的属性，而不是某个引擎的属性：`dma.copy.swiz<N>` 固定了共享存储中的布局，消费者必须与之一致。

**重排宽度。** 模板参数是以字节计的重排宽度，示例使用 `swiz<128>`。更宽的重排能化解更宽的冲突模式，但要求分块范围与该宽度对齐。

**匹配规则。** 搬运上的 `<N>` 必须与 `mma.load.swiz<N>` 一致。若用不带重排的 `mma.load` 读取 `swiz<128>` 的数据，地址就会对不上，读到的是错位内容。编译器不强制这一配对——它是你自行维护的正确性不变式。（第 4 章已介绍，`mma.load.swiz<N>` 属于 MMA 装载家族。）

![Layout conflicts without swizzle vs a swizzle spreading accesses](../assets/images/ch07/fig2_swizzle_dark.png#only-dark)
![Layout conflicts without swizzle vs a swizzle spreading accesses](../assets/images/ch07/fig2_swizzle_light.png#only-light)

## 为何受限的接口反而有效：可表达范围 vs 性能

在暴露原始寻址的语言里，实现数据搬运的程序员拥有很大自由度：任意指针运算、可变步长访问、手工计算冲突规避、自定义重排公式。这种灵活性是双刃剑。可能的数据搬运模式空间巨大，但真正跑得快的子集很窄——它要求读取合并、暂存访问无冲突、重排对齐正确。任何一项出错都会静默地把吞吐降低一个数量级。

Croqtile 选择相反的路：把可表达的模式限制在**保证落在性能甜点区**的那一批。写下 `dma.copy` 时，编译器负责合并、无冲突布局与重排对齐。你没有办法意外写出跨步的非合并读取或冲突的暂存布局——语法根本不允许。

![Expressiveness vs performance: the unconstrained range includes many slow patterns; Croqtile's restricted range maps entirely to the performance sweet spot](../assets/images/ch07/fig8_expressiveness_dark.png#only-dark)
![Expressiveness vs performance: the unconstrained range includes many slow patterns; Croqtile's restricted range maps entirely to the performance sweet spot](../assets/images/ch07/fig8_expressiveness_light.png#only-light)

这个取舍很明确：你放弃写出任意数据搬运的能力，换来写出的每一种搬运都很快。

## 处理不规则访问

前面介绍的 tiling 原语——`chunkat`、`subspan(...).at(...)`——假设张量可被均匀划分为 tile。对于教科书式 GEMM（M、N、K 均为 tile 尺寸的倍数）这足够了。但生产级 kernel 很少如此整齐。Expert 批次从动态偏移开始。卷积窗口相互重叠。最后一个 K-tile 几乎从来不是 TILE_K 的整数倍。有些输入以扁平 buffer 形式到达，需要重塑后 MMA 才能消费。

Croqtile 为这些情况提供四种工具。每种工具都修改编译器解释张量寻址的方式，而不触碰流水线结构——DMA、MMA、事件与 swizzle 全部照常工作。

### 任意偏移窗口：`view` 与 `from`

`chunkat` 将张量划分为大小相等的规则网格。但如果你需要的切片不从 tile 边界开始怎么办？考虑 mixture-of-experts（MoE）kernel：每个 expert 处理不同数量的 token，因此每个 expert 的操作数起始行在运行时确定。你无法预先计算固定的 tile 网格。

`view(M, N).from(row, col)` 定义从任意 `(row, col)` 起算的 `M x N` 矩形——不要求对齐：

```choreo
patch = matrix.view(16, 16).from(37, 50);
```

这是从第 37 行、第 50 列开始的 `[16, 16]` 切片。原点可为任意运行时值。

![chunkat (aligned grid) vs view/from (arbitrary offset window)](../assets/images/ch07/fig4_view_from_dark.png#only-dark)
![chunkat (aligned grid) vs view/from (arbitrary offset window)](../assets/images/ch07/fig4_view_from_light.png#only-light)

`view` / `from` 的威力在于可组合性：切出窗口后，所有下游操作——`subspan`、`chunkat`、`dma.copy`——都把该窗口当作独立张量来处理。流水线不变；只是原点移动了：

```choreo
expert_lhs = lhs.view(expert_M, K).from(expert_offset, 0);
dma.copy expert_lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k) => shared;
```

在 MoE 堆栈中，每个 expert 的 token 批次从动态行 `expert_offset` 开始。`view` / `from` 调用在流水线顶部重接操作数；之后的一切——DMA、staging、MMA 循环——对每个 expert 运行相同的代码。

**何时使用。** 当张量在编译期可被 tile 均匀划分时，优先用 `chunkat`。当窗口的原点或范围在运行时确定，或几何形状与 tile 网格不对齐时，用 `view` / `from`。

### 步长 tile：`.subspan`、`.step` 与 `.at`

`subspan(M, K).at(i, j)` 选取逻辑 tile 索引 `(i, j)` 处、范围为 `[M, K]` 的 tile。默认情况下 tile 紧密排列：tile `(1, 0)` 紧接在 tile `(0, 0)` 之后。但某些布局在 tile 之间留有间距——要么是行间有 padding，要么是 stencil kernel 需要重叠窗口。

添加 `.step(sM, sK)` 覆盖步长：tile 相隔 `sM` 行与 `sK` 列，而非紧密排列：

```choreo
matrix.subspan(16, 16).step(32, 32).at(i, j);
```

![Packed tiling vs strided tiling with .step](../assets/images/ch07/fig5_subspan_step_dark.png#only-dark)
![Packed tiling vs strided tiling with .step](../assets/images/ch07/fig5_subspan_step_light.png#only-light)

Tile `(0,0)` 从 `(0,0)` 开始，但 tile `(1,0)` 从第 32 行（而非第 16 行）开始，留出 16 行间隙。省略 `.step` 时步长等于 tile 尺寸——即紧密排列情形。

**何时需要。** 跳过行间 padding 或保护带。步长小于 tile 范围的重叠 stencil 窗口（如 `16 x 16` tile 以 8 行为步进滑动）。匹配非稠密 tile-major 的外层内存布局——例如 tile 与元数据交错存放。

### 零填充：`.zfill`

每个分块核函数都面临同一边界情况：当 M 或 K 不是分块大小的整数倍时会怎样？沿该轴的最后一个分块是部分的——其中一些元素位于张量边界之外。读取这些地址属于未定义行为。

暴力修复是为边界 tile 增加特殊代码路径。但这破坏了 MMA 循环的均匀性，并添加了编译器无法优化掉的分支。`.zfill` 优雅地解决了这个问题：

```choreo
dma.copy.swiz<128> lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k).zfill
  => lhs_load_s;
```

`.zfill` 作用于 copy 的源侧：越界元素在目标 tile 中写为零。对于 GEMM，零对累加无贡献（0 × 任何值 = 0），因此 MMA 循环完全统一——内部 tile 与边界 tile 使用相同的代码——同时数学上仍然正确。无分支、无特殊情况、无性能惩罚。

![.zfill: zero-padding partial tiles at the tensor boundary](../assets/images/ch07/fig6_zfill_dark.png#only-dark)
![.zfill: zero-padding partial tiles at the tensor boundary](../assets/images/ch07/fig6_zfill_light.png#only-light)

### 布局重解释：`span_as`

有些 kernel 将数据作为扁平一维条带载入，但需要将其作为二维矩阵喂给 MMA。朴素方法会将数据拷贝到一个重新形状的 buffer——浪费 shared memory 和带宽。`span_as` 通过就地重解释已有 buffer 的形状来避免这一问题：

```choreo
flat_buffer.span_as([rows, cols])
```

元素个数不变；仅逻辑秩改变。无数据移动。

```choreo
strip_load = dma.copy data.chunkat(tile) => shared;
tile_2d = strip_load.data.span_as([tile_m, tile_k]);
ma = mma.load tile_2d.chunkat(_, iv_warp);
```

已载入的一维条带被暴露为二维矩阵供 `chunkat` 与 MMA 操作数 load 使用，无需额外拷贝或额外 shared memory。`rows * cols` 必须等于底层存储的 span 长度，否则编译器拒绝程序——该不变量在编译期检查，而非运行时。

![span_as: zero-copy shape reinterpretation from 1D to 2D](../assets/images/ch07/fig7_span_as_dark.png#only-dark)
![span_as: zero-copy shape reinterpretation from 1D to 2D](../assets/images/ch07/fig7_span_as_light.png#only-light)

## 本章小结

| 概念 | 语法 | 作用 |
|---------|--------|------|
| 分块搬运（第 2、6 章） | `dma.copy` / `dma.copy.swiz<N>` | 线程协作的分块传输 |
| Swizzle | `.swiz<N>`（copy）+ `mma.load.swiz<N>` | 搬运与消费者共用的重排布局 |
| 可表达范围权衡 | — | Croqtile 限制模式以确保合并、无冲突的传输 |
| 任意窗口 | `view(M,N).from(r,c)` | 参差不齐或由运行时定位的切片 |
| 步长 tiling | `.subspan().step().at()` | 非紧密布局、重叠 stencil |
| 部分 tile | `.zfill` | 越界元素零填充 |
| 形状重解释 | `span_as([dims])` | 对 staging buffer 的零拷贝形状重塑 |

## 新语法

| 语法 | 含义 |
|--------|---------|
| `dma.copy.swiz<N> src => dst` | 以字节计的重排宽度为 `N` 的搬运 |
| `dma.copy.swiz<N> src => dst` | 以字节计的重排宽度为 `N` 的搬运 |
| `mma.load.swiz<N> src` | 与 swizzle `N` 一致的 MMA 操作数 load |
| `tensor.view(M, N).from(r, c)` | 任意偏移的 `M x N` 窗口 |
| `.subspan(M, K).step(sM, sK).at(i, j)` | 步长 tile 选取 |
| `.zfill` | 在 copy 源侧对越界元素零填充 |
| `span_as([dims])` | 将线性存储重解释为带形状的张量 |

[下一章](ch08-cpp-interop.md)从纯鳄霸迈向 **C++ 互操作**：`__device__` 函数、**寄存器提示**、**预处理器保护**，以及需要贴近硬件时使用的**内联 PTX**。
