# 数据搬运：Tile 搬运引擎

第 1 章使用的是标量索引：选一个位置，读取输入，计算，再写回结果。这种方式很适合解释正确性，但并不是硬件真正想要的数据访问方式。GPU 获取和暂存的是连续的数据块，而不是彼此孤立的单个元素。GPU 编程真正困难的部分，通常并不是算术本身，而是如何组织这些块级传输，让计算单元在正确的时间、从正确的内存层级、以正确的形状拿到数据。

Croqtile 把这件事显式化了。它不把数据搬运当作索引背后的隐式副作用，而是提供一组明确的 tile 搬运语句：

- `dma.copy` 按原样搬运一个 tile。
- `dma.transp<...>` 在搬运过程中做维度置换。
- `dma.pad<...>` 在搬运过程中扩展 tile 并填充边界。
- `tma.copy` 在 tile 形状满足条件时，使用 Hopper 的 Tensor Memory Accelerator 完成同类搬运。

本章介绍这个模型，说明核心语法，并给出用户真正需要知道的规则，帮助你写出既正确又更容易 lower 到高性能路径的 DMA / TMA 代码。

![逐元素 vs 数据块编程模型对比](../assets/images/ch02/fig1_element_vs_block_dark.png#only-dark)
![逐元素 vs 数据块编程模型对比](../assets/images/ch02/fig1_element_vs_block_light.png#only-light)

*左：逐元素思维。右：按 tile 搬运到快速内存。*

<details>
<summary>动画版</summary>

<div markdown>
<video controls style="max-width: 100%; border-radius: 8px; margin: 1em 0;" class="only-dark">
  <source src="/croqtile-tutorial/assets/videos/ch02/anim1_element_vs_block_dark.mp4" type="video/mp4" />
</video>
<video controls style="max-width: 100%; border-radius: 8px; margin: 1em 0;" class="only-light">
  <source src="/croqtile-tutorial/assets/videos/ch02/anim1_element_vs_block_light.mp4" type="video/mp4" />
</video>
</div>

</details>

## DMA 语句家族

Croqtile 的 tile 搬运操作都遵循同一个基本形式：

```choreo
future = engine.operation.modifiers source => destination;
```

关键组成部分如下：

| 部分 | 含义 |
|------|------|
| `future` | 可选的传输结果句柄。用 `future.data` 访问搬运后的 tile，用 `future.span` 查询它的形状。 |
| `engine` | `dma` 表示软件协作复制，`tma` 表示 Hopper TMA 复制。 |
| `operation` | `copy`、`transp<...>` 或 `pad<...>`。 |
| `modifiers` | 可选修饰符，例如 `.async`、`.zfill` 或 `.swiz<N>`。 |
| `source => destination` | 源视图，以及目标内存或目标视图。 |

最小示例：

```choreo
f = dma.copy input.chunkat(block) => shared;
g = dma.transp<1, 0> matrix => local;
h = dma.pad<{2, 1}, {3, 2}, {0, 0}, 0> tile => shared;
```

与第 1 章相比，最关键的变化是：这些语句搬运的是有形状的区域，而不是单个元素。

## 第一个例子：通过 Shared Memory 做分块加法

下面把第 1 章的加法改写成通过 shared memory 暂存 tile：

```choreo
__co__ s32 [64, 128] tiled_add_2d(s32 [64, 128] lhs, s32 [64, 128] rhs) {
  s32 [lhs.span] output;

  parallel {tr, tc} by [4, 8] {
    lhs_load = dma.copy lhs.chunkat(tr, tc) => shared;
    rhs_load = dma.copy rhs.chunkat(tr, tc) => shared;

    foreach {i, j} in lhs_load.span
      output.at(tr # i, tc # j) =
          lhs_load.data.at(i, j) + rhs_load.data.at(i, j);
  }

  return output;
}
```

算术本身仍然是逐元素加法。变化的是数据路径：

1. `chunkat(tr, tc)` 从每个输入里选出一个逻辑 tile。
2. `dma.copy ... => shared` 把这个 tile 暂存到线程块可见的快速内存里。
3. 内层循环读取的是 `lhs_load.data` 和 `rhs_load.data`，而不是全局内存。
4. `tr # i` 和 `tc # j` 把 tile 内坐标映射回全局输出坐标。

这就是 Croqtile 的核心模式：选出一个 tile，搬运它，在搬运后的 tile 上计算。

![分块加法：加载、计算、存储流程](../assets/images/ch02/fig2_tiled_add_dark.png#only-dark)
![分块加法：加载、计算、存储流程](../assets/images/ch02/fig2_tiled_add_light.png#only-light)

*先加载一个 tile，在暂存后的数据上计算，再用全局坐标写回结果。*

<details>
<summary>动画版</summary>

<div markdown>
<video controls style="max-width: 100%; border-radius: 8px; margin: 1em 0;" class="only-dark">
  <source src="/croqtile-tutorial/assets/videos/ch02/anim2_tiled_add_dark.mp4" type="video/mp4" />
</video>
<video controls style="max-width: 100%; border-radius: 8px; margin: 1em 0;" class="only-light">
  <source src="/croqtile-tutorial/assets/videos/ch02/anim2_tiled_add_light.mp4" type="video/mp4" />
</video>
</div>

</details>

## `chunkat`、`subspan` 与 `view().from()`：三种选 Tile 的方式

Croqtile 中常见的 tile 选择方式有三种。

### `chunkat(...)`：等分切块

```choreo
lhs.chunkat(tr, tc)
```

`chunkat` 会根据周围的并行结构，把每个维度均匀切成若干块。如果 `lhs` 的形状是 `[64, 128]`，kernel 使用 `parallel {tr, tc} by [4, 8]`，那么 `lhs.chunkat(tr, tc)` 选中的就是一个 `[16, 16]` tile。

当你的 tiling 正好由并行划分自然决定时，`chunkat` 是最合适的工具。

![chunkat 二维分块选择语义](../assets/images/ch02/fig3_chunkat_dark.png#only-dark)
![chunkat 二维分块选择语义](../assets/images/ch02/fig3_chunkat_light.png#only-light)

*`chunkat(1, 3)` 从规则 tiling 网格中选出一个等分 tile。*

<details>
<summary>动画版</summary>

<div markdown>
<video controls style="max-width: 100%; border-radius: 8px; margin: 1em 0;" class="only-dark">
  <source src="/croqtile-tutorial/assets/videos/ch02/anim3_chunkat_dark.mp4" type="video/mp4" />
</video>
<video controls style="max-width: 100%; border-radius: 8px; margin: 1em 0;" class="only-light">
  <source src="/croqtile-tutorial/assets/videos/ch02/anim3_chunkat_light.mp4" type="video/mp4" />
</video>
</div>

</details>

### `subspan(...).at(...)`：在 tile 网格上显式给出 tile 形状

```choreo
lhs.subspan(64, 64).at(block_m, iv_k)
```

`subspan` 指定 tile 的形状，`.at(...)` 指定你要的是哪个 tile 实例。当 tile 形状本身是算法显式选出来的，例如 GEMM 中的 `[64, 64]` tile，而不是想从 `parallel by ...` 中隐式推导时，它更合适。

### `view(...).from(...)`：显式形状，加上显式元素偏移

```choreo
lhs.view(TILE_M, TILE_K).from(offset_m, offset_k)
```

`view(...).from(...)` 是本章里最通用的选择形式。`view(...)` 表示你要一个怎样形状的区域，`.from(...)` 给出这个区域在元素坐标空间中的起点。

当 tile 起点天然就是元素偏移，而不是 tile 索引，或者这个区域本来就不适合描述成均匀分块时，它就是正确工具。对于动态、非规则的访问模式，`view(...).from(...)` 往往比 `chunkat(...)` 更自然，甚至是唯一合理的写法。

可以这样记这三种形式：

- 当周围并行结构已经定义了规则等分 tiling 时，用 `chunkat(...)`。
- 当你想显式指定 tile 形状和 tile 坐标时，用 `subspan(...).at(...)`。
- 当你想显式指定 tile 形状和元素级起始偏移时，用 `view(...).from(...)`。

## Future、`.data` 与 `.span`

DMA 或 TMA 语句的结果是一个 future 风格的句柄：

```choreo
f = dma.copy input => shared;
```

这个句柄提供两个重要能力：

- `f.data`：搬运后的 tile，本质上是一个带 shape 的张量视图。
- `f.span`：这个 tile 的形状。

示例：

```choreo
f = dma.copy input => shared;
foreach idx in [f.span]
  output.at(idx) = f.data.at(idx);
```

即便是同步复制，只要你希望在后续代码中继续传递这个 tile，给它起名字也是值得的。

![Future 与 .data](../assets/images/ch02/fig_future_data_dark.png#only-dark)
![Future 与 .data](../assets/images/ch02/fig_future_data_light.png#only-light)

*传输结果是一个句柄。通过 `.data` 读取搬运后的 tile，通过 `.span` 读取它的形状。*

## 同步复制与异步复制

默认情况下，复制是同步的：

```choreo
f = dma.copy input => shared;
dma.copy f.data => output;
```

如果你希望传输在后台进行，可以加上 `.async`：

```choreo
f = dma.copy.async input => shared;
wait f;
dma.copy f.data => output;
```

TMA 也一样：

```choreo
f = tma.copy.async input => shared;
wait f;
tma.copy f.data => output;
```

规则很简单：如果你要读取异步传输得到的 `f.data`，就必须先 `wait`。

后面的章节会利用异步复制去重叠加载与计算。本章只需要把契约记住：`.async` 会创建一个进行中的传输，`wait` 让它变成可安全消费的数据。

## 其他操作：`transp` 与 `pad`

`copy` 只是这个家族里最简单的成员。

### `dma.transp<...>`

```choreo
fa = dma.transp<1, 0, 2> a => local;
dma.copy fa.data => o;
```

它在搬运过程中就完成维度置换。不是先复制，再转置，而是直接得到转置布局的数据。

### `dma.pad<...>`

```choreo
f = dma.pad<{2, 1}, {3, 2}, {0, 0}, V>.async input => shared;
wait f;
dma.copy f.data => output;
```

`pad` 会生成一个更大的目标 tile。模板参数描述低端 padding、高端 padding、内部步长式 padding，以及填充值。需要显式边界值时，用 `pad`；它不是单纯的越界保护。

## `#`：组合与范围

Croqtile 里 `#` 有两种相关但不同的用法。

### 组合：`outer # inner`

```choreo
output.at(tr # i, tc # j)
```

它把一个 tile 坐标与 tile 内坐标组合成完整张量中的坐标。

可以把 `tr # i` 读成“第 `tr` 个 tile 里的第 `i` 行”。

![# 组合运算符](../assets/images/ch02/fig_compose_dark.png#only-dark)
![# 组合运算符](../assets/images/ch02/fig_compose_light.png#only-light)

*把 tile 索引与 tile 内偏移组合成全局坐标。*

### 范围：`#name`

```choreo
foreach i in [128 / #tile]
```

这里 `#tile` 表示 `tile` 这个轴的范围，也就是该维度上有多少个 tile。如果 `parallel tile by 8`，那么 `#tile` 就是 `8`。

![# 范围运算符](../assets/images/ch02/fig_extent_dark.png#only-dark)
![# 范围运算符](../assets/images/ch02/fig_extent_light.png#only-light)

*前缀 `#` 表示查询范围，中缀 `#` 表示组合坐标。*

## `span` 与 `span(i)`

`tensor.span` 给出整个张量的完整形状，`tensor.span(i)` 给出单独某个维度。

```choreo
s32 [lhs.span(0), rhs.span(1)] output;
```

这在数据搬运代码里非常常见，因为输出形状往往由不同输入组合而来，而搬运出的 tile 也经常需要保留或重新计算 shape 信息。

![span(i)——选取单个维度](../assets/images/ch02/fig_span_dark.png#only-dark)
![span(i)——选取单个维度](../assets/images/ch02/fig_span_light.png#only-light)

*`span` 表示完整形状，`span(i)` 表示单独一个轴。*

## 内存限定符：`global`、`shared` 与 `local`

搬运语句的目标位置，也决定了这个 tile 最终放在哪一层内存中。

- `global`：设备内存。大、慢，但整个设备都可见。
- `shared`：线程块可见的片上内存。协作式 tile staging 的典型落点。
- `local`：线程私有存储。适合小型私有 tile 或非协作式使用。

```choreo
f0 = dma.copy input => shared;
f1 = dma.copy matrix_tile => local;
dma.copy out_tile => output;
```

对于性能敏感的 kernel，`shared` 是最关键的一层。编译器的协作式 tiled lowering 主要针对 global-to-shared 与 shared-to-global 搬运。`local` 复制仍然有用，但它不是协作式块搬运的主要高性能路径。

![内存限定符 → GPU 硬件](../assets/images/ch02/fig_memory_hierarchy_dark.png#only-dark)
![内存限定符 → GPU 硬件](../assets/images/ch02/fig_memory_hierarchy_light.png#only-light)

*Croqtile 的内存限定符直接映射到 GPU 的内存层级。*

## 部分 Tile、边界保护与 `.zfill`

真实 kernel 经常会遇到尾块。这里最重要的区别不是“最后一个 tile 是否不完整”，而是**源 tile 的形状**与**目标 tile 的形状**是否相同。

例如：

```choreo
dma.copy.swiz<32>.zfill
  lhs.view(ROWS_M, TILE_K).from(base_m, base_k)
    => lhs_load_s;
```

这里 `lhs_load_s` 的形状可能是 `[WARP_M, TILE_K]`，而源视图的形状是 `[ROWS_M, TILE_K]`，并且在最后一个 tile 上 `ROWS_M` 可能小于 `WARP_M`。这时就存在真正的形状不匹配，所以 `.zfill` 的含义是：把有效的源行复制进去，再把目标 tile 剩余的那些行写成 0。

这才是 `.zfill` 正确的使用方式。

如果形状本来就相同，会发生什么：

- 如果源视图和目标 tile 形状相同，编译器会自动为运行时尾块生成边界保护。
- 这种情况下它会生成掩码，避免读写越界元素。
- 对于把一个 `[100]` 张量切成若干个 `[64]` tile 这样的例子，这已经足够了：最后一个 tile 自然只有 36 个有效元素。

因此，`.zfill` **不是**“任何部分 tile 都要用”的通用机制。它只适用于这样一种情况：目标 tile 比有效源区域更大，而且目标中多出来的那部分必须被明确写成 0。

为什么 `.zfill` 重要：

- 如果消费者只读取有效区域，编译器自动生成的掩码通常已经足够。
- 如果消费者会读取整个目标 tile，尤其是 MMA fragment，那么无效区域必须具有确定的值。
- `.zfill` 会把这部分额外区域写成 0。

当有效源区域小于目标 tile，并且后续消费者会把整个目标 tile 当成完整稠密 tile 读取时，就应该使用 `.zfill`。这在带动态行数或边缘 padding 的 GEMM 流水线里非常常见。

如果源和目标形状已经相同，却仍然手动写 `.zfill`，编译器可能会警告这个 `.zfill` 是冗余的。反过来，如果形状不匹配，编译器也可能提示你需要 `.zfill`，否则未覆盖区域会留下未定义值。

## `.swiz<N>`：为消费者选择 Shared Memory 布局

Swizzle 主要决定 tile 在 shared memory 中应该以什么布局存放，以便后续操作高效读取。

典型的 GEMM staging 代码如下：

```choreo
dma.copy.swiz<128> lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k) => lhs_load_s;
ma = mma.load.swiz<128> lhs_load_s.chunkat(_, iv_warp);
```

加载时使用的 swizzle 与计算侧使用的 swizzle 通常应该保持一致。如果你用 `.swiz<128>` 把 tile 暂存到 shared，那么消费者一般也应该用匹配的 swizzle 去读取它。

本章只介绍这个概念。MMA 相关章节会更深入地说明不同矩阵布局该如何搭配不同的 swizzle 值。

## DMA 与 TMA

二者都搬运有形状的 tile。区别在于它们怎么做，以及各自的约束是什么。

| | `dma` | `tma` |
|---|---|---|
| 实现方式 | 软件协作复制 | Hopper Tensor Memory Accelerator |
| 硬件要求 | 通用 GPU 路径 | SM90+ |
| 适用场景 | 动态或不规则 tile，可移植代码 | 固定形状、高带宽 staging |
| 常见目标位置 | `shared` | `shared` |
| 是否支持异步 | 是 | 是 |

TMA 示例：

```choreo
f = tma.copy.async lhs.subspan(64, 64).at(block_m, iv_k) => shared;
wait f;
```

实践上的规则很简单：

- 当 tile 尺寸是动态的、不规则的，或者你要可移植路径时，用 `dma`。
- 当你在 Hopper 上、tile 形状固定、并希望用硬件加速搬运时，用 `tma`。

### TMA 最重要的约束

TMA 的 tile 形状必须是 host 可计算的。运行时 tile 索引本身没有问题：

```choreo
tma.copy lhs.subspan(64, 64).at(block_m, iv_k) => shared;
```

但如果 tile 形状本身依赖 kernel 运行时值，就不行：

```choreo
// 如果 TILE_M 只有在 kernel 内才知道，这种写法不适用于 TMA
tma.copy lhs.subspan(TILE_M, 64).at(block_m, iv_k) => shared;
```

如果 tile 尺寸真的依赖 kernel 运行时状态，请改用 `dma.copy`。

## 如何写出更容易 Lower 到好路径的 DMA / TMA

你不需要理解完整 lowering pass 才能写出好代码，但有几条规则很重要。

### 对高吞吐 DMA 来说

- 优先使用 `global <-> shared` 搬运来做协作式 staging。
- 使用足够多的参与线程。常见的协作组大小通常至少是一个 warp。
- 保持源和目标的秩一致。
- 使用普通的按字节寻址元素类型，至少 1 字节以上。
- `local` 更适合私有 tile，不应该默认当成主要性能路径。

如果这些条件不满足，编译器仍然能生成正确的复制，但可能会退化到更简单的路径。

### 对 TMA 来说

- 在 SM90+ 上使用它。
- 保持 tile 形状从 host 的角度看是固定的。
- 把 tile 暂存到 shared memory。
- 当部分 tile 会被完整 tile 消费者读取时，添加 `.zfill`。

## 小结清单

读完这一章后，下面这些语法应该分别表示：

| 语法 | 含义 |
|------|------|
| `dma.copy src => shared` | 把一个 tile 搬运到 shared memory。 |
| `tma.copy src => shared` | 用 Hopper TMA 把一个 tile 搬运到 shared memory。 |
| `dma.copy.async ...` | 启动一个非阻塞 DMA 传输。 |
| `wait f` | 等待 future `f` 可安全消费。 |
| `future.data` | 访问搬运后的 tile。 |
| `future.span` | 访问搬运后 tile 的形状。 |
| `dma.transp<...>` | 在传输过程中做维度置换。 |
| `dma.pad<...>` | 在传输过程中按显式填充值做 padding。 |
| `.zfill` | 当有效源区域小于目标 tile 时，将未覆盖部分写成 0。 |
| `.swiz<N>` | 为 shared memory 选择 swizzle 布局。 |
| `chunkat(...)` | 从规则 tiling 中选出一个等分 tile。 |
| `subspan(...).at(...)` | 用显式 tile 形状和 tile 坐标选出一个 tile。 |
| `view(...).from(...)` | 用显式形状和显式元素偏移选出一个 tile。 |
| `tile # i` | 把 tile 坐标和 tile 内坐标组合成全局坐标。 |
| `#tile` | 查询 `tile` 这个轴的范围。 |

下一章会在这个基础上引入加载与计算的重叠。一旦你开始把 kernel 看成“反复 staging tile，再在 staged tile 上计算”的循环，双缓冲与流水线就会成为很自然的延伸，而不是全新的概念。
