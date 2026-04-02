# 角色专化与控制流

第 4 章的矩阵乘内核在结构上存在局限：block 内每条线程执行同一程序——加载、相乘、存储，周而复始。张量核心忙于乘法时，内存子系统空闲；DMA 取下一块 K 条带时，张量核心无事可做。加载与计算轮流进行，二者均无法达到满吞吐。

这并非 GPU 所特有的问题，而是**流水线阶段串行化**的普遍瓶颈。凡由单一工作者在两种活动——读数据与处理数据——之间交替执行的系统，都会在每次切换时损失时间。经典解法，无论见于 CPU 线程池、Unix 管道抑或工厂流水线，均为**角色专化**：为不同阶段指派不同工作者，使各人专注一职，并在时间上重叠各阶段。

GPU 的特殊之处在于，CUDA 的默认执行模型为 **SPMD**（Single Program, Multiple Data）：warp 内每条线程执行同一指令流。要在单内核内重叠访存与计算，要么依赖指令级交错（硬件调度器对大 tile 往往调度不佳），要么采用显式 **warp specialization**，由程序员手工划分 warp 并以屏障协同。在原始 CUDA 中，这意味着直接使用 `__syncthreads()`、共享内存标志位，并仔细推断各 warp 的职责。

鳄霸（Croktile）采取不同路径：不以控制流技巧隐式刻画角色边界，而将其提升为**一等语言构造**——`inthreads.async` 在编译期为不同线程子集指派不同指令流。编译器可为各角色生成真正分离的程序，运行时则并发调度之。下图对比二者差异：

![Uniform vs role-specialized execution: sequential alternation vs overlapping stages](../assets/images/ch05/fig1_role_comparison_dark.png#only-dark)
![Uniform vs role-specialized execution: sequential alternation vs overlapping stages](../assets/images/ch05/fig1_role_comparison_light.png#only-light)

*左：单一 warpgroup 在 DMA 与 MMA 之间交替——无重叠。右：两个 warpgroup 承担静态角色——生产者 DMA 与消费者 MMA 并发执行，墙钟时间大致减半。*

鳄霸提供三种控制流原语，各针对一类用途：

- **`inthreads.async`** —— **静态角色划分**：在编译期将不同程序指派给不同线程子集。在单内核内类比 MPMD（Multiple Program, Multiple Data）。
- **`if`** —— **受控执行**：运行时谓词，全体线程求值条件，分歧线程被屏蔽。标准 SPMD 控制流。
- **`shared event`** / **`wait`** / **`trigger`** —— **角色间信令**：使静态划分后的角色得以安全通信的协调机制。

## 以 `inthreads.async` 实现静态角色划分

`inthreads.async (condition)` 的含义是：仅当 `condition` 为真时，相应线程**其程序中才包含本代码块**。它并非「每条线程都求值条件，部分跳过 body」——后者由 `if` 承担。二者区分具有根本性：

- **`inthreads.async`**：编译器为各角色生成独立指令流。属于假子集的线程永远见不到 body——其不在其二进制中。`.async` 后缀表示各角色在不同硬件资源（不同 warpgroup、不同功能单元）上**并发且独立**执行。
- **`if`**：单一指令流、单一程序。全体线程求值谓词；谓词为假的线程被屏蔽。warp 内分歧会使一侧停顿。

何以需要二者？因其解决的问题不同。`inthreads.async` 面向**持久角色指派**——生产者在内核整个生命周期内保持为生产者。`if` 面向**数据相关决策**——若索引越界则跳过本 tile。不会用 `if` 做 warp specialization（它不产生真正的并发），也不会用 `inthreads.async` 做边界检查（它是编译期构造，而非运行时谓词）。

若无 `.async`，`inthreads` 将表示顺序、阻塞的角色执行——线程子集轮流执行。`.async` 修饰符正是上图所示重叠流水线执行得以成立的关键。

矩阵乘的典式模式为 **一生产者加一消费者（1P1C）**：

```choreo
parallel p1 by 2 : group-4 {

  inthreads.async (p1 == 0) {
    // producer: only warpgroup 0 runs this
    // issue DMA / TMA loads, fill shared memory
  }

  inthreads.async (p1 == 1) {
    // consumer: only warpgroup 1 runs this
    // run MMA on shared memory, accumulate results
  }
}
```

**`parallel p1 by 2 : group-4`** —— 两个 warpgroup，每个含四个 warp（每个 warpgroup 128 条线程），由 `p1` 索引。

**`inthreads.async (p1 == 0)`** —— warpgroup 0 编译并执行生产者 body；warpgroup 1 永远不包含此代码。

**`inthreads.async (p1 == 1)`** —— warpgroup 1 执行消费者 body。两块代码为共享地址空间下的分离程序。

## 协调角色：事件概览

静态角色划分得到两个并发程序——但二者共享同一片共享内存。生产者写入、消费者读取。若无协调，消费者可能在生产者写完 tile 之前就读取（竞态），或生产者可能在消费者尚未读完时覆盖 tile（数据冒险）。

鳄霸以 **event** 作为角色间信令机制。event 为在共享内存中声明的轻量级同步令牌：

```choreo
shared event full;
shared event empty;
```

生产者在写完 tile 后调用 `trigger full`，表示「数据已就绪」。消费者在读取前调用 `wait full`，阻塞直至信号到达。对称地，消费者读完之后触发 `empty`（缓冲区可复用），生产者在写下一 tile 之前于 `empty` 上等待。

这是 **基于信用的有界缓冲区** 协议——与网络流控及操作系统有界队列所用模式相同。`full` 表示「数据可用」信用；`empty` 表示「缓冲区空闲」信用。

第 6 章将据此展开完整的双缓冲与软件流水矩阵乘。此处要点在于：event 是 `inthreads.async` 各角色之间的黏合剂——将两个独立程序接成协调的流水线。

## 实践中的角色专化：1P1C 矩阵乘

以下展示该划分如何置于 Hopper 矩阵乘之中。基于 event 的同步有意省略；[第 6 章](ch06-synchronization.md) 给出完整流水线协议。此处关注职责分工：

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  parallel {block_m, block_n} by [cdiv(M, MATMUL_WARP_M), cdiv(N, MATMUL_WARP_N)] : block {
    shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
    shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
    shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;

    parallel p1 by 2 : group-4 {

      inthreads.async (p1 == 0) {
        // Producer: walk K, load tiles into shared
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          dma.copy lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
          dma.copy rhs.chunkat(block_n, iv_k) => rhs_load_s;
        }
      }

      inthreads.async (p1 == 1) {
        // Consumer: walk K, MMA on loaded tiles
        mc = mma.fill.f16 0.0f;
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
            ma = mma.load lhs_load_s.chunkat(_, iv_warp);
            mb = mma.load rhs_load_s.chunkat(_, iv_warp);
            mma.row.row mc, ma, mb;
          }
        }
        mma.store mc, output_s;
        dma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
      }
    }
  }
}
```

**生产者 `foreach`** —— 以 `cdiv(K, MATMUL_TILE_K)` 步沿 K 维遍历；仅 warpgroup 0 向 `lhs_load_s` 与 `rhs_load_s` 发出 `dma.copy`。

**消费者 `mma.fill` / `mma.row.row` / `mma.store`** —— warpgroup 1 从不发出上述 DMA 填充；仅读共享内存、在 `mc` 中累加并写出结果 tile。

**缺失的协调** —— 两侧各自独立沿 K 循环。消费者读取时假定各 K 条带已就绪；使该假定成立即为同步（见[第 6 章](ch06-synchronization.md)）。

## 以 `if` 实现受控执行

有时需要每条线程在运行时求值的谓词。鳄霸的 `if` 语义与 C 类似：

```choreo
if (tile_id < total_tiles) {
  // only execute this body when the condition is true
}
```

作用域内全体线程均测试条件；条件为假的线程跳过 body。这与 `inthreads.async` 相反：单一程序、分歧执行——而非两套分离程序。

该模式最常见于**持久化内核（persistent kernel）**：循环迭代次数可能使部分 block 多出一轮不对应真实 tile 的「填充」迭代。

## 持久调度与 `if` 守卫

在第 3–4 章中，网格随问题规模增长：大致每块输出 tile 对应一个 block。对大矩阵而言，启动次数可能极大。GPU 以**波次（waves）** 运行 block；末波往往使部分 SM 仅部分占用——**尾部利用率不足**。

**持久化内核** 将启动规模固定（常接近 SM 数量），并使每个 block 在多个 tile 上迭代：

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs, global f16 [M, N] output) {
  int total_tiles = cdiv(M, MATMUL_WARP_M) * cdiv(N, MATMUL_WARP_N);

  parallel block_id by NUM_SMS : block {
    shared f16 [MATMUL_WARP_M, MATMUL_TILE_K] lhs_load_s;
    shared f16 [MATMUL_WARP_N, MATMUL_TILE_K] rhs_load_s;
    shared f16 [MATMUL_WARP_M, MATMUL_WARP_N] output_s;

    foreach {tile_iter} in [cdiv(total_tiles, NUM_SMS)] {
      tile_id = tile_iter # block_id;

      if (tile_id < total_tiles) {
        block_m = tile_id / cdiv(N, MATMUL_WARP_N);
        block_n = tile_id % cdiv(N, MATMUL_WARP_N);

        mc = mma.fill.f16 0.0f;
        foreach {iv_k} in [cdiv(K, MATMUL_TILE_K)] {
          dma.copy lhs.subspan(MATMUL_WARP_M, MATMUL_TILE_K).at(block_m, iv_k) => lhs_load_s;
          dma.copy rhs.subspan(MATMUL_WARP_N, MATMUL_TILE_K).at(block_n, iv_k) => rhs_load_s;

          foreach {iv_warp} in [cdiv(MATMUL_TILE_K, MATMUL_WARP_K)] {
            parallel p by 1 : group-4 {
              ma = mma.load lhs_load_s.chunkat(_, iv_warp);
              mb = mma.load rhs_load_s.chunkat(_, iv_warp);
              mma.row.row mc, ma, mb;
            }
          }
        }
        mma.store mc, output_s;
        dma.copy output_s => output.subspan(MATMUL_WARP_M, MATMUL_WARP_N).at(block_m, block_n);
      }
    }
  }
}
```

**`parallel block_id by NUM_SMS : block`** —— 工作者数量固定；`block_id` 标识本持久工作者身份，而非单一输出 tile。

**`foreach {tile_iter} in [cdiv(total_tiles, NUM_SMS)]`** —— 各 block 遍历其份额的迭代；向上取整可能使部分 block 多一次迭代。

**`tile_id = tile_iter # block_id`** —— 将迭代与 block 索引组合，在线性 tile 列表上条带化（与第 2 章相同的 `#` 运算符，此处用于调度）。

**`if (tile_id < total_tiles)`** —— 当条带越过最后一个真实 tile 时跳过 DMA、MMA 与存储。此为运行时守卫，而非角色划分。

![Persistent kernel: striped tiles, block colors, and if guard for padding](../assets/images/ch05/fig2_persistent_kernel_dark.png#only-dark)
![Persistent kernel: striped tiles, block colors, and if guard for padding](../assets/images/ch05/fig2_persistent_kernel_light.png#only-light)

### 数据相关网格与持久网格之对比

| 方面 | 每 tile 一块 | 持久化（`NUM_SMS` 个 block） |
|--------|-------------------|-------------------------------|
| 网格规模 | 随问题增长 | 固定 |
| 尾部利用 | 末波可能使 SM 空闲 | 各 SM 保持忙碌 |
| 额外构造 | 最少 | `total_tiles`、`tile_iter # block_id`、`if` |
| 复杂度 | 较低 | 较高 |

两种布局本身均不改变数学结果；在浮点结合律意义下二者一致。当 `total_tiles` 远大于 SM 数量时，持久调度往往更有收益。

## `parallel.async` 与 `stream s`：非阻塞启动

上文均在 kernel 内部运行。有时所需控制位于**主机侧**：启动网格而不阻塞主机线程，或将不同网格绑定到不同 CUDA 流以使 GPU 上并发执行。

```choreo
parallel.async {px, py} by [grid_m, grid_n] : block {
  stream s;
  // kernel body
}
```

**`parallel.async`** 立即将控制权交还主机——kernel 已入队，主机不等待其完成。这相当于在非默认流上使用 `cudaLaunchKernel` 的鳄霸写法。

**`stream s`** 置于块体内部时，将 kernel 固定到 CUDA 流 `s`。若 SM 资源充足，不同流上的多个 `parallel.async` 块可在 GPU 上重叠。若无 `stream s`，默认流会使各次启动串行化。

此为**主机编排**，而非 kernel 内控制流。它不能替代 `inthreads.async` 的角色划分或 `if` 的运行时谓词——它决定的是相对其他网格，本网格于*何时*、在*何处*运行。

## 新语法

| 语法 | 含义 |
|--------|---------|
| `inthreads.async (condition)` | 静态角色划分——仅满足 `condition` 的线程包含本块 |
| `if (expr) { ... }` | 受控执行——运行时条件，当 `expr` 为假时跳过 body |
| `shared event name` | 在共享内存中声明 event 令牌 |
| `trigger name` | 表明某条件已满足（例如「数据就绪」） |
| `wait name` | 阻塞直至对应 `trigger` 发生 |
| `tile_id = tile_iter # block_id` | 将迭代索引与 block 索引组合以实现 tile 条带化 |
| `int total_tiles = expr` | 鳄霸函数中的局部整型 |
| `parallel.async ... : block` | 非阻塞异步 kernel 启动 |
| `stream s` | 将 kernel 体绑定到 CUDA 流 `s` |

## 本章小结

| 主题 | 要点 |
|-------|----------|
| 流水线串行化 | 单工作者在加载/计算间交替浪费时间；角色专化使各阶段重叠 |
| 静态角色划分 | `inthreads.async` —— kernel 内编译期 MPMD；不同线程子集对应不同程序 |
| 受控执行 | `if` —— 运行时 SPMD 谓词；全体线程求值，分歧线程被屏蔽 |
| Event（预告） | `shared event` / `wait` / `trigger` —— 角色间信令；基于信用的有界缓冲区协议 |
| 持久化内核 | 固定 `NUM_SMS` 个 block、线性 tile id、以 `#` 条带化、以 `if` 守卫 |
| 主机编排 | `parallel.async` / `stream s` —— 与 kernel 内专化正交 |

上文 1P1C 骨架并不完整：若无 `wait` / `trigger`，消费者可能在生产者写入之前读取。[第 6 章](ch06-synchronization.md) 补充完整同步协议——event、`swap` 与双缓冲——从而使流水线安全并以满吞吐运行。
