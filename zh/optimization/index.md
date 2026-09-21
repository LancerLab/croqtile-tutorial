# 性能调优演示

本部分在 H800 PCIe（SM90a，114 个 SM）上对 Croqtile 的 GEMM、attention 与融合 MoE 内核进行迭代优化。每个内核均以连续工作日志的形式撰写：从正确的基线出发，对照硬件上限进行测量，每次只改一处，重新测量，并说明每项优化为何有效。

在深入阅读之前，请先浏览 [环境搭建：TimerOption、TFLOPS 与硬件效率](setup-profiling.md)，了解计时与效率的计算方式。具体的计时范围和对照条件以各案例的说明为准。

## [稠密 GEMM FP16](dense-gemm-fp16.md)

半精度矩阵乘法从 **208 → 382 TFLOPS**（+83%），与 cuBLAS 持平。分块几何、流水线深度、split-output 1p2c，以及 WN=168 的占用率断崖。

## [稀疏 GEMM：FP16 与 E4M3](sparse-gemm.md)

结构化 2:4 稀疏 GEMM，规模 4096 × 8192 × 8192。FP16：**368 → 655 TFLOPS**（+78%）。E4M3：**671 → 1127 TFLOPS**（+68%）。元数据传递、`.co` 与 `.cu` 的边界，以及三阶段不连续性。

## [块缩放 GEMM FP8](blockscale-gemm-fp8.md)

带每块缩放的 FP8 E4M3：**397 → 621 TFLOPS**（+56%）。TMA 与缩放累加的重叠、N256 分块、L2 提升与缩放预取。

## [Flash Attention：因果预填充 D=128](flash-attention-causal-prefill.md)

按当前操作 future 和带代际事件的写法，依次介绍顺序 DMA、1p2c TMA、QK/PV 重叠及持久化任务调度。以非持久化 v4 讲解计算流水线，再分析持久化 v3、逆序因果遍历和延迟行和归约。包含三个尺寸下重新测量的 BF16 性能表、对比图、原始日志与源码哈希，并提供含 CUDA 和 runtime 头文件的独立内核及脚本下载包。
