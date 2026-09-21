# Performance Tuning Demos

In this part, we iteratively optimize Croqtile GEMM, attention, and fused MoE kernels on H800 PCIe (SM90a, 114 SMs). Each is written as a continuous worklog: start from a correct baseline, measure against hardware limits, change one thing, re-measure, and tell the story of why each optimization works.

Before diving in, skim [Setting Up: TimerOption, TFLOPS, and HW Efficiency](setup-profiling.md) for how timing and efficiency are computed — every story uses the same harness.

## [Dense GEMM FP16: From Naive to Tuned](dense-gemm-fp16-from-naive.md)

Five-stage tutorial: naive → shared memory → Hopper TMA+WGMMA → warp specialization → production-tuned. Reaches **471 TFLOPS** (105% of cuBLAS) via a 28-iteration parameter sweep. Each stage introduces new Croqtile primitives with side-by-side generated CUDA. [Download kernel source files](dense-gemm-fp16-from-naive/assets/matmul_tutorial_kernels.tar.gz).

## [Sparse GEMM: FP16 and E4M3](sparse-gemm.md)

Structured 2:4 sparse GEMM at 4096 × 8192 × 8192. FP16: **368 → 655 TFLOPS** (+78%). E4M3: **671 → 1127 TFLOPS** (+68%). Metadata delivery, the `.co` vs `.cu` boundary, and the 3-stage discontinuity.

## [Block-Scaled GEMM FP8](blockscale-gemm-fp8.md)

FP8 E4M3 with per-block scaling: **397 → 621 TFLOPS** (+56%). TMA overlap with scale accumulation, N256 tiles, L2 promotion, and scale prefetch.

## [Flash Attention: Causal Prefill D=128](flash-attention-causal-prefill.md)

Updated for operation futures and generation-aware events: sequential DMA -> 1p2c TMA -> QK/PV overlap -> persistent task scheduling. Uses the current non-persistent v4 as the teaching example and explains persistent v3, reverse causal traversal, and deferred row-sum reduction. Includes freshly remeasured BF16 results for all three shapes, raw logs and source hashes, and standalone kernel/script downloads with matching CUDA and runtime headers.

## [Fused MoE FP8](fused-moe-fp8.md)

Fused Mixture-of-Experts end-to-end kernel for Qwen3.5-35B-A3B inference: **7.11 → 13.14 TFLOPS** (+85%). Kernel fusion (7→4 kernels), `parallel.async`, CUDA Graphs, L2 persistence, QSG load pipelining, and the `__cpp__` escape hatch.
