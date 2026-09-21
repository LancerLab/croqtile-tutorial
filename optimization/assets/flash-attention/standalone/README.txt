# Standalone Flash Attention: Causal Prefill D=128

This package contains the four Croqtile/Choreo kernels used in the tutorial,
their generated CUDA, the required runtime headers, a C++ host harness, and
FA3/TileLang/Triton comparison scripts. No compiler repository checkout or
benchmark directory is needed to compile the included CUDA and run it.

The fixed workload is BF16 BSHD causal forward attention with B=4, H=32,
D=128, and equal Q/KV lengths of 4096, 8192, and 16384. The persistent v3
uses 114 CTAs and is tuned for H800 PCIe. These examples do not implement
arbitrary sequence tails, GQA, dropout, or backward attention.

## Prerequisites

- An NVIDIA Hopper GPU and a working CUDA driver. The published measurements
  use H800 PCIe, SM90a, 114 SMs.
- CUDA Toolkit with NVCC, a C++17 host compiler, Python 3.10 or newer, and
  CUTLASS headers. The reference environment uses CUDA 13.0.88 and CUTLASS
  commit `b50d8fd7196e1c5801c1bd8edf1d64e4bf9add6c`.
- Only the comparison backends need Python GPU packages: the reference
  environment uses PyTorch 2.11.0+cu130, Hopper FlashAttention-3 3.0.0
  (`flash_attn_interface`), TileLang 0.1.9, and Triton 3.6.0.
- Triton+WS uses a separate environment with Triton
  `3.8.0+gita59d32fd` from main, plus PyTorch 2.10.0+cu128. It uses
  `tl.range(warp_specialize=True)` and device tensor descriptors. This
  replaces the removed aref launch options and is not an aref measurement.
  The WS kernel uses one masked causal loop because this compiler revision
  fails on the older split off-diagonal/diagonal loops. It checks that the
  compiler emitted three warp groups (384 threads), not a non-WS fallback.

FA3 here is the Hopper implementation, not the `flash_attn` FA2 package.
Python environments are explicit arguments; no activation script or absolute
home-directory path is embedded in this package. Optional packages are not
installed automatically.

## Build and run

From the extracted `flash-attention-standalone` directory:

```bash
export CUDA_HOME=/usr/local/cuda
export CUTE_HOME=/path/to/cutlass

# Only the four Choreo kernels: no PyTorch or Choreo compiler required.
bash run.sh --gpu 0 --backends v1,v2,v4,v3

# Full comparison, using explicitly selected Python environments.
bash run.sh --gpu 0 \
  --python /path/to/main-environment/bin/python \
  --ws-python /path/to/triton-ws-environment/bin/python
```

The default is 50 warmups, 200 timed launches, and three independent timing
rounds for each shape. Results are placed in a new `results/<UTC timestamp>/`
directory. `results.json` retains all samples, validation results, package
versions, source hashes, GPU clocks/temperature observations, and run order.
`summary.csv` and `summary.md` report median latency and corresponding TFLOPS.
All raw per-backend output is retained alongside these files.
The published run is included as
[results.json](results/20260921-h800-bf16/results.json) and
[summary.csv](results/20260921-h800-bf16/summary.csv), with its raw logs in the
same directory. Running the package creates a new directory and does not
overwrite the published data.

```bash
# Compile only, without accessing a GPU.
bash run.sh --build-only --backends v1,v2,v4,v3

# Reuse locally compiled executables for a longer measurement.
bash run.sh --skip-build --gpu 0 --backends v3,v4 \
  --warmup 100 --repeat 500 --rounds 3
```

`--skip-build` is intended for unchanged sources. Rebuild after any source,
header, compiler, or flag change. A failed build, correctness check, or
backend invocation terminates the suite and leaves `complete: false` in the
partial result. Missing backends are never silently replaced or skipped.

## Modify the Choreo sources

The `kernels/` directory contains editable `.co` sources. The `generated/`
directory contains their matching CUDA snapshots. Recompilation of existing
CUDA does not need Choreo. To regenerate CUDA after editing a kernel, use a
compatible Choreo compiler:

```bash
bash run.sh --build-only --regenerate --backends v4 \
  --choreo /path/to/choreo
```

The original snapshot was generated from compiler checkout `4581f53edd51`
using `-t cute -arch=sm_90a --use-fast-math --stmatrix`, with runtime checks
disabled and zero-cost lowering enabled. The packaged `.co` files use relative
harness/config includes. In v2, the output TMA writes the full 128-row tile:
the older per-consumer sliced destination failed correctness on this compiler.
The corrected source and matching generated CUDA both pass all three shapes.
Runtime headers must remain compatible with the selected compiler.

The host helper retains the benchmark's seed-42 input generation and sampled
CPU reference check. Package changes add high-precision printed timings and
an optional `MHA_SEQ` filter for direct executable use. The suite clears this
filter and runs all three shapes. The persistent scheduler uses its current
default `fixed_cost=4`; the suite clears environment overrides.

## Measurement and verification contract

All backends use BF16 inputs/output, FP32 accumulation, causal masking,
scale `1/sqrt(128)`, Q/K uniform in [-2,2], and V uniform in [-1,1]. C++ and
PyTorch use different RNG implementations, so seed 42 specifies reproducible
inputs for each harness, not bit-identical tensors between harnesses.

Choreo uses its sampled CPU attention oracle with targeted tail queries.
Python baselines are checked independently against explicit FP32
matmul/softmax on selected rows across every batch and head, including row
zero, tile edges, and sequence tails, with all 128 output dimensions checked.
TF32 is disabled for this reference. Both use the benchmark tolerance
`0.05 + 0.1 * abs(reference)`. Checked-element counts and maximum absolute
errors are recorded; these are sampled checks, not exhaustive tensor checks.

The original Triton scripts used FP16. The packaged kernels instead use
BF16 descriptors, probabilities, and outputs, verified before timing.
Triton uses contiguous BHSD tensors; the one-time BSHD/BHSD conversion is
outside the timed region. Other implementations use BSHD directly. Thus the
table compares prearranged device layouts, not application end-to-end
conversion costs.

Timing uses CUDA events around repeated launches without CUDA Graph capture.
Compilation, Triton autotuning, verification, input preparation, and the
persistent scheduler's one-time host setup are excluded from warmed timings.
Python wrappers still allocate output tensors and prepare launch arguments;
host launch gaps can affect CUDA-event elapsed time. The package retains the
benchmark's timing convention rather than claiming instruction-only latency.

The suite measures FA3 at both ends, and runs the other implementations
sequentially between them. No two benchmarks intentionally share the GPU.
GPU clocks are not locked; snapshots and the two FA3 measurements expose
drift. The suite waits for temperature at or below 65 C before each backend.
Use an idle GPU, inspect the raw range as well as the median, and rerun if
outside workloads or clock/temperature drift affect the comparison.

Throughput uses approximate useful causal work:
`TFLOPS = (2 * B * H * S * S * D) / (milliseconds * 1e9)`.
It excludes scalar softmax operations and extra diagonal-tile arithmetic.

## Source provenance

- The compiler project is [open-source Croqtile](https://github.com/LancerLab/croqtile).
  The exact source snapshots used here are included in this download, with
  compiler/dependency revisions recorded in `provenance.json` and source
  SHA-256 hashes in the measurement records. A private checkout is not needed.
- TileLang and Triton kernel bodies are included in `baselines/`, with
  repository-dependent imports and old main programs removed. The Triton
  variants received the BF16 changes described above; Triton+WS additionally
  uses the current Hopper specialization API rather than the old aref API.
- Choreo/Croqtile runtime is distributed under Apache-2.0. The Triton-derived
  kernel and TileLang-derived kernel retain their MIT license notices in
  `licenses/`.
- CUTLASS, CUDA, PyTorch, and FA3 are external dependencies, not redistributed
  in this archive. Measured versions and hashes accompany each results set.
