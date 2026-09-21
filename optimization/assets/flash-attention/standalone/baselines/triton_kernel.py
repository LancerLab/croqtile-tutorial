#!/usr/bin/env python3
"""Triton fused attention baseline for causal_prefill_d128.

Uses the official Triton tutorial flash attention (06-fused-attention.py)
which supports TMA descriptors and autotuned tile configurations on Hopper.
Layout: BHSD internally (tensors are transposed from BSHD).
"""

import torch

# ---------------------------------------------------------------------------
# Import the Triton tutorial fused attention kernel
# ---------------------------------------------------------------------------
import os
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    desc_k, desc_v,
    offset_y, dtype: tl.constexpr, start_m, qk_scale,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
    N_CTX: tl.constexpr, warp_specialize: tl.constexpr,
    IS_HOPPER: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        if (not IS_HOPPER and warp_specialize
                and BLOCK_M == 128 and HEAD_DIM == 128):
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = (acc.reshape([BM, 2, BN // 2])
                          .permute(0, 2, 1).split())
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = (tl.join(acc0, acc1)
                   .permute(0, 2, 1).reshape([BM, BN]))
        else:
            acc = acc * alpha[:, None]
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        acc = tl.dot(p, v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


if is_hip():
    _NUM_STAGES_OPTIONS = [1]
elif supports_host_descriptor():
    _NUM_STAGES_OPTIONS = [2, 3, 4]
else:
    _NUM_STAGES_OPTIONS = [2, 3, 4]

_fwd_configs = [
    triton.Config(
        {"BLOCK_M": BM, "BLOCK_N": BN},
        num_stages=s, num_warps=w,
        pre_hook=_host_descriptor_pre_hook,
    )
    for BM in [64, 128]
    for BN in [32, 64, 128]
    for s in _NUM_STAGES_OPTIONS
    for w in [4, 8]
]


def _keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (
        is_cuda()
        and torch.cuda.get_device_capability()[0] == 9
        and BLOCK_M * BLOCK_N < 128 * 128
        and conf.num_warps == 8
    )


def _prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]
    STAGE = kwargs["STAGE"]
    return [
        conf for conf in configs
        if conf.kwargs.get("BLOCK_M", 0) <= N_CTX
        and (conf.kwargs.get("BLOCK_M", 0) >= conf.kwargs.get("BLOCK_N", 0)
             or STAGE == 1)
    ]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(
            desc_or_ptr, shape, strides, block_shape
        )


@triton.autotune(
    configs=list(filter(_keep, _fwd_configs)),
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
    prune_configs_by={"early_config_prune": _prune_invalid_configs},
)
@triton.jit
def _attn_fwd(
    sm_scale, M,
    Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
    IS_HOPPER: tl.constexpr,
):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.bfloat16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(
            desc_v, shape=[HEAD_DIM, y_dim], strides=[N_CTX, 1],
            block_shape=[HEAD_DIM, BLOCK_N],
        )
    else:
        desc_v = _maybe_make_tensor_desc(
            desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_N, HEAD_DIM],
        )
    desc_k = _maybe_make_tensor_desc(
        desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    q = desc_q.load([qo_offset_y, 0])
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q,
            desc_k, desc_v,
            offset_y, dtype, start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            4 - STAGE, offs_m, offs_n, N_CTX,
            warp_specialize, IS_HOPPER,
        )
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q,
            desc_k, desc_v,
            offset_y, dtype, start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            2, offs_m, offs_n, N_CTX,
            warp_specialize, IS_HOPPER,
        )
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


def triton_attention_fwd(q, k, v, causal, sm_scale):
    """Run Triton fused attention forward.

    q, k, v: [B, H, S, D] contiguous bf16.
    Returns output [B, H, S, D].
    """
    HEAD_DIM_K = q.shape[-1]
    o = torch.empty_like(q)
    stage = 3 if causal else 1
    extra_kern_args = {}

    M = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2]),
        device=q.device, dtype=torch.float32,
    )

    warp_specialize = False
    if is_hopper() and not causal:
        warp_specialize = True

    if supports_host_descriptor() and not (is_hopper() and warp_specialize):
        y_dim = q.shape[0] * q.shape[1] * q.shape[2]
        dummy_block = [1, 1]
        desc_q = TensorDescriptor(
            q, shape=[y_dim, HEAD_DIM_K],
            strides=[HEAD_DIM_K, 1], block_shape=dummy_block,
        )
        desc_v = TensorDescriptor(
            v, shape=[y_dim, HEAD_DIM_K],
            strides=[HEAD_DIM_K, 1], block_shape=dummy_block,
        )
        desc_k = TensorDescriptor(
            k, shape=[y_dim, HEAD_DIM_K],
            strides=[HEAD_DIM_K, 1], block_shape=dummy_block,
        )
        desc_o = TensorDescriptor(
            o, shape=[y_dim, HEAD_DIM_K],
            strides=[HEAD_DIM_K, 1], block_shape=dummy_block,
        )
    else:
        desc_q, desc_v, desc_k, desc_o = q, v, k, o

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    def grid(META):
        return (
            triton.cdiv(q.shape[2], META["BLOCK_M"]),
            q.shape[0] * q.shape[1],
            1,
        )

    _attn_fwd[grid](
        sm_scale, M,
        q.shape[0], q.shape[1],
        desc_q, desc_k, desc_v, desc_o,
        N_CTX=q.shape[2],
        HEAD_DIM=HEAD_DIM_K,
        FP8_OUTPUT=False,
        STAGE=stage,
        warp_specialize=warp_specialize,
        IS_HOPPER=is_hopper(),
        **extra_kern_args,
    )
    return o
