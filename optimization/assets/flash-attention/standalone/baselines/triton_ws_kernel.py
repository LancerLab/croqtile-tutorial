#!/usr/bin/env python3
"""Triton fused attention with current Hopper warp specialization.

Uses tl.range(warp_specialize=True), device-side tensor descriptors, and
Triton's Hopper specialization pass. The measured compiler revision is
recorded in provenance.json. This is not the old aref_auto_ws implementation.

Layout: BHSD internally (tensors are transposed from BSHD).
Select the matching Python environment with --ws-python.
"""

import torch

import triton
import triton.language as tl


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


@triton.jit
def _attn_fwd_inner_ws(
    acc, l_i, m_i, q,
    desc_k, desc_v,
    offs_hz, start_m, qk_scale,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
    N_CTX: tl.constexpr, fp8_v: tl.constexpr,
):
    if STAGE == 1:
        lo = 0
        hi = start_m * BLOCK_M
    elif STAGE == 2:
        lo = start_m * BLOCK_M
        hi = (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    elif STAGE == 4:
        # Keep one causal loop: this revision's Hopper WS pass fails on the
        # old separate off-diagonal/diagonal loops. Every tile applies a mask.
        lo = 0
        hi = (start_m + 1) * BLOCK_M
    else:
        lo = 0
        hi = N_CTX
    lo = lo.to(tl.int32)
    hi = hi.to(tl.int32)
    offs_kv = offs_hz * N_CTX + lo

    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=True):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = desc_k.load([offs_kv, 0])
        qk = tl.dot(q, k.T)
        if STAGE == 2 or STAGE == 4:
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
        acc = acc * alpha[:, None]
        if fp8_v:
            v = desc_v.load([offs_hz * HEAD_DIM, start_n])
            p = p.to(tl.float8e5)
            acc = tl.dot(p, v.T, acc)
        else:
            v = desc_v.load([offs_kv, 0])
            p = p.to(q.dtype)
            acc = tl.dot(p, v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offs_kv += BLOCK_N
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_ws(
    Q, K, V,
    sm_scale, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    q_desc_ptr = tl.make_tensor_descriptor(
        Q, [Z * H * N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_M, HEAD_DIM])
    k_desc_ptr = tl.make_tensor_descriptor(
        K, [Z * H * N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_N, HEAD_DIM])
    v_desc_ptr = tl.make_tensor_descriptor(
        V, [Z * H * N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_N, HEAD_DIM])

    o_desc_ptr = tl.make_tensor_descriptor(
        Out, [Z * H * N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_M, HEAD_DIM])

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    q = q_desc_ptr.load([off_hz * N_CTX + start_m * BLOCK_M, 0])

    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner_ws(
            acc, l_i, m_i, q,
            k_desc_ptr, v_desc_ptr,
            off_hz, start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            4 if STAGE == 3 else 3, offs_m, offs_n,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,
        )
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    o_desc_ptr.store([off_hz * N_CTX + start_m * BLOCK_M, 0],
                     acc.to(Out.type.element_ty))


def triton_ws_attention_fwd(q, k, v, causal, sm_scale):
    """Triton fused attention with Hopper warp specialization.

    q, k, v: [B, H, S, D] contiguous bf16.
    Returns output [B, H, S, D].
    """
    HEAD_DIM_K = q.shape[-1]
    o = torch.empty_like(q)
    stage = 3 if causal else 1
    Z, H, N_CTX = q.shape[:3]
    BLOCK_M, BLOCK_N = 128, 128
    # Hopper's current specialization pass requires a four-warp input and
    # partitions it into producer/consumer warp groups during compilation.
    NUM_WARPS = 4
    NUM_STAGES = 2

    M = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2]),
        device=q.device, dtype=torch.float32,
    )

    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device=q.device, dtype=torch.uint8)
    triton.set_allocator(alloc_fn)

    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H, 1)

    compiled = _attn_fwd_ws[grid](
        q, k, v,
        sm_scale, M, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z, H,
        N_CTX=N_CTX,
        HEAD_DIM=HEAD_DIM_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        STAGE=stage,
        num_stages=NUM_STAGES,
        num_warps=NUM_WARPS,
    )
    if compiled.metadata.num_warps != 12:
        raise RuntimeError("Expected Hopper WS lowering to three warp groups")
    return o
