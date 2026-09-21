#!/usr/bin/env python3
"""BF16 attention baselines with independent FP32 reference-row checks."""

import argparse
import importlib.metadata
import json
import torch


def verify(q, k, v, output):
    batch, seq, heads, dim = q.shape
    max_error = 0.0
    failures = 0
    checked = 0
    torch.backends.cuda.matmul.allow_tf32 = False
    positions = torch.arange(seq, device=q.device)
    for b in range(batch):
        for h in range(heads):
            # Include the causal origin, tile edges, mid-sequence, and tail.
            rows = sorted(set([0, 1, 63, 64, 127, 128, seq // 2, seq - 2, seq - 1]
                + [((b * heads + h) * 997 + j * 7919) % seq for j in range(7)]))
            indices = torch.tensor(rows, device=q.device)
            scores = q[b, indices, h].float() @ k[b, :, h].float().T
            scores *= dim ** -0.5
            scores.masked_fill_(positions[None, :] > indices[:, None], -float("inf"))
            reference = torch.softmax(scores, dim=-1) @ v[b, :, h].float()
            actual = output[b, indices, h].float()
            error = (actual - reference).abs()
            bad = (~torch.isfinite(actual)) | (error > 0.05 + 0.1 * reference.abs())
            failures += int(bad.sum().item())
            max_error = max(max_error, float(error.max().item()))
            checked += actual.numel()
    if failures:
        raise RuntimeError(f"Reference check failed: {failures}/{checked}, max error {max_error}")
    return {"sequence": seq, "passed": True, "checked": checked,
            "max_abs_err": max_error, "reference": "FP32 selected rows, all output dimensions"}


def event_time(launch, warmup, repeat):
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(repeat):
        launch()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["fa3", "tilelang", "triton", "triton_ws"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("This tutorial is specialized for Hopper SM90")
    versions = {"torch": torch.__version__, "cuda": torch.version.cuda}
    for name in ("triton", "tilelang", "flash_attn_3"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    if args.backend == "fa3":
        from flash_attn_interface import flash_attn_func
    elif args.backend == "tilelang":
        from baselines.tilelang_kernel import flashattn_kernel
    elif args.backend == "triton":
        from baselines.triton_kernel import triton_attention_fwd
    else:
        from baselines.triton_ws_kernel import triton_ws_attention_fwd

    result = {"versions": versions, "verification": [], "measurements": []}
    for seq in (4096, 8192, 16384):
        generator = torch.Generator(device="cuda").manual_seed(42)
        q, k, v = [torch.empty((4, seq, 32, 128), device="cuda", dtype=torch.bfloat16)
                   for _ in range(3)]
        q.uniform_(-2, 2, generator=generator)
        k.uniform_(-2, 2, generator=generator)
        v.uniform_(-1, 1, generator=generator)
        if args.backend == "fa3":
            def launch():
                value = flash_attn_func(q, k, v, softmax_scale=128 ** -0.5,
                                        causal=True, num_splits=0)
                return value[0] if isinstance(value, tuple) else value
        elif args.backend == "tilelang":
            kernel = flashattn_kernel(4, 32, seq, 128, True,
                                     block_M=128, block_N=128, num_stages=2, threads=256)
            def launch():
                return kernel(q, k, v)
        else:
            qb, kb, vb = [x.permute(0, 2, 1, 3).contiguous() for x in (q, k, v)]
            attention = triton_attention_fwd if args.backend == "triton" else triton_ws_attention_fwd
            def launch():
                return attention(qb, kb, vb, True, 128 ** -0.5).permute(0, 2, 1, 3)
        print(f"Preparing {args.backend}, S={seq}", flush=True)
        output = launch()  # Includes compilation/autotuning, outside measurement.
        torch.cuda.synchronize()
        check = verify(q, k, v, output)
        result["verification"].append(check)
        print("VERIFY", json.dumps(check), flush=True)
        for round_id in range(args.rounds):
            ms = event_time(launch, args.warmup, args.repeat)
            result["measurements"].append({"sequence": seq, "round": round_id, "ms": ms})
            print(f"S={seq} round={round_id} ms={ms:.9f}", flush=True)
        del output, q, k, v
        if args.backend in ("triton", "triton_ws"):
            del qb, kb, vb
        torch.cuda.empty_cache()
    print("RESULT_JSON=" + json.dumps(result))


if __name__ == "__main__":
    main()
