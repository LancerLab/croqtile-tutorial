#!/usr/bin/env python3
"""Build and measure the standalone causal attention tutorial."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
KERNELS = {
    "v1": "v1_manual_baseline",
    "v2": "v2_manual_s2_1p2c_tma",
    "v3": "v3_fa3_overlap",
    "v4": "v4_nonpersistent_fa",
}
BACKENDS = ["fa3", "v1", "v2", "v4", "v3", "tilelang", "triton", "triton_ws"]
SEQUENCES = [4096, 8192, 16384]


def run_logged(command, path, env=None):
    print("Running:", " ".join(map(str, command)), flush=True)
    with path.open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=1200)
    text = path.read_text()
    if result.returncode:
        print(text[-6000:], file=sys.stderr)
        raise RuntimeError(f"Exit {result.returncode}; see {path}")
    return text


def gpu_state(gpu):
    fields = ("index,name,uuid,driver_version,compute_cap,temperature.gpu,"
              "clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used")
    return subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}", f"--query-gpu={fields}",
         "--format=csv,noheader,nounits"], text=True).strip()


def cooldown(gpu):
    # Keep thermal conditions bounded without changing clocks or power limits.
    deadline = time.monotonic() + 120
    while True:
        temperature = int(subprocess.check_output(
            ["nvidia-smi", f"--id={gpu}", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"], text=True).strip())
        if temperature <= 65:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("GPU did not cool below 65 C; retry on an idle GPU")
        time.sleep(3)


def build(args, backends):
    cutlass = Path(args.cutlass).expanduser() if args.cutlass else None
    if not cutlass or not (cutlass / "include/cutlass/cutlass.h").is_file():
        raise RuntimeError("Set CUTE_HOME or pass --cutlass /path/to/cutlass")
    cuda = Path(args.cuda)
    for name in backends:
        if name not in KERNELS:
            continue
        stem = KERNELS[name]
        source = Path("generated") / f"{stem}.cu"
        if args.regenerate or not (ROOT / source).is_file():
            run_logged([args.choreo, "-es", "-t", "cute", "-arch=sm_90a",
                        "--use-fast-math", "--stmatrix",
                        "--disable-runtime-check=true", "--zero-cost=true",
                        f"kernels/{stem}.co", "-o", str(source)],
                       ROOT / "build" / f"{name}-codegen.log")
        command = [str(cuda / "bin/nvcc"), str(source), "-o", f"build/{name}",
                   "-std=c++17", "-O3", "-gencode", "arch=compute_90a,code=sm_90a",
                   "--use_fast_math",
                   "--expt-relaxed-constexpr", "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
                   "-D__CHOREO_TARGET_CUTE__", "-D__USE_CUDA_TYPE__",
                   "-Iruntime", "-I.", f"-I{cutlass / 'include'}",
                   "-Xcompiler", "-static-libstdc++", "-lcuda",
                   "-Xptxas=-v"]
        run_logged(command, ROOT / "build" / f"{name}-nvcc.log")


def parse_choreo(text, require_verify):
    values = []
    for match in re.finditer(
            r"^B=4 H=32 SEQ=(\d+) causal prefill\n(.*?)(?=^B=4 H=32|\Z)",
            text, re.MULTILINE | re.DOTALL):
        seq, body = int(match[1]), match[2]
        verification = re.search(
            r"max_abs_err=([\deE.+-]+).*?fail_rate=0 \(0/(\d+)\) PASS", body)
        if require_verify and not verification:
            raise RuntimeError(f"Missing successful verification for S={seq}")
        timing = re.search(r"Timing avg ms: ([\deE.+-]+)", body)
        values.append({"sequence": seq,
                       "ms": float(timing[1]) if timing else None,
                       "verification": {"passed": True,
                           "max_abs_err": float(verification[1]),
                           "checked": int(verification[2])} if verification else None})
    if sorted(x["sequence"] for x in values) != SEQUENCES:
        raise RuntimeError("Expected exactly the three configured shapes")
    return values


def source_hashes():
    paths = [p
            for directory in ("kernels", "generated", "runtime", "baselines")
            for p in sorted((ROOT / directory).glob("*")) if p.is_file()]
    paths += [ROOT / p for p in ("run.py", "baseline.py", "fa_helper.hpp", "bench_configs.inc")]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def save_results(result, directory):
    (directory / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    rows = []
    groups = {}
    for item in result["measurements"]:
        groups.setdefault((item["backend"], item["sequence"]), []).append(item["ms"])
    for (backend, seq), times in groups.items():
        median = statistics.median(times)
        rows.append({"backend": backend, "sequence": seq, "samples": len(times),
                     "median_ms": median, "min_ms": min(times), "max_ms": max(times),
                     "tflops": 2 * 4 * 32 * seq * seq * 128 / (median * 1e9)})
    if rows:
        with (directory / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        header = "| Implementation | S=4096 ms / TFLOPS | S=8192 ms / TFLOPS | S=16384 ms / TFLOPS |"
        lines = [header, "| --- | ---: | ---: | ---: |"]
        for backend in dict.fromkeys(row["backend"] for row in rows):
            cells = []
            for seq in SEQUENCES:
                item = next((r for r in rows if r["backend"] == backend and r["sequence"] == seq), None)
                cells.append(f"{item['median_ms']:.4f} / {item['tflops']:.1f}" if item else "pending")
            lines.append("| " + " | ".join([backend] + cells) + " |")
        (directory / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ws-python", default=os.environ.get("TRITON_WS_PYTHON", sys.executable))
    parser.add_argument("--cutlass", default=os.environ.get("CUTE_HOME", ""))
    parser.add_argument("--cuda", default=os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    parser.add_argument("--choreo", default=os.environ.get("CHOREO_BIN", "choreo"))
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    backends = args.backends.split(",")
    if any(b not in BACKENDS for b in backends):
        parser.error("Unknown backend")
    if args.warmup < 1 or args.repeat < 1 or args.rounds < 1:
        parser.error("Warmup, repeat, and rounds must be positive")
    (ROOT / "build").mkdir(exist_ok=True)
    (ROOT / "generated").mkdir(exist_ok=True)
    if not args.skip_build:
        build(args, backends)
    if args.build_only:
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(args.output).resolve() if args.output else ROOT / "results" / stamp
    directory.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu,
               CHOREO_TIMING_WARMUP=str(args.warmup),
               CHOREO_TIMING_REPEAT=str(args.repeat),
               CUDA_CACHE_PATH=str(ROOT / "build/cuda-cache"),
               TRITON_CACHE_DIR=str(ROOT / "build/triton-cache"),
               TILELANG_CACHE_DIR=str(ROOT / "build/tilelang-cache"),
               PYTHONUNBUFFERED="1")
    for key in ("CHOREO_SKIP_VERIFY", "CHOREO_DISABLE_TIMING", "MHA_SEQ",
                "MHA_VERIFY_SAMPLES", "CHOREO_SCHED_FIXED_COST"):
        env.pop(key, None)
    result = {"timestamp_utc": stamp, "warmup": args.warmup, "repeat": args.repeat,
              "rounds": args.rounds, "dtype": "bf16", "batch": 4, "heads": 32,
              "dimension": 128, "sequences": SEQUENCES, "source_sha256": source_hashes(),
              "gpu_before": gpu_state(args.gpu), "measurements": [],
              "verification": [], "runs": [], "complete": False}
    try:
        # FA3 at both ends exposes thermal/clock drift across the suite.
        order = backends + (["fa3"] if "fa3" in backends and len(backends) > 1 else [])
        for index, backend in enumerate(order):
            cooldown(args.gpu)
            before = gpu_state(args.gpu)
            prefix = f"{index:02d}-{backend}"
            if backend in KERNELS:
                verify_env = {**env, "CHOREO_DISABLE_TIMING": "1"}
                text = run_logged([str(ROOT / "build" / backend)],
                                  directory / f"{prefix}-verify.log", verify_env)
                for item in parse_choreo(text, True):
                    result["verification"].append({"backend": backend, **item})
                for round_id in range(args.rounds):
                    text = run_logged([str(ROOT / "build" / backend)],
                                      directory / f"{prefix}-round{round_id}.log",
                                      {**env, "CHOREO_SKIP_VERIFY": "1"})
                    for item in parse_choreo(text, False):
                        if item["ms"] is None or item["ms"] <= 0:
                            raise RuntimeError("Invalid timing")
                        result["measurements"].append({"backend": backend,
                            "sequence": item["sequence"], "round": round_id,
                            "suite_position": index, "ms": item["ms"]})
            else:
                python = args.ws_python if backend == "triton_ws" else args.python
                text = run_logged([python, "baseline.py", "--backend", backend,
                    "--rounds", str(args.rounds), "--warmup", str(args.warmup),
                    "--repeat", str(args.repeat)], directory / f"{prefix}.log", env)
                marker = next(line for line in text.splitlines() if line.startswith("RESULT_JSON="))
                data = json.loads(marker.removeprefix("RESULT_JSON="))
                if len(data["verification"]) != 3 or not all(v["passed"] for v in data["verification"]):
                    raise RuntimeError("Baseline verification incomplete")
                if len(data["measurements"]) != 3 * args.rounds:
                    raise RuntimeError("Baseline measurements incomplete")
                result["verification"].extend({"backend": backend, **v} for v in data["verification"])
                result["measurements"].extend({"backend": backend, "suite_position": index, **v}
                                              for v in data["measurements"])
                result.setdefault("versions", {})[backend] = data["versions"]
            result["runs"].append({"backend": backend, "position": index,
                                   "gpu_before": before, "gpu_after": gpu_state(args.gpu)})
            save_results(result, directory)
        result["complete"] = True
    finally:
        result["gpu_after"] = gpu_state(args.gpu)
        save_results(result, directory)
    print((directory / "summary.md").read_text())
    print("Results:", directory)


if __name__ == "__main__":
    main()
