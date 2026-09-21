#!/usr/bin/env python3
"""Render the tutorial's measured throughput chart (requires matplotlib).

Run beside the downloaded summary CSV, or pass --csv and --output-dir.
Only plots existing measurements; does not rerun or change benchmarks.
Use --language zh for Chinese labels (requires a Noto Sans CJK font). Chinese
SVGs embed glyph outlines so viewers do not need this font installed.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BACKENDS = [
    ("fa3", "FlashAttention-3"),
    ("v1", "Croqtile v1 (sequential)"),
    ("v2", "Croqtile v2 (1p2c TMA)"),
    ("v4", "Croqtile v4 (non-persistent)"),
    ("v3", "Croqtile v3 (persistent)"),
    ("tilelang", "TileLang"),
    ("triton", "Triton 3.6 (no WS)"),
    ("triton_ws", "Triton main (Hopper WS)"),
]
THEMES = {
    "light": {
        "bg": "#ffffff", "text": "#172b3a", "muted": "#526373",
        "grid": "#e2e8ed", "band": "#f1f8f5",
        "colors": ["#586575", "#99c4b6", "#66ab92", "#258971", "#136952",
                   "#d69942", "#8883b3", "#6393b2"],
    },
    "dark": {
        "bg": "#1e2129", "text": "#edf2f7", "muted": "#b5c0cc",
        "grid": "#39414d", "band": "#233830",
        "colors": ["#a5b0bd", "#71a996", "#63b294", "#46c4a0", "#85dfb7",
                   "#eab667", "#aaa3da", "#80b8dc"],
    },
}


LABELS_ZH = {
    "fa3": "FlashAttention-3",
    "v1": "Croqtile v1\uff08\u987a\u5e8f\u6267\u884c\uff09",
    "v2": "Croqtile v2\uff081p2c TMA\uff09",
    "v4": "Croqtile v4\uff08\u975e\u6301\u4e45\u5316\uff09",
    "v3": "Croqtile v3\uff08\u6301\u4e45\u5316\uff09",
    "tilelang": "TileLang",
    "triton": "Triton 3.6\uff08\u65e0 WS\uff09",
    "triton_ws": "Triton main\uff08Hopper WS\uff09"
}
TEXT = {
    "en": {
        "title": "Flash Attention: causal prefill, D=128",
        "subtitle": "NVIDIA H800 PCIe  |  BF16  |  B=4, H=32  |  Higher is better",
        "sequence": "S = ",
        "range": "Bars: throughput from median latency. Whiskers: full observed range, not confidence intervals.",
        "protocol": "2026-09-21  |  50 warmups + 200 launches per round  |  3 rounds; FA3: 6 bracketed rounds"
    },
    "zh": {
        "title": "Flash Attention\uff1a\u56e0\u679c\u9884\u586b\u5145\uff0cD=128",
        "subtitle": "NVIDIA H800 PCIe  |  BF16  |  B=4, H=32  |  \u541e\u5410\u91cf\u8d8a\u9ad8\u8d8a\u597d",
        "sequence": "\u5e8f\u5217\u957f\u5ea6 S = ",
        "range": "\u67f1\u957f\uff1a\u7531\u5ef6\u8fdf\u4e2d\u4f4d\u6570\u6362\u7b97\u7684\u541e\u5410\u91cf\u3002\u987b\u7ebf\uff1a\u5b8c\u6574\u5b9e\u6d4b\u8303\u56f4\uff0c\u5e76\u975e\u7f6e\u4fe1\u533a\u95f4\u3002",
        "protocol": "2026-09-21  |  \u6bcf\u8f6e\u9884\u70ed 50 \u6b21\u3001\u8ba1\u65f6 200 \u6b21  |  \u5171 3 \u8f6e\uff1bFA3 \u9996\u5c3e\u5404 3 \u8f6e"
    }
}


def chinese_font():
    # Matplotlib may expose only the first face of a CJK font collection.
    available = {font.name for font in font_manager.fontManager.ttflist}
    for family in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC"):
        if family in available:
            return family
    raise RuntimeError("Install Noto Sans CJK before rendering Chinese charts")


def draw(rows, theme, output, language="en"):
    strings = TEXT[language]
    plt.rcParams.update({
        "font.family": chinese_font() if language == "zh" else "DejaVu Sans",
        "svg.fonttype": "path" if language == "zh" else "none",
        "svg.hashsalt": "flash-attention-performance",
        "font.size": 10.5,
    })
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 6.5), sharey=True)
    fig.patch.set_facecolor(theme["bg"])
    fig.subplots_adjust(left=0.205, right=0.978, top=0.77, bottom=0.19,
                        wspace=0.13)
    fig.text(0.035, 0.955, strings["title"],
             fontsize=19, weight="bold", color=theme["text"], va="top")
    fig.text(0.035, 0.895,
             strings["subtitle"],
             fontsize=11, color=theme["muted"], va="top")

    for ax, seq in zip(axes, (4096, 8192, 16384)):
        ax.set_facecolor(theme["bg"])
        ax.axhspan(0.5, 4.5, color=theme["band"], zorder=0)
        for y, (backend, _) in enumerate(BACKENDS):
            row = rows[(backend, seq)]
            ms = float(row["median_ms"])
            tflops = float(row["tflops"])
            work = 2 * 4 * 32 * seq * seq * 128 / 1e9
            assert abs(tflops - work / ms) < 1e-8
            low = work / float(row["max_ms"])
            high = work / float(row["min_ms"])
            ax.barh(y, tflops, height=0.58,
                    color=theme["colors"][y], zorder=2)
            ax.errorbar(tflops, y, xerr=[[tflops - low], [high - tflops]],
                        fmt="none", ecolor=theme["text"], elinewidth=0.9,
                        capsize=3, capthick=0.9, zorder=3)
            ax.text(high + 9, y, f"{tflops:.1f}", va="center",
                    fontsize=10, color=theme["text"])
        ax.set_title(f"{strings['sequence']}{seq:,}", fontsize=13, weight="bold",
                     color=theme["text"], pad=14)
        ax.set_xlim(0, 500)
        ax.set_xticks(range(0, 501, 100))
        ax.set_xlabel("TFLOPS", color=theme["muted"], labelpad=9)
        ax.set_yticks(range(len(BACKENDS)))
        ax.set_yticklabels([LABELS_ZH[backend] if language == "zh" else label
                            for backend, label in BACKENDS],
                           color=theme["text"])
        ax.set_ylim(len(BACKENDS) - 0.45, -0.65)
        ax.tick_params(axis="both", length=0, pad=8,
                       colors=theme["muted"], labelsize=10)
        ax.grid(axis="x", color=theme["grid"], linewidth=0.8, zorder=1)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.text(0.035, 0.085,
             strings["range"],
             fontsize=10, color=theme["muted"])
    fig.text(0.035, 0.048,
             strings["protocol"],
             fontsize=10, color=theme["muted"])
    fig.savefig(output, facecolor=theme["bg"],
                metadata={"Date": None, "Title": "Measured causal attention throughput"})
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=root /
        "standalone/results/20260921-h800-bf16/summary.csv")
    parser.add_argument("--output-dir", type=Path, default=root)
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    args = parser.parse_args()
    with args.csv.open(newline="") as stream:
        data = list(csv.DictReader(stream))
    rows = {(row["backend"], int(row["sequence"])): row for row in data}
    assert len(rows) == len(data) == 24, "Expected eight backends and three shapes"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        prefix = "performance-zh" if args.language == "zh" else "performance"
        output = args.output_dir / f"{prefix}-{name}.svg"
        draw(rows, theme, output, args.language)
        print(output)


if __name__ == "__main__":
    main()
