#!/usr/bin/env python3
"""Generate tutorial diagram PNGs for the Dense GEMM FP16 tutorial."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "img"
OUT.mkdir(exist_ok=True)

DARK_BG = "#1e1e2e"
ACCENT_GREEN = "#40c057"
ACCENT_RED = "#ff6b6b"
ACCENT_BLUE = "#74c0fc"
ACCENT_YELLOW = "#ffd43b"
ACCENT_PURPLE = "#b197fc"
ACCENT_ORANGE = "#ffa94d"
GRAY = "#adb5bd"
WHITE = "#e4e4e7"

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor': DARK_BG,
    'text.color': WHITE,
    'axes.labelcolor': WHITE,
    'xtick.color': GRAY,
    'ytick.color': GRAY,
    'font.family': 'DejaVu Sans',
    'font.size': 11,
})


def fig1_v0_memory_access():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis('off')
    ax.set_title("v0: Memory Access Pattern — 4 Threads", fontsize=14, fontweight='bold', pad=15, color=WHITE)

    # Matrix A
    a_x, a_y, a_w, a_h = 0.3, 1.0, 3.5, 3.5
    ax.add_patch(mpatches.FancyBboxPatch((a_x, a_y), a_w, a_h, boxstyle="round,pad=0.1",
                                          facecolor="#2d2d44", edgecolor=ACCENT_BLUE, linewidth=1.5))
    ax.text(a_x + a_w/2, a_y + a_h + 0.25, "A  [M × K]", ha='center', fontsize=12, fontweight='bold', color=ACCENT_BLUE)
    for i, (label, color) in enumerate([("row 0", ACCENT_GREEN), ("row 1", ACCENT_ORANGE),
                                         ("row 0", ACCENT_GREEN), ("row 1", ACCENT_ORANGE)]):
        y = a_y + a_h - 0.5 - i * 0.8
        ax.annotate('', xy=(a_x + a_w - 0.1, y), xytext=(a_x + 0.2, y),
                     arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax.text(a_x + 0.25, y + 0.15, label, fontsize=9, color=color)

    # Matrix B
    b_x, b_y, b_w, b_h = 5.8, 1.0, 3.5, 3.5
    ax.add_patch(mpatches.FancyBboxPatch((b_x, b_y), b_w, b_h, boxstyle="round,pad=0.1",
                                          facecolor="#2d2d44", edgecolor=ACCENT_PURPLE, linewidth=1.5))
    ax.text(b_x + b_w/2, b_y + b_h + 0.25, "B  [N × K]  (row = col of B^T)", ha='center', fontsize=12, fontweight='bold', color=ACCENT_PURPLE)
    labels_b = [("col 0", ACCENT_GREEN), ("col 0", ACCENT_GREEN),
                ("col 1", ACCENT_ORANGE), ("col 1", ACCENT_ORANGE)]
    threads = ["thread(0,0)", "thread(1,0)", "thread(0,1)", "thread(1,1)"]
    for i, ((label, color), thr) in enumerate(zip(labels_b, threads)):
        y = b_y + b_h - 0.5 - i * 0.8
        ax.annotate('', xy=(b_x + b_w - 0.1, y), xytext=(b_x + 0.2, y),
                     arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax.text(b_x + 0.25, y + 0.15, label, fontsize=9, color=color)
        ax.text(b_x + b_w + 0.15, y, thr, fontsize=8, color=GRAY, va='center')

    # Arrows from A to B
    for i in range(4):
        y = a_y + a_h - 0.5 - i * 0.8
        ax.annotate('', xy=(b_x, y), xytext=(a_x + a_w + 0.05, y),
                     arrowprops=dict(arrowstyle='->', color=ACCENT_RED, lw=1.5, linestyle='dotted'))

    # Bottom annotation
    ax.text(5, 0.55, "Each thread reads:  row_m of A + col_n of B = 2 × 8192 × 2B = 32 KB from HBM",
            ha='center', fontsize=10, color=ACCENT_YELLOW)
    ax.text(5, 0.15, "Same rows/cols re-read redundantly. Total traffic ≈ 2 TB  (minimum ≈ 400 MB)",
            ha='center', fontsize=9, color=GRAY)

    fig.tight_layout()
    fig.savefig(OUT / "v0_memory_access.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig2_v1_tile_reuse():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title("v1: Tile Reuse — Sliding Window Along K", fontsize=14, fontweight='bold', pad=15, color=WHITE)

    stages = ["iv_k=0", "iv_k=1", "...", "iv_k=63"]
    colors = [ACCENT_BLUE, ACCENT_PURPLE, GRAY, ACCENT_GREEN]
    for i, (label, color) in enumerate(zip(stages, colors)):
        x = 0.5 + i * 2.4
        if label == "...":
            ax.text(x + 0.7, 3.0, "⋯", fontsize=24, ha='center', va='center', color=GRAY)
            continue
        # HBM box
        ax.add_patch(mpatches.FancyBboxPatch((x, 3.5), 1.5, 1.0, boxstyle="round,pad=0.05",
                                              facecolor="#2d2d44", edgecolor=color, linewidth=1.5))
        ax.text(x + 0.75, 4.05, "HBM tile", ha='center', fontsize=8, color=color)
        ax.text(x + 0.75, 3.7, label, ha='center', fontsize=9, fontweight='bold', color=WHITE)
        # Arrow down
        ax.annotate('', xy=(x + 0.75, 2.8), xytext=(x + 0.75, 3.45),
                     arrowprops=dict(arrowstyle='->', color=ACCENT_YELLOW, lw=2))
        ax.text(x + 0.95, 3.1, "dma", fontsize=8, color=ACCENT_YELLOW, style='italic')
        # SRAM box
        ax.add_patch(mpatches.FancyBboxPatch((x, 1.8), 1.5, 1.0, boxstyle="round,pad=0.05",
                                              facecolor="#1a3a1a", edgecolor=ACCENT_GREEN, linewidth=1.5))
        ax.text(x + 0.75, 2.35, "SRAM tile", ha='center', fontsize=8, color=ACCENT_GREEN)
        ax.text(x + 0.75, 2.0, "[32×128]", ha='center', fontsize=9, fontweight='bold', color=WHITE)
        # Arrow down to compute
        ax.annotate('', xy=(x + 0.75, 1.1), xytext=(x + 0.75, 1.75),
                     arrowprops=dict(arrowstyle='->', color=ACCENT_ORANGE, lw=2))

    # Compute row
    ax.add_patch(mpatches.FancyBboxPatch((0.5, 0.3), 9.0, 0.7, boxstyle="round,pad=0.05",
                                          facecolor="#3d2d1a", edgecolor=ACCENT_ORANGE, linewidth=1.5))
    ax.text(5.0, 0.65, "acc += lhs_s × rhs_s    (1024 threads, scalar FMA)", ha='center', fontsize=11, color=ACCENT_ORANGE)

    # Annotation
    ax.text(5.0, 1.3, "Each 8 KB tile loaded once, read 32× by threads in the block  →  reuse factor = 32",
            ha='center', fontsize=9, color=GRAY)

    fig.tight_layout()
    fig.savefig(OUT / "v1_tile_reuse.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig3_double_buffering():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title("Double-Buffering (STAGES=2): TMA Latency Fully Hidden", fontsize=14, fontweight='bold', pad=12, color=WHITE)

    # Time axis
    ax.annotate('', xy=(10.5, 4.3), xytext=(0.5, 4.3),
                 arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))
    ax.text(10.6, 4.3, "time", fontsize=10, color=GRAY, va='center')

    # Producer row
    ax.text(0.15, 3.5, "Producer\n(wg=0)", fontsize=9, color=ACCENT_BLUE, fontweight='bold', va='center')
    prod_tiles = [(1.5, "tile[0]\nslot 0", ACCENT_BLUE),
                  (3.5, "tile[1]\nslot 1", ACCENT_PURPLE),
                  (5.5, "tile[2]\nslot 0", ACCENT_BLUE),
                  (7.5, "tile[3]\nslot 1", ACCENT_PURPLE)]
    for x, label, color in prod_tiles:
        ax.add_patch(mpatches.FancyBboxPatch((x, 3.1), 1.6, 0.8, boxstyle="round,pad=0.05",
                                              facecolor=color+"20", edgecolor=color, linewidth=2))
        ax.text(x + 0.8, 3.5, label, ha='center', va='center', fontsize=8, color=WHITE)

    # Consumer row
    ax.text(0.15, 1.8, "Consumer\n(wg=1,2)", fontsize=9, color=ACCENT_GREEN, fontweight='bold', va='center')
    cons_tiles = [(2.5, "WGMMA\ntile[0]\nslot 0", ACCENT_GREEN),
                  (4.5, "WGMMA\ntile[1]\nslot 1", "#2d8a2d"),
                  (6.5, "WGMMA\ntile[2]\nslot 0", ACCENT_GREEN),
                  (8.5, "WGMMA\ntile[3]\nslot 1", "#2d8a2d")]
    for x, label, color in cons_tiles:
        ax.add_patch(mpatches.FancyBboxPatch((x, 1.3), 1.6, 1.0, boxstyle="round,pad=0.05",
                                              facecolor=color+"20", edgecolor=color, linewidth=2))
        ax.text(x + 0.8, 1.8, label, ha='center', va='center', fontsize=7, color=WHITE)

    # Overlap annotation
    ax.annotate('', xy=(4.0, 2.6), xytext=(2.0, 2.6),
                 arrowprops=dict(arrowstyle='<->', color=ACCENT_YELLOW, lw=2))
    ax.text(3.0, 2.75, "overlap", fontsize=9, color=ACCENT_YELLOW, ha='center', fontweight='bold')

    # full/empty arrows
    for px, cx in [(2.3, 2.5), (4.3, 4.5), (6.3, 6.5)]:
        ax.annotate('', xy=(cx + 0.3, 2.35), xytext=(px + 0.5, 3.05),
                     arrowprops=dict(arrowstyle='->', color=ACCENT_YELLOW, lw=1, linestyle='--'))
    ax.text(1.5, 2.5, "full[0]↓", fontsize=7, color=ACCENT_YELLOW)

    # Bottom: comparison with STAGES=1
    ax.add_patch(mpatches.FancyBboxPatch((0.5, 0.05), 10.0, 0.8, boxstyle="round,pad=0.05",
                                          facecolor="#2d1a1a", edgecolor=ACCENT_RED, linewidth=1))
    ax.text(5.5, 0.45, "STAGES=1 (v3):  no overlap — producer must wait for consumer to finish → TMA latency exposed",
            ha='center', fontsize=9, color=ACCENT_RED)

    fig.tight_layout()
    fig.savefig(OUT / "double_buffering_timeline.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig4_warpn_sweep():
    fig, ax = plt.subplots(figsize=(9, 5))

    warp_ns = [128, 136, 144, 152, 160, 168, 176, 192, 208, 224]
    tflops =  [365, 386, 434, 402, 457, 457, 476, 466, None, None]
    colors_list = []
    for i, (wn, tf) in enumerate(zip(warp_ns, tflops)):
        if tf is None:
            colors_list.append(ACCENT_RED)
        elif wn == 192:
            colors_list.append(ACCENT_GREEN)
        else:
            colors_list.append(ACCENT_BLUE)

    valid_ns = [wn for wn, tf in zip(warp_ns, tflops) if tf is not None]
    valid_tf = [tf for tf in tflops if tf is not None]
    valid_colors = [c for c, tf in zip(colors_list, tflops) if tf is not None]

    bars = ax.bar([str(n) for n in valid_ns], valid_tf, color=valid_colors, width=0.7, edgecolor='none')

    # Baseline line
    ax.axhline(y=337, color=ACCENT_ORANGE, linestyle='--', lw=1.5, alpha=0.7)
    ax.text(0.02, 340, "v3 baseline (337)", fontsize=8, color=ACCENT_ORANGE, transform=ax.get_yaxis_transform())

    # cuBLAS line
    ax.axhline(y=447.5, color=ACCENT_YELLOW, linestyle='--', lw=1.5, alpha=0.7)
    ax.text(0.02, 450, "cuBLAS (447)", fontsize=8, color=ACCENT_YELLOW, transform=ax.get_yaxis_transform())

    # Value labels
    for bar, tf in zip(bars, valid_tf):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + 5, str(tf), ha='center', fontsize=9, color=WHITE, fontweight='bold')

    # Mark winner
    winner_idx = valid_ns.index(192)
    ax.text(bars[winner_idx].get_x() + bars[winner_idx].get_width()/2, valid_tf[winner_idx] + 18,
            "← winner (v4)", fontsize=10, color=ACCENT_GREEN, fontweight='bold', ha='center')

    # Failed markers
    fail_x_positions = ["208", "224"]
    for fx in fail_x_positions:
        ax.text(len(valid_ns) - 0.5 + fail_x_positions.index(fx) * 0.6, 350,
                f"WN={fx}\n✗ FAIL", fontsize=8, color=ACCENT_RED, ha='center', fontweight='bold')

    ax.set_xlabel("WARP_N", fontsize=12, fontweight='bold')
    ax.set_ylabel("TFLOPS", fontsize=12, fontweight='bold')
    ax.set_title("WARP_N Sweep (STAGES=2, CONSUMERS=2, TILE_K=64)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(300, 520)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRAY)
    ax.spines['bottom'].set_color(GRAY)

    fig.tight_layout()
    fig.savefig(OUT / "warpn_sweep.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig5_optimization_ladder():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    labels = ["v0\nnaive", "v1\nSMEM", "v2\nTMA+WGMMA", "v3\nwarpspec", "v4\ntuned", "cuBLAS"]
    values = [0.38, 1.51, 284.4, 337.0, 471.3, 447.5]
    colors_l = [ACCENT_RED, ACCENT_ORANGE, ACCENT_BLUE, ACCENT_PURPLE, ACCENT_GREEN, ACCENT_YELLOW]

    bars = ax.barh(range(len(labels)), values, color=colors_l, height=0.65, edgecolor='none')

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_xlabel("TFLOPS (H800 PCIe, 8192³, f16)", fontsize=11, fontweight='bold')
    ax.set_title("The Optimization Ladder", fontsize=14, fontweight='bold', pad=12)

    for bar, val in zip(bars, values):
        x_pos = bar.get_width() + 5 if val > 50 else bar.get_width() + 2
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f"{val:.1f}", va='center', fontsize=10,
                color=WHITE, fontweight='bold')

    # Speedup annotations
    speedups = ["", "3.9×", "188×", "1.19×", "1.40×", ""]
    for i, s in enumerate(speedups):
        if s:
            ax.text(max(values[i], values[i-1]) + 40, i, s, va='center', fontsize=9,
                    color=ACCENT_YELLOW, fontweight='bold')

    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRAY)
    ax.spines['bottom'].set_color(GRAY)
    ax.set_xlim(0, 550)

    fig.tight_layout()
    fig.savefig(OUT / "optimization_ladder.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    fig1_v0_memory_access()
    print("✓ v0_memory_access.png")
    fig2_v1_tile_reuse()
    print("✓ v1_tile_reuse.png")
    fig3_double_buffering()
    print("✓ double_buffering_timeline.png")
    fig4_warpn_sweep()
    print("✓ warpn_sweep.png")
    fig5_optimization_ladder()
    print("✓ optimization_ladder.png")
    print(f"\nAll images saved to {OUT}")
