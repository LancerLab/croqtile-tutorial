"""
Shared color palettes for Croqtile tutorial figures.
Provides dark and light themes so every Manim script can generate
two variants that toggle with MkDocs Material's #only-dark / #only-light.
"""
import os
import sys

DARK = {
    "bg":        "#0a0a0a",
    "fg":        "#e8f5e9",
    "fg2":       "#a5d6a7",
    "fg3":       "#66bb6a",
    "dim":       "#4a6350",
    "fill":      "#1a2e1c",
    "stroke":    "#2e4830",
    "separator": "#2e4830",

    "green":     "#4ade80",
    "green_dk":  "#1B5E20",
    "blue":      "#60a5fa",
    "orange":    "#fb923c",
    "red":       "#f87171",
    "yellow":    "#facc15",
    "purple":    "#c084fc",
    "pink":      "#f472b6",
    "teal":      "#5eead4",
    "gold":      "#fbbf24",

    "tile":      "#4ade80",
    "elem":      "#fb923c",
    "arrow":     "#60a5fa",
    "global_c":  "#fb923c",
    "shared_c":  "#60a5fa",
    "local_c":   "#4ade80",
    "future_c":  "#c084fc",
    "data_c":    "#4ade80",

    "dram_c":    "#374940",
    "l2_c":      "#3d5c47",
    "sm_c":      "#1B5E20",
    "smem_c":    "#2E7D32",
    "reg_c":     "#4CAF50",
    "label_c":   "#a5d6a7",
    "arrow_c":   "#60a5fa",

    "lhs_c":     "#60a5fa",
    "rhs_c":     "#fb923c",
    "out_c":     "#4ade80",

    "grid_c":    "#374940",
    "hl_c":      "#facc15",
    "dim_c":     "#66bb6a",
    "extent_c":  "#f472b6",

    "term_bg":   "#0d1117",
    "term_bar":  "#161b22",
    "term_border": "#30363d",
    "prompt_c":  "#4ade80",
    "cmd_c":     "#e6edf3",
    "output_c":  "#8b949e",
    "success_c": "#3fb950",
    "cursor_c":  "#58a6ff",

    "blue_tile":  "#60a5fa",
    "orange_tile": "#fb923c",
    "red_accent":  "#f87171",
    "green_ok":    "#4ade80",
    "gray_bg":     "#1a2e1c",
    "purple_role": "#c084fc",
}

LIGHT = {
    "bg":        "#ffffff",
    "fg":        "#0f172a",
    "fg2":       "#334155",
    "fg3":       "#475569",
    "dim":       "#64748b",
    "fill":      "#f1f5f9",
    "stroke":    "#cbd5e1",
    "separator": "#e2e8f0",

    "green":     "#16a34a",
    "green_dk":  "#166534",
    "blue":      "#2563eb",
    "orange":    "#ea580c",
    "red":       "#dc2626",
    "yellow":    "#a16207",
    "purple":    "#7c3aed",
    "pink":      "#db2777",
    "teal":      "#0d9488",
    "gold":      "#b45309",

    "tile":      "#16a34a",
    "elem":      "#ea580c",
    "arrow":     "#2563eb",
    "global_c":  "#ea580c",
    "shared_c":  "#2563eb",
    "local_c":   "#16a34a",
    "future_c":  "#7c3aed",
    "data_c":    "#16a34a",

    "dram_c":    "#e2e8f0",
    "l2_c":      "#cbd5e1",
    "sm_c":      "#475569",
    "smem_c":    "#16a34a",
    "reg_c":     "#22c55e",
    "label_c":   "#334155",
    "arrow_c":   "#2563eb",

    "lhs_c":     "#2563eb",
    "rhs_c":     "#ea580c",
    "out_c":     "#16a34a",

    "grid_c":    "#f1f5f9",
    "hl_c":      "#a16207",
    "dim_c":     "#475569",
    "extent_c":  "#db2777",

    "term_bg":   "#f8fafc",
    "term_bar":  "#e2e8f0",
    "term_border": "#cbd5e1",
    "prompt_c":  "#16a34a",
    "cmd_c":     "#1e293b",
    "output_c":  "#64748b",
    "success_c": "#16a34a",
    "cursor_c":  "#2563eb",

    "blue_tile":  "#2563eb",
    "orange_tile": "#ea580c",
    "red_accent":  "#dc2626",
    "green_ok":    "#16a34a",
    "gray_bg":     "#f1f5f9",
    "purple_role": "#7c3aed",
}

THEMES = {"dark": DARK, "light": LIGHT}


def get_colors(name="dark"):
    return THEMES[name]


def parse_theme():
    """Read MANIM_THEME env var (dark|light), default to dark."""
    theme = os.environ.get("MANIM_THEME", "dark")
    if theme not in THEMES:
        theme = "dark"
    return get_colors(theme), theme
