"""
Figure 2: SM86 (Ampere) vs SM90 (Hopper) — cooperation scope for MMA.
Left: one warp, : group. Right: four warps, : group-4 (WGMMA).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import *

from theme import parse_theme

C, THEME = parse_theme()


class SM86vsSM90(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(
            "MMA cooperation scope: Ampere vs Hopper",
            font_size=22,
            color=C["fg"],
            font="Monospace",
        )
        title.to_edge(UP, buff=0.28)
        self.add(title)

        def panel(label, subtitle, threads, warps, spec, ops, accent, shift_x, n_threads):
            header = Text(label, font_size=18, color=accent, font="Monospace")
            sub = Text(subtitle, font_size=12, color=C["fg2"], font="Monospace")
            t1 = Text(threads, font_size=13, color=C["fg"], font="Monospace")
            t2 = Text(warps, font_size=12, color=C["fg3"], font="Monospace")
            sp = Text(spec, font_size=15, color=C["yellow"], font="Monospace")
            op = Text(ops, font_size=12, color=C["fg2"], font="Monospace")

            # Mini warp visualization (explicit n_threads — avoid "32" in "128")
            dots = VGroup()
            per_row = 16 if n_threads == 128 else 8
            rows = (n_threads + per_row - 1) // per_row
            for r in range(rows):
                for c in range(min(per_row, n_threads - r * per_row)):
                    d = Dot(radius=0.06, color=accent, fill_opacity=0.85)
                    d.move_to(
                        np.array(
                            [
                                (c - per_row / 2) * 0.22,
                                (rows / 2 - r) * 0.22,
                                0,
                            ]
                        )
                    )
                    dots.add(d)

            cap = Text(
                f"{n_threads} threads",
                font_size=12,
                color=C["dim"],
                font="Monospace",
            )
            cap.next_to(dots, DOWN, buff=0.12)

            body = VGroup(header, sub, t1, t2, sp, op, dots, cap).arrange(
                DOWN, buff=0.14, aligned_edge=LEFT
            )
            frame = RoundedRectangle(
                corner_radius=0.14,
                width=5.6,
                height=6.2,
                fill_color=accent,
                fill_opacity=0.06,
                stroke_color=accent,
                stroke_width=2,
            )
            grp = VGroup(frame, body)
            grp.move_to(ORIGIN + RIGHT * shift_x)
            return grp

        left = panel(
            "SM86 (Ampere)",
            "warp-scoped tensor cores",
            "1 warp  =  32 threads",
            "one SIMD lockstep unit",
            "parallel … : group",
            "wmma / mma.sync class ISA",
            C["blue"],
            -3.15,
            32,
        )
        right = panel(
            "SM90 (Hopper)",
            "warpgroup WGMMA",
            "4 warps  =  128 threads",
            "cooperative wide MMA",
            "parallel … : group-4",
            "same mnemonics, wider issue",
            C["purple"],
            3.15,
            128,
        )

        self.add(left, right)

        vs = Text("vs", font_size=20, color=C["fg3"], font="Monospace")
        vs.move_to(ORIGIN + UP * 0.2)
        self.add(vs)

        foot = Text(
            "Croqtile maps : group vs : group-4 to the correct register groups and barriers.",
            font_size=12,
            color=C["fg3"],
            font="Monospace",
        )
        foot.to_edge(DOWN, buff=0.32)
        self.add(foot)
