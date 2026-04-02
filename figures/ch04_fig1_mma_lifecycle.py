"""
Figure 1: MMA lifecycle — fill → load → multiply → store,
with register vs shared-memory placement.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import *

from theme import parse_theme

C, THEME = parse_theme()


class MMALifecycle(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(
            "Tensor-Core MMA Lifecycle (one K step)",
            font_size=22,
            color=C["fg"],
            font="Monospace",
        )
        title.to_edge(UP, buff=0.3)
        self.add(title)

        # Legend
        reg_box = Square(side_length=0.22, color=C["reg_c"], fill_opacity=0.35, stroke_width=1.2)
        smem_box = Square(side_length=0.22, color=C["smem_c"], fill_opacity=0.35, stroke_width=1.2)
        leg_reg = Text("Registers (mc, ma, mb)", font_size=12, color=C["fg2"], font="Monospace")
        leg_smem = Text("Shared memory", font_size=12, color=C["fg2"], font="Monospace")
        leg_row = VGroup(
            reg_box, leg_reg,
            smem_box.next_to(leg_reg, RIGHT, buff=0.35).shift(LEFT * 0.1),
            leg_smem.next_to(smem_box, RIGHT, buff=0.12),
        ).arrange(RIGHT, buff=0.15)
        leg_row.next_to(title, DOWN, buff=0.2)
        self.add(leg_row)

        y_main = -0.15
        step_w = 2.85
        x0 = -4.4

        steps = [
            {
                "n": "1  fill",
                "op": "mma.fill",
                "detail": "init accumulator tile mc",
                "where": "reg",
                "arrow_to": "mc in registers",
            },
            {
                "n": "2  load",
                "op": "mma.load",
                "detail": "A,B tiles from smem",
                "where": "smem_to_reg",
                "arrow_to": "ma, mb in registers",
            },
            {
                "n": "3  mma",
                "op": "mma.row.row",
                "detail": "C += A × B",
                "where": "reg",
                "arrow_to": "accumulate in mc",
            },
            {
                "n": "4  store",
                "op": "mma.store",
                "detail": "flush mc to smem",
                "where": "reg_to_smem",
                "arrow_to": "output tile in smem",
            },
        ]

        for i, st in enumerate(steps):
            x = x0 + i * step_w
            center = np.array([x, y_main, 0])

            if st["where"] == "reg":
                fill_c = C["reg_c"]
                stroke_c = C["reg_c"]
            elif st["where"] == "smem_to_reg":
                fill_c = C["smem_c"]
                stroke_c = C["reg_c"]
            else:
                fill_c = C["reg_c"]
                stroke_c = C["smem_c"]

            box = RoundedRectangle(
                corner_radius=0.12,
                width=2.55,
                height=1.55,
                fill_color=fill_c,
                fill_opacity=0.12,
                stroke_color=stroke_c,
                stroke_width=2,
            )
            box.move_to(center)

            n_t = Text(st["n"], font_size=14, color=C["yellow"], font="Monospace")
            op_t = Text(st["op"], font_size=16, color=C["blue"], font="Monospace")
            det_t = Text(st["detail"], font_size=12, color=C["fg"], font="Monospace")
            col = VGroup(n_t, op_t, det_t).arrange(DOWN, buff=0.12)
            col.move_to(center)
            self.add(box, col)

            if i < len(steps) - 1:
                a_start = center + RIGHT * 1.35
                a_end = center + RIGHT * (step_w - 1.35)
                arr = Arrow(
                    a_start,
                    a_end,
                    buff=0,
                    stroke_width=2.5,
                    color=C["arrow_c"],
                    max_tip_length_to_length_ratio=0.12,
                )
                self.add(arr)

        foot = Text(
            "Loop 2–3 over K; run 4 once to write the accumulated tile.",
            font_size=12,
            color=C["fg3"],
            font="Monospace",
        )
        foot.to_edge(DOWN, buff=0.35)
        self.add(foot)

        hint = Text(
            "Hardware: ~16×16×16 FP16 tile per instruction; Croqtile hides fragment layouts.",
            font_size=12,
            color=C["dim"],
            font="Monospace",
        )
        hint.next_to(foot, UP, buff=0.15)
        self.add(hint)
