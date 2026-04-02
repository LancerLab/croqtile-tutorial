"""
Figure 1 (ch05): Uniform vs Specialized execution timelines.
Left: single warpgroup does DMA then MMA sequentially (no overlap).
Right: two warpgroups — producer DMA and consumer MMA overlap in time.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class RoleComparison(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(
            "Uniform vs Role-Specialized Execution",
            font_size=22,
            color=C["fg"],
            font="Monospace",
        )
        title.to_edge(UP, buff=0.3)
        self.add(title)

        # --- LEFT: Uniform (one warpgroup, sequential) ---
        left_lbl = Text(
            "Uniform: one warpgroup",
            font_size=14,
            color=C["fg2"],
            font="Monospace",
        )
        left_lbl.move_to(LEFT * 3.5 + UP * 2.2)
        self.add(left_lbl)

        left_sub = Text(
            "DMA and MMA take turns — no overlap",
            font_size=11,
            color=C["dim"],
            font="Monospace",
        )
        left_sub.next_to(left_lbl, DOWN, buff=0.1)
        self.add(left_sub)

        wg_lbl = Text("WG0", font_size=12, color=C["fg3"], font="Monospace")
        wg_lbl.move_to(LEFT * 5.8 + UP * 1.1)
        self.add(wg_lbl)

        dma_color = C["blue"]
        mma_color = C["purple_role"]

        seq_blocks = [
            (-5.4, 1.1, dma_color, "DMA"),
            (-4.1, 1.2, mma_color, "MMA"),
            (-2.7, 1.1, dma_color, "DMA"),
            (-1.4, 1.2, mma_color, "MMA"),
        ]
        for x0, w, col, lab in seq_blocks:
            r = Rectangle(
                width=w,
                height=0.42,
                fill_color=col,
                fill_opacity=0.35,
                stroke_color=col,
                stroke_width=1.5,
            )
            r.move_to(RIGHT * (x0 + w / 2) + UP * 0.85)
            self.add(r)
            t = Text(lab, font_size=11, color=C["fg"], font="Monospace")
            t.move_to(r)
            self.add(t)

        left_axis = Line(
            LEFT * 5.5 + UP * 0.35,
            RIGHT * 0.2 + UP * 0.35,
            color=C["dim"],
            stroke_width=1.5,
        )
        left_time = Text("time →", font_size=10, color=C["dim"], font="Monospace")
        left_time.next_to(left_axis, RIGHT, buff=0.08)
        self.add(left_axis, left_time)

        total_left = Text(
            "total ≈ 4 × (DMA + MMA)",
            font_size=11,
            color=C["red"],
            font="Monospace",
        )
        total_left.next_to(left_axis, DOWN, buff=0.15)
        self.add(total_left)

        # --- Divider ---
        divider = DashedLine(
            UP * 2.5,
            DOWN * 2.5,
            color=C["dim"],
            stroke_width=1,
            dash_length=0.08,
        )
        self.add(divider)

        # --- RIGHT: Specialized (two warpgroups, overlapping) ---
        right_lbl = Text(
            "Specialized: two warpgroups",
            font_size=14,
            color=C["fg2"],
            font="Monospace",
        )
        right_lbl.move_to(RIGHT * 3.5 + UP * 2.2)
        self.add(right_lbl)

        right_sub = Text(
            "Producer DMA + Consumer MMA overlap",
            font_size=11,
            color=C["dim"],
            font="Monospace",
        )
        right_sub.next_to(right_lbl, DOWN, buff=0.1)
        self.add(right_sub)

        prod_lbl = Text("Producer", font_size=12, color=C["blue"], font="Monospace")
        prod_lbl.move_to(RIGHT * 1.0 + UP * 1.35)
        self.add(prod_lbl)

        cons_lbl = Text("Consumer", font_size=12, color=C["purple_role"], font="Monospace")
        cons_lbl.move_to(RIGHT * 1.0 + UP * 0.5)
        self.add(cons_lbl)

        dma_spec = [
            (1.6, 1.1, "DMA"),
            (2.9, 1.1, "DMA"),
            (4.2, 1.1, "DMA"),
        ]
        for x0, w, lab in dma_spec:
            r = Rectangle(
                width=w,
                height=0.42,
                fill_color=dma_color,
                fill_opacity=0.35,
                stroke_color=dma_color,
                stroke_width=1.5,
            )
            r.move_to(RIGHT * (x0 + w / 2) + UP * 1.1)
            self.add(r)
            t = Text(lab, font_size=11, color=C["fg"], font="Monospace")
            t.move_to(r)
            self.add(t)

        mma_spec = [
            (2.2, 1.2, "MMA"),
            (3.5, 1.2, "MMA"),
            (4.8, 1.2, "MMA"),
        ]
        for x0, w, lab in mma_spec:
            r = Rectangle(
                width=w,
                height=0.42,
                fill_color=mma_color,
                fill_opacity=0.35,
                stroke_color=mma_color,
                stroke_width=1.5,
            )
            r.move_to(RIGHT * (x0 + w / 2) + UP * 0.25)
            self.add(r)
            t = Text(lab, font_size=11, color=C["fg"], font="Monospace")
            t.move_to(r)
            self.add(t)

        right_axis = Line(
            RIGHT * 0.8 + DOWN * 0.3,
            RIGHT * 6.5 + DOWN * 0.3,
            color=C["dim"],
            stroke_width=1.5,
        )
        right_time = Text("time →", font_size=10, color=C["dim"], font="Monospace")
        right_time.next_to(right_axis, RIGHT, buff=0.08)
        self.add(right_axis, right_time)

        total_right = Text(
            "total ≈ max(DMA, MMA) × stages",
            font_size=11,
            color=C["green"],
            font="Monospace",
        )
        total_right.next_to(right_axis, DOWN, buff=0.15)
        self.add(total_right)

        # --- Bottom annotations ---
        note = Text(
            "inthreads.async assigns static roles — overlap is real concurrency, not interleaving",
            font_size=11,
            color=C["fg3"],
            font="Monospace",
        )
        note.to_edge(DOWN, buff=0.5)
        self.add(note)
