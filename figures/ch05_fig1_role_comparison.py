"""
Figure 1 (ch05): Uniform vs Specialized execution timelines.
Top: single warpgroup does DMA then MMA sequentially (no overlap).
Bottom: two warpgroups — producer DMA and consumer MMA overlap in time.
Vertical (top-bottom) layout for compactness.
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
            "Uniform vs Structured-Concurrent Execution",
            font_size=22,
            color=C["fg"],
            font="Monospace",
        )
        title.to_edge(UP, buff=0.3)
        self.add(title)

        dma_color = C["blue"]
        mma_color = C["purple_role"]

        # --- TOP: Uniform (one warpgroup, sequential) ---
        top_y = 1.8

        top_lbl = Text(
            "Uniform: one warpgroup (sequential)",
            font_size=14,
            color=C["fg2"],
            font="Monospace",
        )
        top_lbl.move_to(UP * top_y)
        self.add(top_lbl)

        wg_lbl = Text("WG0", font_size=12, color=C["fg3"], font="Monospace")
        wg_lbl.move_to(LEFT * 5.8 + UP * (top_y - 0.7))
        self.add(wg_lbl)

        bar_y = top_y - 0.9
        seq_blocks = [
            (-5.0, 1.4, dma_color, "DMA"),
            (-3.4, 1.5, mma_color, "MMA"),
            (-1.7, 1.4, dma_color, "DMA"),
            (-0.1, 1.5, mma_color, "MMA"),
            (1.6, 1.4, dma_color, "DMA"),
            (3.2, 1.5, mma_color, "MMA"),
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
            r.move_to(RIGHT * (x0 + w / 2) + UP * bar_y)
            self.add(r)
            t = Text(lab, font_size=12, color=C["fg"], font="Monospace")
            t.move_to(r)
            self.add(t)

        top_axis = Line(
            LEFT * 5.2 + UP * (bar_y - 0.45),
            RIGHT * 5.2 + UP * (bar_y - 0.45),
            color=C["dim"],
            stroke_width=1.5,
        )
        top_time = Text("time ->", font_size=12, color=C["dim"], font="Monospace")
        top_time.next_to(top_axis, RIGHT, buff=0.08)
        self.add(top_axis, top_time)

        total_top = Text(
            "total = sum(DMA + MMA)  -- no overlap",
            font_size=12,
            color=C["red"],
            font="Monospace",
        )
        total_top.next_to(top_axis, DOWN, buff=0.12)
        self.add(total_top)

        # --- Horizontal separator ---
        sep = DashedLine(
            LEFT * 5.8 + DOWN * 0.05,
            RIGHT * 5.8 + DOWN * 0.05,
            color=C["dim"],
            stroke_width=1,
            dash_length=0.08,
        )
        self.add(sep)

        # --- BOTTOM: Specialized (two warpgroups, overlapping) ---
        bot_y = -0.6

        bot_lbl = Text(
            "Structured concurrent: two warpgroups (overlapping)",
            font_size=14,
            color=C["fg2"],
            font="Monospace",
        )
        bot_lbl.move_to(UP * bot_y)
        self.add(bot_lbl)

        prod_lbl = Text("Producer", font_size=12, color=C["blue"], font="Monospace")
        prod_lbl.move_to(LEFT * 5.4 + UP * (bot_y - 0.5))
        self.add(prod_lbl)

        cons_lbl = Text(
            "Consumer", font_size=12, color=C["purple_role"], font="Monospace"
        )
        cons_lbl.move_to(LEFT * 5.4 + UP * (bot_y - 1.2))
        self.add(cons_lbl)

        prod_bar_y = bot_y - 0.7
        cons_bar_y = bot_y - 1.4

        dma_spec = [
            (-4.0, 1.4, "DMA"),
            (-2.4, 1.4, "DMA"),
            (-0.8, 1.4, "DMA"),
            (0.8, 1.4, "DMA"),
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
            r.move_to(RIGHT * (x0 + w / 2) + UP * prod_bar_y)
            self.add(r)
            t = Text(lab, font_size=12, color=C["fg"], font="Monospace")
            t.move_to(r)
            self.add(t)

        mma_spec = [
            (-3.0, 1.5, "MMA"),
            (-1.3, 1.5, "MMA"),
            (0.4, 1.5, "MMA"),
            (2.1, 1.5, "MMA"),
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
            r.move_to(RIGHT * (x0 + w / 2) + UP * cons_bar_y)
            self.add(r)
            t = Text(lab, font_size=12, color=C["fg"], font="Monospace")
            t.move_to(r)
            self.add(t)

        bot_axis = Line(
            LEFT * 5.2 + UP * (cons_bar_y - 0.45),
            RIGHT * 5.2 + UP * (cons_bar_y - 0.45),
            color=C["dim"],
            stroke_width=1.5,
        )
        bot_time = Text("time ->", font_size=12, color=C["dim"], font="Monospace")
        bot_time.next_to(bot_axis, RIGHT, buff=0.08)
        self.add(bot_axis, bot_time)

        total_bot = Text(
            "total = max(DMA, MMA) x stages  -- overlap",
            font_size=12,
            color=C["green"],
            font="Monospace",
        )
        total_bot.next_to(bot_axis, DOWN, buff=0.12)
        self.add(total_bot)

        # --- Bottom annotation ---
        note = Text(
            "inthreads.async partitions threads into concurrent regions"
            " -- overlap is real concurrency, not interleaving",
            font_size=12,
            color=C["fg3"],
            font="Monospace",
        )
        note.to_edge(DOWN, buff=0.3)
        self.add(note)
