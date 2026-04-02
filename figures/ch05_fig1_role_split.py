"""
Figure 1: 1P1C role split — producer warpgroup (DMA) and consumer warpgroup (MMA)
on a shared timeline, showing hardware overlap (not sequential handoff).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class Ch05Fig1RoleSplit(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(
            "1P1C: Producer DMA and Consumer MMA Overlap",
            font_size=22,
            color=C["fg"],
            font="Monospace",
        )
        title.to_edge(UP, buff=0.35)
        self.add(title)

        sub = Text(
            "Two warpgroups — different straight-line programs, concurrent time",
            font_size=13,
            color=C["fg2"],
            font="Monospace",
        )
        sub.next_to(title, DOWN, buff=0.12)
        self.add(sub)

        # Timeline baseline
        t0, t1 = LEFT * 5.2 + DOWN * 0.85, RIGHT * 5.2 + DOWN * 0.85
        axis = Line(t0, t1, color=C["dim"], stroke_width=2)
        time_lbl = Text("time →", font_size=12, color=C["dim"], font="Monospace")
        time_lbl.next_to(axis, RIGHT, buff=0.15)
        self.add(axis, time_lbl)

        tick_y = DOWN * 0.95
        for _, x in enumerate([-4.5, -1.5, 1.5, 4.5]):
            tick = Line(
                UP * 0.06 + RIGHT * x + tick_y,
                DOWN * 0.06 + RIGHT * x + tick_y,
                color=C["dim"],
                stroke_width=1,
            )
            self.add(tick)

        # Producer row (warpgroup 0): DMA / TMA
        prod_lbl = Text("Producer (WG0)", font_size=14, color=C["blue"], font="Monospace")
        prod_lbl.move_to(LEFT * 5.8 + UP * 0.55)
        self.add(prod_lbl)

        dma_color = C["blue"]
        dma_segments = [
            (-5.0, 1.4, "DMA"),
            (-3.2, 1.4, "DMA"),
            (-1.4, 1.4, "DMA"),
            (0.4, 1.4, "DMA"),
        ]
        for x0, w, lab in dma_segments:
            r = Rectangle(
                width=w,
                height=0.42,
                fill_color=dma_color,
                fill_opacity=0.35,
                stroke_color=dma_color,
                stroke_width=1.5,
            )
            r.move_to(RIGHT * (x0 + w / 2) + UP * 0.35)
            self.add(r)
            t = Text(lab, font_size=12, color=C["fg"], font="Monospace")
            t.move_to(r)
            self.add(t)

        # Consumer row (warpgroup 1): MMA — shifted to overlap producer bars
        cons_lbl = Text("Consumer (WG1)", font_size=14, color=C["purple_role"], font="Monospace")
        cons_lbl.move_to(LEFT * 5.8 + DOWN * 0.05)
        self.add(cons_lbl)

        mma_color = C["purple_role"]
        mma_segments = [
            (-4.0, 1.2, "MMA"),
            (-2.2, 1.2, "MMA"),
            (-0.4, 1.2, "MMA"),
            (1.4, 1.2, "MMA"),
        ]
        for x0, w, lab in mma_segments:
            r = Rectangle(
                width=w,
                height=0.42,
                fill_color=mma_color,
                fill_opacity=0.35,
                stroke_color=mma_color,
                stroke_width=1.5,
            )
            r.move_to(RIGHT * (x0 + w / 2) + DOWN * 0.35)
            self.add(r)
            t = Text(lab, font_size=12, color=C["fg"], font="Monospace")
            t.move_to(r)
            self.add(t)

        note = Text(
            "inthreads.async — structural roles; overlap is real concurrency, not a branch",
            font_size=12,
            color=C["fg3"],
            font="Monospace",
        )
        note.next_to(axis, DOWN, buff=0.55)
        self.add(note)
