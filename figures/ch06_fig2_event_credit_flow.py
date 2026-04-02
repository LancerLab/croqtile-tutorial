"""
Ch06 Fig2: Event credit flow for one pipeline stage (full / empty handshakes).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import *

from theme import parse_theme

C, THEME = parse_theme()


class EventCreditFlow(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(
            "One stage: full / empty credit flow",
            font_size=22,
            color=C["fg"],
            font="Monospace",
        )
        title.to_edge(UP, buff=0.35)
        self.add(title)

        def stage_box(text_lines, color, width=3.4, height=1.55):
            lines = "\n".join(text_lines)
            r = Rectangle(
                width=width,
                height=height,
                fill_color=color,
                fill_opacity=0.12,
                stroke_color=color,
                stroke_width=2,
            )
            t = Text(lines, font_size=13, color=C["fg2"], font="Monospace", line_spacing=0.65)
            g = VGroup(r, t)
            return g

        # Bootstrap
        boot = Text(
            "Bootstrap (consumer): trigger empty[stage]\n→ slot starts free (credit to producer)",
            font_size=13,
            color=C["teal"],
            font="Monospace",
            line_spacing=0.55,
        )
        boot.move_to(UP * 2.35)
        self.add(boot)

        prod = stage_box(
            [
                "Producer",
                "wait empty[stage]",
                "dma.copy → shared",
                "trigger full[stage]",
            ],
            C["blue"],
        )
        prod.move_to(LEFT * 2.85 + DOWN * 0.15)

        cons = stage_box(
            [
                "Consumer",
                "wait full[stage]",
                "MMA (+ mma.commit)",
                "trigger empty[stage]",
            ],
            C["orange"],
        )
        cons.move_to(RIGHT * 2.85 + DOWN * 0.15)

        self.add(prod, cons)

        a1 = Arrow(
            boot.get_bottom() + DOWN * 0.05,
            prod[0].get_top() + UP * 0.02,
            buff=0.08,
            stroke_width=2.5,
            color=C["arrow"],
            max_tip_length_to_length_ratio=0.12,
        )
        a2 = Arrow(
            prod[0].get_right() + RIGHT * 0.02,
            cons[0].get_left() + LEFT * 0.02,
            buff=0.08,
            stroke_width=2.5,
            color=C["purple"],
            max_tip_length_to_length_ratio=0.1,
        )
        handoff = Text("handoff", font_size=12, color=C["purple"], font="Monospace")
        handoff.next_to(a2, UP, buff=0.02)

        a3 = CurvedArrow(
            cons[0].get_bottom() + LEFT * 0.5 + DOWN * 0.05,
            prod[0].get_bottom() + RIGHT * 0.5 + DOWN * 0.05,
            angle=-2.1,
            stroke_width=2.5,
            color=C["green"],
        )
        cycle = Text("cycle repeats\nfor next iv_k / ring slot", font_size=12, color=C["green"], font="Monospace")
        cycle.next_to(a3, DOWN, buff=0.15)

        self.add(a1, a2, handoff, a3, cycle)

        foot = Text(
            "Ring index stage = iv_k % MATMUL_STAGES reuses physical slots; events serialize access.",
            font_size=12,
            color=C["dim"],
            font="Monospace",
        )
        foot.to_edge(DOWN, buff=0.4)
        self.add(foot)
