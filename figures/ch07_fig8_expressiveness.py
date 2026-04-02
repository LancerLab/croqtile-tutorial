"""
Ch07 Fig8: Expressiveness vs Performance.
Three concentric circles:
  Outer (largest): CUDA expressible range — wide but includes many slow patterns.
  Middle: Croqtile range — restricted but fully contains the sweet spot.
  Inner (smallest): Performance sweet spot — coalesced, conflict-free, aligned.

Key relationship: Croqtile ⊃ sweet spot, CUDA ⊃ Croqtile.
By restricting expressiveness, Croqtile guarantees every program is in the sweet spot.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *
import numpy as np

C, THEME = parse_theme()


class ExpressivenessVsPerformance(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(
            "Expressiveness vs Performance",
            font_size=28, color=C["fg"], font="Monospace",
        )
        title.to_edge(UP, buff=0.25)
        self.add(title)

        sub = Text(
            "data-movement patterns: who can express what?",
            font_size=14, color=C["fg2"], font="Monospace",
        )
        sub.next_to(title, DOWN, buff=0.08)
        self.add(sub)

        center = DOWN * 0.15

        # Outer: CUDA (largest)
        cuda_r = 2.8
        cuda_circle = Circle(
            radius=cuda_r, color=C["orange"], stroke_width=3,
            fill_color=C["orange"], fill_opacity=0.04,
        )
        cuda_circle.move_to(center)
        self.add(cuda_circle)

        cuda_lbl = Text(
            "CUDA expressible range",
            font_size=16, color=C["orange"], font="Monospace",
        )
        cuda_lbl.move_to(cuda_circle.get_top() + DOWN * 0.32)
        self.add(cuda_lbl)

        # Middle: Croqtile — larger than sweet spot, fully contains it
        croq_r = 1.7
        croq_circle = Circle(
            radius=croq_r, color=C["blue"], stroke_width=3.5,
            fill_color=C["blue"], fill_opacity=0.10,
        )
        croq_circle.move_to(center)
        self.add(croq_circle)

        croq_lbl = Text(
            "Croqtile range",
            font_size=17, color=C["blue"], font="Monospace",
        )
        croq_lbl.move_to(center + UP * 1.15)
        self.add(croq_lbl)

        croq_sub = Text(
            "dma.copy · tma.copy · swizzle",
            font_size=13, color=C["fg2"], font="Monospace",
        )
        croq_sub.move_to(center + UP * 0.82)
        self.add(croq_sub)

        # Inner: Performance sweet spot (smallest)
        sweet_r = 0.9
        sweet_circle = Circle(
            radius=sweet_r, color=C["green"], stroke_width=3,
            fill_color=C["green"], fill_opacity=0.14,
        )
        sweet_circle.move_to(center)
        self.add(sweet_circle)

        sweet_lbl = Text(
            "Performance",
            font_size=16, color=C["green"], font="Monospace",
        )
        sweet_lbl.move_to(center + UP * 0.22)
        self.add(sweet_lbl)

        sweet_lbl2 = Text(
            "sweet spot",
            font_size=16, color=C["green"], font="Monospace",
        )
        sweet_lbl2.move_to(center + DOWN * 0.06)
        self.add(sweet_lbl2)

        sweet_sub = Text(
            "coalesced · aligned · conflict-free",
            font_size=12, color=C["fg3"], font="Monospace",
        )
        sweet_sub.move_to(center + DOWN * 0.38)
        self.add(sweet_sub)

        # Bad patterns in the outer ring (CUDA-only zone, outside Croqtile)
        bad_patterns = [
            ("strided\nuncoalesced",   2.35,  55),
            ("bank\nconflicts",        2.35, 115),
            ("misaligned\nswizzle",    2.35, 180),
            ("scalar\nloads",          2.35, 240),
            ("divergent\naddressing",  2.35, 320),
        ]
        for txt, radius, angle_deg in bad_patterns:
            angle = np.deg2rad(angle_deg)
            pos = center + RIGHT * radius * np.cos(angle) + UP * radius * np.sin(angle)
            x_mark = Text("\u2717", font_size=18, color=C["red"], font="Monospace")
            x_mark.move_to(pos)
            label = Text(txt, font_size=12, color=C["red"], font="Monospace")
            label.next_to(x_mark, DOWN, buff=0.04)
            self.add(x_mark, label)

        # Annotation: Croqtile ⊇ sweet spot
        brace_note = Text(
            "Croqtile \u2287 sweet spot",
            font_size=14, color=C["blue"], font="Monospace",
        )
        brace_note.move_to(center + DOWN * 1.35)
        self.add(brace_note)

        foot = Text(
            "every Croqtile program is in the sweet spot \u2014 trade expressiveness for guaranteed performance",
            font_size=12, color=C["dim"], font="Monospace",
        )
        foot.to_edge(DOWN, buff=0.22)
        self.add(foot)
