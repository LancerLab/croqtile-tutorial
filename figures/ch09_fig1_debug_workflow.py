"""
Ch09 Fig1: Debugging workflow — compile-time shapes → one tile → sync → layout → GDB.
Top-to-bottom flow with decision-style prompts between stages.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from manim import *

from theme import parse_theme

C, THEME = parse_theme()


class DebugWorkflow(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(
            "Debugging Croqtile kernels: narrowing order",
            font_size=22,
            color=C["fg"],
            font="Monospace",
        )
        title.to_edge(UP, buff=0.28)
        self.add(title)

        def step_box(main: str, detail: str, accent: str):
            r = RoundedRectangle(
                corner_radius=0.12,
                width=5.6,
                height=1.05,
                fill_color=C["fill"],
                fill_opacity=0.9,
                stroke_color=accent,
                stroke_width=2,
            )
            t1 = Text(main, font_size=15, color=C["fg"], font="Monospace")
            t2 = Text(detail, font_size=12, color=C["fg2"], font="Monospace", line_spacing=0.45)
            inner = VGroup(t1, t2).arrange(DOWN, buff=0.12)
            inner.move_to(r.get_center())
            return VGroup(r, inner)

        def decision_lbl():
            return Text("still wrong?", font_size=12, color=C["dim"], font="Monospace")

        # Stack: boxes centered at x=0, decreasing y
        s1 = step_box(
            "1  Compile-time shapes",
            "println!  —  chunkat, subspan, span extents",
            C["blue"],
        )
        s2 = step_box(
            "2  One tile / thread",
            "println with guard  —  block (0,0), single thread",
            C["teal"],
        )
        s3 = step_box(
            "3  Synchronization",
            "event ordering  —  full/empty, producer vs consumer",
            C["orange"],
        )
        s4 = step_box(
            "4  Layout",
            "row vs column major, MMA / swizzle consistency",
            C["purple"],
        )
        s5 = step_box(
            "5  GDB + RTTI",
            "pointer validity, __dbg_* tensors, -g -O0",
            C["red"],
        )

        col = VGroup(s1, s2, s3, s4, s5).arrange(DOWN, buff=0.42)
        col.move_to(ORIGIN + DOWN * 0.15)
        self.add(col)

        # Decision labels + arrows between stacked groups (between centers)
        # col children: 0=s1, 1=s2, ...
        for i in range(4):
            top = col[i]
            bot = col[i + 1]
            mid_y = (top.get_bottom()[1] + bot.get_top()[1]) / 2
            dl = decision_lbl()
            dl.move_to(np.array([2.85, mid_y, 0]))
            a = Arrow(
                top.get_bottom() + DOWN * 0.02,
                bot.get_top() + UP * 0.02,
                buff=0.05,
                stroke_width=2.5,
                color=C["arrow_c"],
                max_tip_length_to_length_ratio=0.14,
            )
            self.add(a, dl)

        foot = Text(
            "Prefer this order: cheap compile-time checks before heavy runtime prints.",
            font_size=12,
            color=C["dim"],
            font="Monospace",
        )
        foot.to_edge(DOWN, buff=0.32)
        self.add(foot)
