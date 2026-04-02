"""
Figure 3: Space Specifiers — the GPU execution hierarchy.
Shows block > group-4 (warpgroup) > group (warp) > thread,
with thread counts and typical operations at each level.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class SpaceSpecifiers(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("Space Specifiers → GPU Execution Hierarchy",
                      font_size=22, color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.25)
        self.add(title)

        levels = [
            {
                "spec": ": block", "hw": "Thread Block (CTA)",
                "threads": "up to 1024 threads", "color": C["orange"],
                "ops": "grid-level work distribution",
                "mem": "shared memory visible here",
            },
            {
                "spec": ": group-4", "hw": "Warpgroup (128 threads)",
                "threads": "4 warps = 128 threads", "color": C["purple"],
                "ops": "WGMMA (Hopper tensor cores)",
                "mem": "cooperative wide MMA",
            },
            {
                "spec": ": group", "hw": "Warp (32 threads)",
                "threads": "32 threads in lockstep", "color": C["blue"],
                "ops": "mma.sync / wmma (Ampere)",
                "mem": "warp-level shuffles",
            },
            {
                "spec": ": thread", "hw": "Thread",
                "threads": "1 thread", "color": C["green"],
                "ops": "scalar / per-element SIMD",
                "mem": "registers (local memory)",
            },
        ]

        y_start = 2.0
        y_step = 1.45

        for i, lv in enumerate(levels):
            y = y_start - i * y_step
            color = lv["color"]

            outer = Rectangle(width=12, height=1.2, fill_color=color,
                              fill_opacity=0.08, stroke_color=color, stroke_width=1.5)
            outer.move_to(UP * y)

            spec_t = Text(lv["spec"], font_size=18, color=color, font="Monospace")
            spec_t.move_to(LEFT * 4.8 + UP * y)

            hw_t = Text(lv["hw"], font_size=14, color=C["fg"], font="Monospace")
            hw_t.move_to(LEFT * 1.8 + UP * y + UP * 0.15)

            threads_t = Text(lv["threads"], font_size=12, color=C["fg3"],
                              font="Monospace")
            threads_t.move_to(LEFT * 1.8 + UP * y + DOWN * 0.2)

            ops_t = Text(lv["ops"], font_size=12, color=color, font="Monospace")
            ops_t.move_to(RIGHT * 2.0 + UP * y + UP * 0.15)

            mem_t = Text(lv["mem"], font_size=12, color=C["fg3"], font="Monospace")
            mem_t.move_to(RIGHT * 2.0 + UP * y + DOWN * 0.2)

            self.add(outer, spec_t, hw_t, threads_t, ops_t, mem_t)

            if i < len(levels) - 1:
                arr = Arrow(UP * (y - 0.65), UP * (y - y_step + 0.65),
                            buff=0, stroke_width=1.5, color=C["dim"],
                            max_tip_length_to_length_ratio=0.08)
                contains = Text("contains ↓", font_size=12, color=C["dim"],
                                font="Monospace")
                contains.next_to(arr, RIGHT, buff=0.1)
                self.add(arr, contains)

        # Column headers
        hdr_spec = Text("Specifier", font_size=12, color=C["fg3"], font="Monospace")
        hdr_spec.move_to(LEFT * 4.8 + UP * 2.8)
        hdr_hw = Text("Hardware Unit", font_size=12, color=C["fg3"], font="Monospace")
        hdr_hw.move_to(LEFT * 1.8 + UP * 2.8)
        hdr_ops = Text("Typical Operations", font_size=12, color=C["fg3"], font="Monospace")
        hdr_ops.move_to(RIGHT * 2.0 + UP * 2.8)
        self.add(hdr_spec, hdr_hw, hdr_ops)
