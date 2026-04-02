"""
Figure 2: Persistent kernel — fixed block count, linear tile striping,
with an if guard skipping out-of-bounds tile indices.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class Ch05Fig2PersistentKernel(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(
            "Persistent Kernel: Fixed Blocks, Striped Tiles, if Guard",
            font_size=20,
            color=C["fg"],
            font="Monospace",
        )
        title.to_edge(UP, buff=0.35)
        self.add(title)

        sub = Text(
            "tile_id = tile_iter # block_id   ·   skip when tile_id ≥ total_tiles",
            font_size=12,
            color=C["fg2"],
            font="Monospace",
        )
        sub.next_to(title, DOWN, buff=0.12)
        self.add(sub)

        # Parameters (illustrative)
        num_sms = 4
        total_tiles = 10
        n_cols = 5
        n_rows = 3
        cell_w, cell_h = 0.85, 0.55
        gap = 0.12
        origin = LEFT * 3.8 + UP * 0.85

        # Linear tile indices in a grid layout (0..14 slots, dim invalid)
        block_colors = [C["blue"], C["orange"], C["teal"], C["pink"]]
        idx = 0
        for row in range(n_rows):
            for col in range(n_cols):
                x = origin + RIGHT * col * (cell_w + gap) + DOWN * row * (cell_h + gap)
                valid = idx < total_tiles
                fill = C["green"] if valid else C["dim"]
                stroke = C["stroke"] if valid else C["red"]
                op = 0.28 if valid else 0.08
                r = Rectangle(
                    width=cell_w,
                    height=cell_h,
                    fill_color=fill,
                    fill_opacity=op,
                    stroke_color=stroke,
                    stroke_width=2 if not valid else 1.2,
                )
                r.move_to(x)
                self.add(r)
                if valid:
                    # Color by block_id = tile_id % NUM_SMS
                    bc = block_colors[idx % num_sms]
                    inner = Rectangle(
                        width=cell_w - 0.12,
                        height=cell_h - 0.22,
                        fill_color=bc,
                        fill_opacity=0.45,
                        stroke_width=0,
                    )
                    inner.move_to(x)
                    self.add(inner)
                    tid = Text(str(idx), font_size=14, color=C["fg"], font="Monospace")
                    tid.move_to(x + UP * 0.06)
                    self.add(tid)
                    bid = Text(f"b{idx % num_sms}", font_size=12, color=C["fg2"], font="Monospace")
                    bid.move_to(x + DOWN * 0.12)
                    self.add(bid)
                else:
                    cross1 = Line(
                        x + LEFT * cell_w * 0.35 + UP * cell_h * 0.25,
                        x + RIGHT * cell_w * 0.35 + DOWN * cell_h * 0.25,
                        color=C["red"],
                        stroke_width=1.5,
                    )
                    cross2 = Line(
                        x + LEFT * cell_w * 0.35 + DOWN * cell_h * 0.25,
                        x + RIGHT * cell_w * 0.35 + UP * cell_h * 0.25,
                        color=C["red"],
                        stroke_width=1.5,
                    )
                    self.add(cross1, cross2)
                    skip = Text("—", font_size=16, color=C["red"], font="Monospace")
                    skip.move_to(x)
                    self.add(skip)
                idx += 1

        legend_title = Text("Stripe key (tile_id % NUM_SMS)", font_size=12, color=C["fg2"], font="Monospace")
        legend_title.move_to(RIGHT * 4.2 + UP * 1.1)
        self.add(legend_title)
        for i in range(num_sms):
            row = DOWN * i * 0.38
            sq = Square(side_length=0.22, fill_color=block_colors[i], fill_opacity=0.5, stroke_width=0)
            sq.move_to(RIGHT * 3.35 + UP * 0.75 + row)
            lb = Text(f"block_id = {i}", font_size=12, color=C["fg"], font="Monospace")
            lb.next_to(sq, RIGHT, buff=0.2)
            self.add(sq, lb)

        guard_box = Rectangle(
            width=5.6,
            height=0.65,
            fill_color=C["red"],
            fill_opacity=0.06,
            stroke_color=C["red"],
            stroke_width=1.2,
        )
        guard_box.move_to(LEFT * 0.2 + DOWN * 1.85)
        self.add(guard_box)
        guard_txt = Text(
            'if (tile_id < total_tiles) { … compute tile … }',
            font_size=12,
            color=C["fg"],
            font="Monospace",
        )
        guard_txt.move_to(guard_box)
        self.add(guard_txt)

        foot = Text(
            "Grid launch size fixed (e.g. NUM_SMS); foreach may pad — guard avoids OOB stores",
            font_size=12,
            color=C["dim"],
            font="Monospace",
        )
        foot.next_to(guard_box, DOWN, buff=0.45)
        self.add(foot)
