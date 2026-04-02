"""
Figure 1: Virtual Parallelism — abstract tasks scheduled on different hardware.
Shows that "parallel" is a virtual concept; the same 8 tasks can be mapped
to 1-at-a-time (sequential), 4-at-a-time (CPU), or 8-at-a-time (GPU).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()

class VirtualParallelism(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("Parallelism Is a Virtual Concept", font_size=26,
                      color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.add(title)

        subtitle = Text("8 independent tasks — same work, different schedules",
                         font_size=14, color=C["fg2"], font="Monospace")
        subtitle.next_to(title, DOWN, buff=0.15)
        self.add(subtitle)

        task_colors = [C["blue"], C["orange"], C["green"], C["purple"],
                       C["teal"], C["pink"], C["yellow"], C["red"]]

        def make_task_row(y_pos, label, n_cols, tasks_per_col):
            grp = VGroup()
            lbl = Text(label, font_size=14, color=C["fg2"], font="Monospace")
            lbl.move_to(LEFT * 5.5 + UP * y_pos)
            grp.add(lbl)

            task_idx = 0
            col_start_x = -3.0
            col_width = 0.45
            col_gap = 0.15
            row_gap = 0.05

            for col in range(n_cols):
                col_x = col_start_x + col * (tasks_per_col * col_width + col_gap * 2)
                for row in range(tasks_per_col):
                    if task_idx >= 8:
                        break
                    r = Rectangle(width=col_width, height=0.35,
                                  fill_color=task_colors[task_idx],
                                  fill_opacity=0.7,
                                  stroke_color=task_colors[task_idx],
                                  stroke_width=1)
                    r.move_to(RIGHT * (col_x + row * (col_width + row_gap)) + UP * y_pos)
                    t = Text(f"T{task_idx}", font_size=12, color=C["fg"],
                             font="Monospace").move_to(r)
                    grp.add(r, t)
                    task_idx += 1

            time_arrow = Arrow(
                start=LEFT * 3.2 + UP * (y_pos - 0.4),
                end=RIGHT * 5.5 + UP * (y_pos - 0.4),
                buff=0, stroke_width=1, color=C["dim"],
                max_tip_length_to_length_ratio=0.02
            )
            time_lbl = Text("time →", font_size=12, color=C["dim"],
                            font="Monospace")
            time_lbl.next_to(time_arrow, DOWN, buff=0.05).align_to(time_arrow, RIGHT)
            grp.add(time_arrow, time_lbl)
            return grp

        # Sequential: 1-at-a-time (8 columns of 1)
        seq_label = Text("Sequential", font_size=16, color=C["elem"],
                         font="Monospace")
        seq_label.move_to(LEFT * 5.2 + UP * 1.5)
        seq_sub = Text("1 core", font_size=12, color=C["fg3"], font="Monospace")
        seq_sub.next_to(seq_label, DOWN, buff=0.08)
        self.add(seq_label, seq_sub)

        seq_x = -3.0
        for i in range(8):
            r = Rectangle(width=0.5, height=0.4, fill_color=task_colors[i],
                           fill_opacity=0.7, stroke_color=task_colors[i], stroke_width=1)
            r.move_to(RIGHT * (seq_x + i * 0.6) + UP * 1.5)
            t = Text(f"T{i}", font_size=12, color=C["fg"], font="Monospace").move_to(r)
            self.add(r, t)

        arr1 = Arrow(LEFT * 3.2 + UP * 0.95, RIGHT * 2.2 + UP * 0.95, buff=0,
                     stroke_width=1, color=C["dim"], max_tip_length_to_length_ratio=0.03)
        self.add(arr1)
        self.add(Text("time →", font_size=12, color=C["dim"], font="Monospace"
                       ).next_to(arr1, DOWN, buff=0.03).align_to(arr1, RIGHT))

        # 4-wide: CPU style (2 columns of 4)
        cpu_label = Text("4-wide (CPU)", font_size=16, color=C["blue"],
                         font="Monospace")
        cpu_label.move_to(LEFT * 5.2 + DOWN * 0.1)
        cpu_sub = Text("4 cores", font_size=12, color=C["fg3"], font="Monospace")
        cpu_sub.next_to(cpu_label, DOWN, buff=0.08)
        self.add(cpu_label, cpu_sub)

        cpu_x = -3.0
        for step in range(2):
            for lane in range(4):
                idx = step * 4 + lane
                r = Rectangle(width=0.5, height=0.4, fill_color=task_colors[idx],
                               fill_opacity=0.7, stroke_color=task_colors[idx], stroke_width=1)
                r.move_to(RIGHT * (cpu_x + step * 2.8 + lane * 0.0) + DOWN * (0.35 - lane * 0.45))
                # stack vertically within each time step
                r.move_to(RIGHT * (cpu_x + step * 2.5) + DOWN * (lane * 0.45 - 0.55))
                t = Text(f"T{idx}", font_size=12, color=C["fg"], font="Monospace").move_to(r)
                self.add(r, t)

        arr2 = Arrow(LEFT * 3.2 + DOWN * 1.5, RIGHT * 2.2 + DOWN * 1.5, buff=0,
                     stroke_width=1, color=C["dim"], max_tip_length_to_length_ratio=0.03)
        self.add(arr2)
        self.add(Text("time →", font_size=12, color=C["dim"], font="Monospace"
                       ).next_to(arr2, DOWN, buff=0.03).align_to(arr2, RIGHT))

        # 8-wide: GPU style (1 column of 8)
        gpu_label = Text("8-wide (GPU)", font_size=16, color=C["green"],
                         font="Monospace")
        gpu_label.move_to(LEFT * 5.2 + DOWN * 2.2)
        gpu_sub = Text("1 warp", font_size=12, color=C["fg3"], font="Monospace")
        gpu_sub.next_to(gpu_label, DOWN, buff=0.08)
        self.add(gpu_label, gpu_sub)

        gpu_x = -3.0
        for lane in range(8):
            r = Rectangle(width=0.5, height=0.35, fill_color=task_colors[lane],
                           fill_opacity=0.7, stroke_color=task_colors[lane], stroke_width=1)
            r.move_to(RIGHT * gpu_x + DOWN * (2.2 + lane * 0.38 - 1.3))
            t = Text(f"T{lane}", font_size=12, color=C["fg"], font="Monospace").move_to(r)
            self.add(r, t)

        arr3 = Arrow(LEFT * 3.2 + DOWN * 3.4, RIGHT * 2.2 + DOWN * 3.4, buff=0,
                     stroke_width=1, color=C["dim"], max_tip_length_to_length_ratio=0.03)
        self.add(arr3)
        self.add(Text("time →", font_size=12, color=C["dim"], font="Monospace"
                       ).next_to(arr3, DOWN, buff=0.03).align_to(arr3, RIGHT))

        # Key takeaway
        box = Rectangle(width=8, height=0.6, fill_color=C["fill"],
                        fill_opacity=0.8, stroke_color=C["green"], stroke_width=1)
        box.move_to(RIGHT * 1.5 + DOWN * 3.4)
        msg = Text("Same 8 tasks. The hardware decides what runs simultaneously.",
                    font_size=12, color=C["fg"], font="Monospace")
        msg.move_to(box)
        self.add(box, msg)
