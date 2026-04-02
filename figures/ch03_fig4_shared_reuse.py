"""
Figure 4: Shared Memory Reuse — why => shared matters.
Left: without shared (each thread loads its own copy from global).
Right: with shared (one DMA fills shared, all threads read from it).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class SharedReuse(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("Data Reuse: local vs shared", font_size=24,
                      color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.3)
        self.add(title)

        sep = DashedLine(UP * 2.5, DOWN * 3.0, color=C["dim"], dash_length=0.1)
        self.add(sep)

        # --- Left: => local (no reuse) ---
        lt = Text("=> local (no reuse)", font_size=16, color=C["local_c"],
                   font="Monospace")
        lt.move_to(LEFT * 3.5 + UP * 2.1)
        self.add(lt)

        lx = LEFT * 3.5

        glob_l = Rectangle(width=4.5, height=0.7, fill_color=C["global_c"],
                            fill_opacity=0.15, stroke_color=C["global_c"], stroke_width=1.5)
        glob_l.move_to(lx + UP * 1.2)
        gl_t = Text("Global Memory (tile A)", font_size=12, color=C["global_c"],
                     font="Monospace").move_to(glob_l)
        self.add(glob_l, gl_t)

        # 4 threads each with own local copy
        for i in range(4):
            tx = lx + LEFT * 1.5 + RIGHT * i * 1.1
            # local box
            lb = Rectangle(width=0.9, height=0.5, fill_color=C["local_c"],
                           fill_opacity=0.2, stroke_color=C["local_c"], stroke_width=1)
            lb.move_to(tx + DOWN * 0.2)
            ll = Text(f"local_{i}", font_size=12, color=C["local_c"],
                       font="Monospace").move_to(lb)

            # thread circle
            tc = Circle(radius=0.2, fill_color=C["green"], fill_opacity=0.6,
                        stroke_color=C["fg"], stroke_width=1)
            tc.move_to(tx + DOWN * 1.2)
            tt = Text(f"t{i}", font_size=12, color=C["fg"], font="Monospace").move_to(tc)

            # arrow global -> local
            a1 = Arrow(glob_l.get_bottom() + RIGHT * (i - 1.5) * 0.8,
                       lb.get_top(), buff=0.05, stroke_width=1.2,
                       color=C["global_c"],
                       max_tip_length_to_length_ratio=0.12)
            # arrow local -> thread
            a2 = Arrow(lb.get_bottom(), tc.get_top(), buff=0.05,
                       stroke_width=1.2, color=C["local_c"],
                       max_tip_length_to_length_ratio=0.12)

            self.add(lb, ll, tc, tt, a1, a2)

        cost_l = Text("4 copies of the same tile\n4× bandwidth", font_size=12,
                       color=C["red"], font="Monospace")
        cost_l.move_to(lx + DOWN * 2.2)
        self.add(cost_l)

        # --- Right: => shared (reuse) ---
        rt_title = Text("=> shared (reuse)", font_size=16, color=C["shared_c"],
                         font="Monospace")
        rt_title.move_to(RIGHT * 3.5 + UP * 2.1)
        self.add(rt_title)

        rx = RIGHT * 3.5

        glob_r = Rectangle(width=4.5, height=0.7, fill_color=C["global_c"],
                            fill_opacity=0.15, stroke_color=C["global_c"], stroke_width=1.5)
        glob_r.move_to(rx + UP * 1.2)
        gr_t = Text("Global Memory (tile A)", font_size=12, color=C["global_c"],
                     font="Monospace").move_to(glob_r)
        self.add(glob_r, gr_t)

        # one shared box
        sb = Rectangle(width=3.5, height=0.6, fill_color=C["shared_c"],
                        fill_opacity=0.2, stroke_color=C["shared_c"], stroke_width=2)
        sb.move_to(rx + DOWN * 0.1)
        sl = Text("Shared Memory (1 copy)", font_size=12, color=C["shared_c"],
                   font="Monospace").move_to(sb)
        self.add(sb, sl)

        # one DMA arrow
        dma = Arrow(glob_r.get_bottom(), sb.get_top(), buff=0.05,
                    stroke_width=3, color=C["arrow"],
                    max_tip_length_to_length_ratio=0.08)
        dma_lbl = Text("1× dma.copy", font_size=12, color=C["arrow"],
                        font="Monospace")
        dma_lbl.next_to(dma, RIGHT, buff=0.1)
        self.add(dma, dma_lbl)

        # 4 threads reading from shared
        for i in range(4):
            tx = rx + LEFT * 1.3 + RIGHT * i * 0.9
            tc = Circle(radius=0.2, fill_color=C["green"], fill_opacity=0.6,
                        stroke_color=C["fg"], stroke_width=1)
            tc.move_to(tx + DOWN * 1.2)
            tt = Text(f"t{i}", font_size=12, color=C["fg"], font="Monospace").move_to(tc)

            a = Arrow(sb.get_bottom() + RIGHT * (i - 1.5) * 0.6,
                      tc.get_top(), buff=0.05, stroke_width=1.2,
                      color=C["shared_c"],
                      max_tip_length_to_length_ratio=0.12)
            self.add(tc, tt, a)

        cost_r = Text("1 copy, all threads read it\n1× bandwidth", font_size=12,
                       color=C["green"], font="Monospace")
        cost_r.move_to(rx + DOWN * 2.2)
        self.add(cost_r)
