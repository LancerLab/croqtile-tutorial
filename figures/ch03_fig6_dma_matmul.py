"""
Figure 6: DMA Matmul — block grid with K-loop and shared memory.
Shows the outer block grid, the K-loop loading tiles into shared,
and the inner thread grid computing within each block.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class DmaMatmul(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text("DMA Matmul: Block Grid + Shared Memory", font_size=20,
                      color=C["fg"], font="Monospace")
        title.to_edge(UP, buff=0.25)
        self.add(title)

        # --- Left: LHS and RHS matrices ---
        lhs_box = Rectangle(width=1.8, height=2.4, fill_color=C["lhs_c"],
                             fill_opacity=0.15, stroke_color=C["lhs_c"], stroke_width=1.5)
        lhs_box.move_to(LEFT * 5.5 + UP * 0.3)
        lhs_lbl = Text("lhs\n[128,256]", font_size=12, color=C["lhs_c"],
                        font="Monospace").move_to(lhs_box)
        self.add(lhs_box, lhs_lbl)

        rhs_box = Rectangle(width=2.4, height=1.8, fill_color=C["rhs_c"],
                             fill_opacity=0.15, stroke_color=C["rhs_c"], stroke_width=1.5)
        rhs_box.move_to(LEFT * 5.5 + DOWN * 2.2)
        rhs_lbl = Text("rhs [256,256]", font_size=12, color=C["rhs_c"],
                        font="Monospace").move_to(rhs_box)
        self.add(rhs_box, rhs_lbl)

        # --- Center: One block detail ---
        block_box = Rectangle(width=5.5, height=4.5, fill_color=C["orange"],
                               fill_opacity=0.05, stroke_color=C["orange"],
                               stroke_width=2)
        block_box.move_to(RIGHT * 0.5 + DOWN * 0.2)
        block_title = Text("Block (px, py) — 1 of 128 blocks", font_size=12,
                            color=C["orange"], font="Monospace")
        block_title.move_to(block_box.get_top() + DOWN * 0.25)
        self.add(block_box, block_title)

        # Shared memory area
        smem_box = Rectangle(width=4.5, height=1.0, fill_color=C["shared_c"],
                              fill_opacity=0.15, stroke_color=C["shared_c"],
                              stroke_width=1.5)
        smem_box.move_to(RIGHT * 0.5 + UP * 1.0)
        smem_lbl = Text("Shared Memory", font_size=12, color=C["shared_c"],
                         font="Monospace")
        smem_lbl.move_to(smem_box.get_top() + DOWN * 0.2)
        self.add(smem_box, smem_lbl)

        # Two tiles in shared
        lhs_tile = Rectangle(width=1.8, height=0.5, fill_color=C["lhs_c"],
                              fill_opacity=0.3, stroke_color=C["lhs_c"], stroke_width=1)
        lhs_tile.move_to(LEFT * 0.6 + UP * 0.75)
        lt_lbl = Text("lhs_load [16,16]", font_size=12, color=C["lhs_c"],
                       font="Monospace").move_to(lhs_tile)

        rhs_tile = Rectangle(width=1.8, height=0.5, fill_color=C["rhs_c"],
                              fill_opacity=0.3, stroke_color=C["rhs_c"], stroke_width=1)
        rhs_tile.move_to(RIGHT * 1.6 + UP * 0.75)
        rt_lbl = Text("rhs_load [16,16]", font_size=12, color=C["rhs_c"],
                       font="Monospace").move_to(rhs_tile)
        self.add(lhs_tile, lt_lbl, rhs_tile, rt_lbl)

        # DMA arrows into shared
        dma1 = Arrow(lhs_box.get_right(), lhs_tile.get_left(), buff=0.1,
                     stroke_width=2, color=C["arrow"],
                     max_tip_length_to_length_ratio=0.06)
        dma2 = Arrow(rhs_box.get_right() + UP * 0.5, rhs_tile.get_left() + DOWN * 0.1,
                     buff=0.1, stroke_width=2, color=C["arrow"],
                     max_tip_length_to_length_ratio=0.06)
        dma_lbl = Text("dma.copy => shared", font_size=12, color=C["arrow"],
                        font="Monospace")
        dma_lbl.next_to(dma1, UP, buff=0.05)
        self.add(dma1, dma2, dma_lbl)

        # Thread grid inside block
        tg_label = Text("256 threads (16×16) : thread", font_size=12,
                         color=C["green"], font="Monospace")
        tg_label.move_to(RIGHT * 0.5 + DOWN * 0.4)
        self.add(tg_label)

        tg_origin = LEFT * 0.8 + DOWN * 1.0
        for r in range(4):
            for c_ in range(4):
                rect = Rectangle(width=0.45, height=0.35,
                                 fill_color=C["green"], fill_opacity=0.15,
                                 stroke_color=C["green"], stroke_width=0.5)
                rect.move_to(tg_origin + RIGHT * c_ * 0.5 + DOWN * r * 0.4)
                self.add(rect)

        tg_note = Text("(showing 4×4 of 16×16)", font_size=12,
                        color=C["dim"], font="Monospace")
        tg_note.move_to(RIGHT * 2.0 + DOWN * 1.5)
        self.add(tg_note)

        # K-loop annotation
        k_box = Rectangle(width=2.0, height=0.5, fill_color=C["purple"],
                           fill_opacity=0.15, stroke_color=C["purple"], stroke_width=1)
        k_box.move_to(RIGHT * 4.5 + UP * 0.3)
        k_lbl = Text("foreach tile_k\nin [16]", font_size=12, color=C["purple"],
                      font="Monospace").move_to(k_box)
        k_arrow = CurvedArrow(k_box.get_left() + DOWN * 0.1,
                               smem_box.get_right() + DOWN * 0.1,
                               angle=-0.5, color=C["purple"],
                               stroke_width=1.5)
        self.add(k_box, k_lbl, k_arrow)

        # Compute annotation
        compute = Text("output.at(px#qx, py#qy)\n+= lhs_load.data × rhs_load.data",
                        font_size=12, color=C["fg2"], font="Monospace")
        compute.move_to(RIGHT * 0.5 + DOWN * 2.8)
        self.add(compute)
