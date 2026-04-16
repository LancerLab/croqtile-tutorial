"""
Ch08 Fig1 (combined): .co compilation flow with concrete code regions.
Left: kernel.co with three contiguous colored regions (no borders between them).
Middle: croqtile compiler.
Right: generated intermediate code + nvcc/GPU binary alongside.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from theme import parse_theme
from manim import *

C, THEME = parse_theme()


class Ch08CompilationFlow(Scene):
    def construct(self):
        self.camera.background_color = C["bg"]

        title = Text(
            "How .co files compile",
            font_size=26, color=C["fg"], font="Monospace",
        )
        title.to_edge(UP, buff=0.22)
        self.add(title)

        # ═══ Left: kernel.co source ═══
        lx = -4.5
        src_lbl = Text("kernel.co", font_size=18, color=C["fg"], font="Monospace")
        src_lbl.move_to(RIGHT * lx + UP * 2.3)
        self.add(src_lbl)

        region_specs = [
            (
                [("__device__ float fast_rsqrt(x) {", None),
                 ("  return __frsqrt_rn(x);", None),
                 ("}", None)],
                C["purple"], "\u2460 __device__", 0,
            ),
            (
                [("__co__ void kernel(f32 [M,N] in) {", None),
                 ("  parallel ... : block {", None),
                 ('    __cpp__("asm volatile(...)");', None),
                 ("    dma.copy ...; mma ...", None),
                 ("  }", None),
                 ("}", None)],
                C["green"], "\u2461 __co__ + \u2462 __cpp__", 1,
            ),
            (
                [("int main() {", None),
                 ("  auto buf = make_spandata();", None),
                 ("  kernel(buf.view());", None),
                 ("}", None)],
                C["blue"], "\u2463 host C++", 2,
            ),
        ]

        line_h = 0.26
        code_w = 4.2
        y_cursor = 1.9
        region_boxes = []

        for lines, col, tag, _ in region_specs:
            region_h = len(lines) * line_h + 0.15
            r = Rectangle(
                width=code_w, height=region_h,
                fill_color=col, fill_opacity=0.10,
                stroke_width=0,
            )
            r.move_to(RIGHT * lx + UP * (y_cursor - region_h / 2))
            self.add(r)
            region_boxes.append(r)
            
            if _ == 0:
                tag_t = Text(tag, font_size=12, color=col, font="Monospace")
                tag_t.move_to(r.get_corner(UR) + LEFT * 0.75 + DOWN * 0.70)
                self.add(tag_t)
            elif _ == 1:
                tag_t = Text(tag, font_size=12, color=col, font="Monospace")
                tag_t.move_to(r.get_corner(UR) + LEFT * 1.2 + DOWN * 1.4)
                self.add(tag_t)
            else:
                tag_t = Text(tag, font_size=12, color=col, font="Monospace")
                tag_t.move_to(r.get_corner(UR) + LEFT * 0.75 + DOWN * 1)
                self.add(tag_t)

            for i, (ln, _) in enumerate(lines):
                lead_spaces = len(ln) - len(ln.lstrip(" "))
                content = ln.lstrip(" ")
                t = Text(content, font_size=12, color=C["fg2"], font="Monospace")
                t.move_to(r.get_top() + DOWN * (0.16 + i * line_h))
                # Geometric indent: independent of Text's whitespace behavior.
                indent_step = 0.09
                t.align_to(r.get_left() + RIGHT * (0.12 + lead_spaces * indent_step), LEFT)
                self.add(t)

            y_cursor -= region_h

        # Thin border around entire source file
        total_top = region_boxes[0].get_top()[1]
        total_bot = region_boxes[-1].get_bottom()[1]
        outer = Rectangle(
            width=code_w + 0.06, height=total_top - total_bot + 0.06,
            stroke_color=C["dim"], stroke_width=1.2, fill_opacity=0,
        )
        outer.move_to(RIGHT * lx + UP * ((total_top + total_bot) / 2))
        self.add(outer)

        # ═══ Middle: croqtile compiler ═══
        mid_x = -0.3
        compiler = RoundedRectangle(
            width=2.0, height=0.65, corner_radius=0.12,
            fill_color=C["green"], fill_opacity=0.18,
            stroke_color=C["green"], stroke_width=2.5,
        )
        compiler.move_to(RIGHT * mid_x * 2.2 + UP * 0.12)
        comp_lbl = Text("Croqtile", font_size=16, color=C["green"], font="Monospace")
        comp_sub = Text("compiler", font_size=13, color=C["fg3"], font="Monospace")
        comp_lbl.move_to(compiler.get_center() + UP * 0.1)
        comp_sub.move_to(compiler.get_center() + DOWN * 0.14)
        self.add(compiler, comp_lbl, comp_sub)

        # Arrow: __co__ region -> compiler
        arr_in = Arrow(
            region_boxes[1].get_right(), compiler.get_left(),
            buff=0.08, stroke_width=2.5, color=C["arrow"],
            tip_length=0.12,
        )
        self.add(arr_in)

        # ═══ Right: generated intermediate code ═══
        rx = 3.2

        gen_lbl = Text("generated intermediate", font_size=16, color=C["fg"], font="Monospace")
        gen_lbl.move_to(RIGHT * rx * 0.9 + UP * 2.3)
        self.add(gen_lbl)

        def out_region(label, sub, y, col, h=0.55):
            r = Rectangle(
                width=3.4, height=h,
                fill_color=col, fill_opacity=0.10,
                stroke_width=0,
            )
            r.move_to(RIGHT * rx * 0.9 + UP * y + DOWN * 0.5)
            self.add(r)
            t = Text(label, font_size=13, color=col, font="Monospace")
            t.move_to(r.get_center() + UP * 0.08)
            self.add(t)
            s = Text(sub, font_size=12, color=C["dim"], font="Monospace")
            s.move_to(r.get_center() + DOWN * 0.12)
            self.add(s)
            return r

        o1 = out_region("__device__ fast_rsqrt", "copied verbatim", 1.65, C["purple"])
        o2 = out_region("__global__ __choreo_kernel", "generated from __co__", 0.95, C["green"])
        o3 = out_region('asm volatile("setmaxnreg")', "spliced from __cpp__", 0.35, C["orange"])
        o4 = out_region("int main() { ... }", "host launch wrapper", -0.25, C["blue"])

        # Thin outer border around generated code
        gen_top = o1.get_top()[1]
        gen_bot = o4.get_bottom()[1]
        gen_outer = Rectangle(
            width=3.46, height=gen_top - gen_bot + 0.06,
            stroke_color=C["dim"], stroke_width=1.2, fill_opacity=0,
        )
        gen_outer.move_to(RIGHT * rx * 0.9 + UP * ((gen_top + gen_bot) / 2))
        self.add(gen_outer)

        # Arrows: compiler -> generated outputs
        for o in [o1, o2, o3]:
            a = Arrow(
                compiler.get_right(), o.get_left(),
                buff=0.06, stroke_width=2, color=C["arrow"],
                tip_length=0.12,
            )
            self.add(a)

        # Pass-through arrows: __device__ and host bypassing compiler
        for src, dst in [(region_boxes[0], o1), (region_boxes[2], o4)]:
            a = Arrow(
                src.get_right(), dst.get_left(),
                buff=0.06, stroke_width=1.8, color=C["dim"],
                tip_length=0.2,
            )
            self.add(a)

        # ═══ Far right: nvcc + GPU binary (next to generated code) ═══
        frx = 5.9

        nvcc = RoundedRectangle(
            width=1, height=0.55, corner_radius=0.1,
            fill_color=C["fg3"], fill_opacity=0.15,
            stroke_color=C["fg3"], stroke_width=2,
        )
        nvcc.move_to(RIGHT * frx + UP * 0.4)
        nvcc_lbl = Text("nvcc", font_size=16, color=C["fg"], font="Monospace")
        nvcc_lbl.move_to(nvcc.get_center())
        self.add(nvcc, nvcc_lbl)

        binary = RoundedRectangle(
            width=1.6, height=0.50, corner_radius=0.1,
            fill_color=C["green"], fill_opacity=0.2,
            stroke_color=C["green"], stroke_width=2,
        )
        binary.move_to(RIGHT * frx + DOWN * 0.7)
        bin_lbl = Text("GPU binary", font_size=14, color=C["green"], font="Monospace")
        bin_lbl.move_to(binary.get_center())
        self.add(binary, bin_lbl)

        # Arrow: generated code -> nvcc
        arr_nvcc = Arrow(
            gen_outer.get_right() + UP * 0.2, nvcc.get_left(),
            buff=0.06, stroke_width=2.5, color=C["arrow"],
            tip_length=0.12,
        )
        self.add(arr_nvcc)

        # Arrow: nvcc -> binary
        arr_bin = Arrow(
            nvcc.get_bottom(), binary.get_top(),
            buff=0.04, stroke_width=2.5, color=C["arrow"],
            tip_length=0.12,
        )
        self.add(arr_bin)

        foot = Text(
            "__device__ \u2192 GPU verbatim  |  __co__ \u2192 GPU transformed  |  host \u2192 CPU  |  __cpp__ \u2192 spliced",
            font_size=12, color=C["dim"], font="Monospace",
        )
        foot.to_edge(DOWN, buff=0.18)
        self.add(foot)
