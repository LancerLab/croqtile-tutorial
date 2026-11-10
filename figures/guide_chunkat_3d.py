"""Figure 10: reproduce the [6, 17, 128] chunkat example with Manim.

Each of the six planes contains 17 rows of four chunks. A chunk spans
32 contiguous elements of the last dimension. The selected chunk is
(p, index) = (1, {16, 2}), with element origin (1, 16, 64).
"""
from manim import *
from guide_diagrams import C, heading, foot, label


class ChunkatThreeDimensions(Scene):
    def construct(self):
        heading(self, 'chunkat: from a tensor to one three-dimensional chunk',
                'lhs [6, 17, 128]  |  parallel p by 6  |  with index in [17, 4]')
        self.add(label('Six outer slices', -4.65, 1.95, 22, weight=SEMIBOLD),
                 label('Slice p = 1', -0.65, 1.95, 22, C['blue'], weight=SEMIBOLD),
                 label('Selected chunk', 3.85, 1.95, 22, C['orange'], weight=SEMIBOLD))

        # Six complete planes, with the original example's second plane selected.
        for p in range(6):
            x = -5.4 + (p % 3) * 0.84
            y = 0.85 - (p // 3) * 1.93
            color = C['blue'] if p == 1 else C['stroke']
            plane = Rectangle(width=0.65, height=1.47, stroke_color=color,
                              stroke_width=1.8 if p == 1 else 1.0,
                              fill_color=C['blue'] if p == 1 else C['fill'],
                              fill_opacity=0.10 if p == 1 else 0.6).move_to([x, y, 0])
            self.add(plane)
            for row in range(1, 17):
                yy = y + 1.47/2 - row*1.47/17
                self.add(Line([x-.325, yy, 0], [x+.325, yy, 0],
                              stroke_width=0.6, color=color))
            for col in range(1, 4):
                xx = x-.325+col*.65/4
                self.add(Line([xx, y-.735, 0], [xx, y+.735, 0],
                              stroke_width=0.6, color=color))
            self.add(label(f'p = {p}', x, y-.94, 16,
                           C['blue'] if p == 1 else C['fg2']))
        self.add(Arrow([-3.25, 0.1, 0], [-2.05, 0.1, 0], color=C['blue'],
                       stroke_width=2, buff=.12, tip_length=.13))

        # Preserve all 17 × 4 chunk positions rather than eliding the selected row.
        x0, top, cw, ch = -1.65, 1.15, .5, .18
        for col in range(4):
            self.add(label(str(col), x0+(col+.5)*cw, 1.45, 16, C['fg2']))
        for row in range(17):
            for col in range(4):
                selected = (row, col) == (16, 2)
                self.add(Rectangle(width=cw-.02, height=ch-.012,
                                   stroke_color=C['orange'] if selected else C['stroke'],
                                   stroke_width=1.2 if selected else .75,
                                   fill_color=C['orange'] if selected else C['fill'],
                                   fill_opacity=.35 if selected else .7)
                         .move_to([x0+(col+.5)*cw, top-(row+.5)*ch, 0]))
            if row in (0, 4, 8, 12, 16):
                self.add(label(str(row), x0-.3, top-(row+.5)*ch, 15,
                               C['orange'] if row == 16 else C['fg2']))
        self.add(label('17 rows × 4 chunks', -.65, -2.25, 19),
                 label('index = {16, 2}', -.65, -2.65, 19, C['orange'], mono=True))

        # A shallow prism makes the 1 × 1 × 32 extent explicit.
        a = np.array([2.25, .7, 0]); b = np.array([5.45, .7, 0])
        depth = np.array([-.23, .28, 0]); down = np.array([0, -.30, 0])
        self.add(Polygon(a, b, b+depth, a+depth, color=C['orange'],
                         stroke_width=1.2, fill_opacity=.09),
                 Polygon(b, b+depth, b+depth+down, b+down, color=C['orange'],
                         stroke_width=1.2, fill_opacity=.18),
                 Polygon(a, b, b+down, a+down, color=C['orange'],
                         stroke_width=1.2, fill_opacity=.20))
        for i in range(1, 32):
            pos = a+(b-a)*i/32
            self.add(Line(pos, pos+down, color=C['orange'], stroke_width=.6))
        self.add(label('32 contiguous elements', 3.85, 1.3, 19, C['orange']),
                 label('Chunk shape: [1, 1, 32]', 3.85, -.15, 20),
                 label('Chunk index: (1, 16, 2)', 3.85, -.75, 19),
                 label('Element origin: (1, 16, 64)', 3.85, -1.35, 19),
                 label('Last dimension: 64–95', 3.85, -1.95, 19, C['orange']))
        self.add(Arrow([.55, -1.82, 0], [2.05, .50, 0], color=C['orange'],
                       stroke_width=2, buff=.1, tip_length=.14))
        foot(self, '[6, 17, 128] / [6, 17, 4] = [1, 1, 32]  |  Bounds determine shape; current indices select the chunk.')
