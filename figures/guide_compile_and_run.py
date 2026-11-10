"""Static publication layout of the compile-and-run terminal example."""
from manim import *
from guide_diagrams import C, heading, foot, label


class GuideCompileAndRun(Scene):
    def construct(self):
        heading(self, 'Compile, run, and inspect compiler help',
                'The complete ele_add.co program is compiled and checked from the command line.')
        commands = [
            ('$ ls', 'ele_add.co'),
            ('$ croq ele_add.co -o ele_add',
             'info: transpiling ele_add.co ...\ninfo: compiling CUDA kernel ...\n'
             'info: linking with nvcc ...\ninfo: build successful -> ele_add'),
            ('$ ./ele_add', 'Test Passed'),
        ]
        for x in (-3.2, 3.2):
            self.add(RoundedRectangle(width=5.85, height=4.75, corner_radius=.10,
                                      fill_color=C['fill'], fill_opacity=.6,
                                      stroke_color=C['stroke'], stroke_width=1.2)
                     .move_to([x, -.2, 0]))
        y = 1.7
        for command, output in commands:
            self.add(label(command, -5.8, y, 18, C['blue'], 5.25,
                           mono=True, align='left'))
            y -= .42
            for line in output.splitlines():
                self.add(label(line, -5.8, y, 16,
                               C['green'] if 'successful' in line or line == 'Test Passed' else C['fg2'],
                               5.25, mono=True, align='left'))
                y -= .37
            y -= .20
        help_lines = [
            ('$ croq --help', C['blue']),
            ('Usage: croq [options] <input.co>', C['fg']),
            ('Options:', C['fg']),
            ('-o <file>     Output binary name', C['fg2']),
            ('-es           Emit source only (transpile)', C['fg2']),
            ('-t <target>   Target platform (cuda, ...)', C['fg2']),
            ('-v            Verbose compilation', C['fg2']),
            ('--help        Show this help message', C['fg2']),
        ]
        for i, (line, color) in enumerate(help_lines):
            self.add(label(line, .6, 1.7-i*.5, 18 if i == 0 else 16,
                           color, 5.25, mono=True, align='left'))
        foot(self, 'Test Passed confirms the example’s result check. Compiler help lists the available options.')
