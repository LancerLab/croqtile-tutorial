"""Publication diagram: mma syntax."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import mma_syntax


class MMASyntax(Scene):
    def construct(self):
        mma_syntax(self)
