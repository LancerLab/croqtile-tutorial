"""Publication diagram: element vs block."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import element_vs_block


class ElementVsBlock(Scene):
    def construct(self):
        element_vs_block(self)
