"""Publication diagram: chunkat."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import chunkat


class ChunkatSemantics(Scene):
    def construct(self):
        chunkat(self)
