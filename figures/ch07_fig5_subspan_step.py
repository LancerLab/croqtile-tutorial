"""Publication diagram: subspan step."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import subspan_step


class SubspanStep(Scene):
    def construct(self):
        subspan_step(self)
