"""Publication diagram: expressiveness."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import expressiveness


class ExpressivenessVsPerformance(Scene):
    def construct(self):
        expressiveness(self)
