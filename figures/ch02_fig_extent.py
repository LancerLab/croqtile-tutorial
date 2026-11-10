"""Publication diagram: extent."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import extent


class ExtentOperator(Scene):
    def construct(self):
        extent(self)
