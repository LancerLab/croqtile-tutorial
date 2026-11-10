"""Publication diagram: span."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import span


class SpanDimension(Scene):
    def construct(self):
        span(self)
