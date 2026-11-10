"""Publication diagram: span as."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import span_as


class SpanAs(Scene):
    def construct(self):
        span_as(self)
