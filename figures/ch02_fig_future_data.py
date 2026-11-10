"""Publication diagram: future data."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import future_data


class FutureData(Scene):
    def construct(self):
        future_data(self)
