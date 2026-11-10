"""Publication diagram: view from."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import view_from


class ViewFrom(Scene):
    def construct(self):
        view_from(self)
