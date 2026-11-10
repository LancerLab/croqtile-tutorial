"""Publication diagram: register layout."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import register_layout


class RegisterLoading(Scene):
    def construct(self):
        register_layout(self)
