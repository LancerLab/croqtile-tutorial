"""Publication diagram: descriptor."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import descriptor


class TMADescriptor(Scene):
    def construct(self):
        descriptor(self)
