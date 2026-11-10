"""Publication diagram: scalar matmul."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import scalar_matmul


class ScalarMatmul(Scene):
    def construct(self):
        scalar_matmul(self)
