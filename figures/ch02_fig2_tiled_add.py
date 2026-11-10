"""Publication diagram: tiled add."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import tiled_add


class TiledAdd(Scene):
    def construct(self):
        tiled_add(self)
