"""Publication diagram: zfill."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import zfill


class ZFill(Scene):
    def construct(self):
        zfill(self)
