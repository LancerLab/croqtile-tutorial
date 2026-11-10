"""Publication diagram: logical physical."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import logical_physical


class LogicalVsPhysical(Scene):
    def construct(self):
        logical_physical(self)
