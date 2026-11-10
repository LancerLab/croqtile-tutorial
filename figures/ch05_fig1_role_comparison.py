"""Publication diagram: timeline."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import timeline


class RoleComparison(Scene):
    def construct(self):
        timeline(self, roles=True)
