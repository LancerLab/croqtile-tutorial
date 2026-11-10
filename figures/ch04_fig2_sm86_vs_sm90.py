"""Publication diagram: group scope."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import group_scope


class SM86vsSM90(Scene):
    def construct(self):
        group_scope(self)
