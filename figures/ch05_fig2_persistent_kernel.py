"""Publication diagram: persistent."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import persistent


class Ch05Fig2PersistentKernel(Scene):
    def construct(self):
        persistent(self)
