"""Publication diagram: virtual parallelism."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import virtual_parallelism


class VirtualParallelism(Scene):
    def construct(self):
        virtual_parallelism(self)
