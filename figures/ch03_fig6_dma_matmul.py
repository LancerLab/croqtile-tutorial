"""Publication diagram: dma matmul."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import dma_matmul


class DmaMatmul(Scene):
    def construct(self):
        dma_matmul(self)
