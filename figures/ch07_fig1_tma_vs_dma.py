"""Publication diagram: tma vs dma."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import tma_vs_dma


class TmaVsDma(Scene):
    def construct(self):
        tma_vs_dma(self)
