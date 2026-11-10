"""Publication diagram: event credit."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from manim import Scene
from guide_diagrams import event_credit


class EventCreditFlow(Scene):
    def construct(self):
        event_credit(self)
