from typing import List
from . import Particle

__all__ = ['Method']

class Method():
    def __init__(self, particles: List[Particle]):
        self.particles: List[Particle] = particles

    def do_method(self):
        pass