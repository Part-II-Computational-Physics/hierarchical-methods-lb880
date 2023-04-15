from typing import List

import numpy as np
import matplotlib.pyplot as plt

from . import cells
from ..general import Particle

__all__ = ['BH']

class BH():
    def __init__(self, particles: List[Particle], max_level: int, theta: float, n_crit:int = 2) -> None:
        self.particles: List[Particle] = particles
        self.max_level: int = max_level
        self.theta: float = theta
        self.n_crit: int = n_crit

    def create_root(self) -> None:
        self.root = cells.RootCell(0.5+0.5j, 1, self.particles, self.max_level, self.theta, self.n_crit)

    def do_bh(self, zero_potentials: bool = False):
        if zero_potentials:
            for particle in self.particles:
                particle.potential = 0.0
        
        self.create_root()
        self.root.create_tree()
        self.root.populate_mass_CoM()
        self.root.evaluate_particle_potentials()

    def plot(self):
        fig,ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)

        points = [source.centre for source in self.particles]
        X,Y = np.real(points), np.imag(points)

        ax.scatter(X,Y)

        import matplotlib.patches as patches

        def draw_rectangles(cell: cells.Cell):
            corner = cell.centre - cell.size*(0.5+0.5j)
            p = patches.Rectangle((corner.real,corner.imag),cell.size,cell.size, fill=False, color='red')
            ax.add_patch(p)
            if cell.bit_children == 0:
                return
            else:
                for child in cell.children:
                    if child:
                        draw_rectangles(child)

        draw_rectangles(self.root)

        plt.show()
    
