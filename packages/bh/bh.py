from typing import List

import math
import numpy as np
import matplotlib.pyplot as plt

from . import cells
from ..general import Method, Particle

__all__ = ['BH']

class BH(Method):
    """Class to hold the Barnes-Hut method of N-Body Interaction. Constructs a
    tree for the particles, used to approximate far-field interactions.

    Parameters
    ----------
    particles : List[Particle]
        List of `Particle` object to act on with the method.
    theta : float
        Value of theta to use in the Barnes-Hut algorithm.
        Lower value has greater accuracy at compute time cost.
        `theta = 0.5` 'typical'.
    terms : int, options
        Number of terms in the 'expansion of the multipole'. If default of `1`
        then uses CoM method instead.
    n_crit : int, default 2
        Number of particles in a cell to split at.
        Default value of `2` will leave one particle per leaf cell (as each
        cell splits when it has 2 particles in it).
    max_level : int, optional
        The maximum depth the tree is allowed to recurse to.
        Default value of -1 chooses value of log2(number of particles),
        (twice the depth for 1 particle per cell).

    Attributes
    ----------
    particles : List[Particle]
        List of `Particle` object to act on with the method.
    theta : float
        Value of theta to use in the Barnes-Hut algorithm.
        Lower value has greater accuracy at compute time cost.
    terms : int
        Number of terms in the 'expansion of the multipole'. If `0` then uses
        CoM method instead.
    n_crit : int
        Number of particles in a cell to split at.
        Default value of `2` will leave one particle per leaf cell (as each
        cell splits when it has 2 particles in it).
    max_level : int
        The maximum depth the tree is allowed to recurse to.
        Default value of -1 chooses value of log2(number of particles),
        (twice the depth for 1 particle per cell).
    root : RootCell
        The root of the Barnes-Hut tree.
    """

    def __init__(self, particles: List[Particle], theta: float, terms: int = 0,
                 n_crit: int = 2, max_level: int = -1) -> None:
        super().__init__(particles)
        self.theta: float = theta
        self.n_crit: int = n_crit
        self.terms: int = terms
        if max_level > -1:
            self.max_level: int = max_level
        else:
            self.max_level: int = int(math.log2(len(particles)))

        self.create_root()

    def create_root(self) -> None:
        """Create root cell as `RootCell` object."""
        self.root = cells.RootCell(0.5+0.5j, 1, self.particles, self.theta,
                                   self.terms, self.n_crit, self.max_level)

    def do_method(self, zero_potentials: bool = True, zero_forces: bool = True):
        """Perform the full Barnes-Hut algorithm. Constructing the tree and
        evaluating the particles.

        Parameters
        ----------
        zero_potentials : bool, default True
            Controls if particle potentials are reset to zero in the process.
            Default of True resets.
        zero_forces : bool, default True
            Controls if particle forces are reset to zero in the process.
            Default of True resets.
        """

        if zero_potentials:
            for particle in self.particles:
                particle.potential = 0.0
        if zero_forces:
            for particle in self.particles:
                particle.force_per = np.zeros(2, dtype=float)
        
        self.create_root()
        self.root.create_tree()
        if self.terms > 0:
            self.root.populate_multipoles()
            self.root.evaluate_particle_potentials()
        else:
            self.root.populate_mass_CoM()
            self.root.evaluate_particle_potentials(use_CoM_far=True)

    def plot(self, fmm_grid: bool = False):
        """Plot the particles positions, with the particle positions overlain.

        Parameters
        ----------
        fmm_grid : bool, defaul False
            Uses the minor axis to plot the (assumed) finest grid used for the
            FMM grid. Assumed as based on a max_level of half the BH max level.
        """

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)

        points = [particle.centre for particle in self.particles]
        X,Y = np.real(points), np.imag(points)

        ax.scatter(X,Y)

        import matplotlib.patches as patches

        def draw_rectangles(cell: cells.Cell):
            corner = cell.centre - cell.size*(0.5+0.5j)
            p = patches.Rectangle((corner.real, corner.imag),
                                  cell.size, cell.size,
                                  fill=False, color='red')
            ax.add_patch(p)
            if cell.bit_children == 0:
                return
            else:
                for child in cell.children:
                    if child:
                        draw_rectangles(child)

        draw_rectangles(self.root)

        if fmm_grid:
            first_val = 1 / (2**(int(self.max_level/2)))
            # stop of 1 the bounds of the grid, but will never appear
            ticks = np.arange(first_val, 1, first_val)
            ax.set_xticks(ticks, minor=True)
            ax.set_yticks(ticks, minor=True)
            ax.grid(True, 'minor')

        plt.show()
