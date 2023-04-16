from typing import List

import math

from .cell import Cell
from ...general import Particle

__all__ = ['RootCell']

class RootCell(Cell):
    """Class for the root cell of the Barnes-Hut tree. Methods used to act on
    the whole tree.

    Inherits from `Cell`.

    Parameters
    ----------
    centre : complex
        Complex coordinates of the centre of the cell.
    size : float
        Size of the side of the box.
    particles : List[Particle]
        The list of `Particle` objects to add into the tree.
    theta : float
        Value of theta to use in the Barnes-Hut algorithm.
        Lower value has greater accuracy at compute time cost.
        `theta = 0.5` 'typical.
    n_crit : int
        Number of particles in a cell to split at.
        Default value of `2` will leave one particle per leaf cell (as each
        cell splits when it has 2 particles in it).
    max_level : int
        The maximum depth the tree is allowed to recurse to.
    
    Attributes (Additional)
    ----------
    theta : float
        Value of theta to use in the Barnes-Hut algorithm.
        Lower value has greater accuracy at compute time cost.
        `theta = 0.5` 'typical'.
    max_level : int
        The maximum depth the tree is allowed to recurse to.
    cells : List[Cell]
        List of all cells in the tree, in order they were created
    """

    def __init__(self, centre: complex, size: float, particles: List[Particle],
                 theta: float, n_crit: int, max_level: int) -> None:
        super().__init__(centre, size, parent=None)
        
        self.particles: List[Particle] = particles
        self.n_particles: int = len(particles)
        self.theta: float = theta
        self.n_crit: int = n_crit
        self.max_level: int = max_level

        self.cells: List[Cell] = [self]
    
    def create_tree(self) -> None:
        """Create the full Barnes-Hut tree by distributing the particles in the
        root cell.
        """
        
        # split if more particles than n_crit
        if self.n_particles >= self.n_crit:
            self._split_cell(self.max_level, self.cells)

    def populate_mass_CoM(self) -> None:
        """Calculate the total mass and CoM for every cell in the tree."""

        # iterate from leaf nodes first
        for cell in reversed(self.cells):
            cell._get_mass_CoM()
    
    def print_tree_CoMs(self) -> None:
        """Print all the total masses and CoM in the tree."""

        def _print_CoM(cell: Cell, level: int):
            print('\t'*level, cell.total_mass, cell.CoM, cell)
            for child in cell.children:
                if child:
                    _print_CoM(child, level+1)
        
        _print_CoM(self, 0)

    def evaluate_particle_potentials(self) -> None:
        """Calculate the value of the potential felt by every particle in the
        tree. Using the Barnes-Hut theta condition to decide if far or near
        field interaction.
        
        Requires total masses and CoM of all cells to have been calculated.
        """

        def _interact_with_cell(particle: Particle, cell: Cell) -> None:

            # don't interact with cells particle is in, unless they are leaf
            if particle in cell.particles:
                if cell.bit_children:
                    # look through the children
                    for child in cell.children:
                        if child:
                            _interact_with_cell(particle, child)
                # if leaf cell, need to pairwise interact with all but self
                elif cell.n_particles > 1: # not just the particle
                    for other in cell.particles:
                        if other != particle:
                            particle.potential -= other.charge \
                                * math.log(abs(particle.centre - other.centre))
                return

            # check theta to see if should go deeper
            z0_abs = abs(particle.centre - cell.CoM)
            if cell.size < self.theta * z0_abs: # far cell
                # CoM interact
                particle.potential -= cell.total_mass * math.log(z0_abs)
                return
            
            # near cell, go deeper if children
            if cell.bit_children:
                for child in cell.children:
                    if child:
                        _interact_with_cell(particle, child)
            else:
                for other in cell.particles:
                    particle.potential -= other.charge \
                        * math.log(abs(particle.centre - other.centre))

        # consider each particle in the system
        for particle in self.particles:
            # start from the top of the tree and then work recursively down
            # compare to theta at each point
            _interact_with_cell(particle, self)