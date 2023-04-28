from typing import List

import math
import numpy as np

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
    terms : int
        Number of terms in the 'expansion of the multipole'. If `0` then uses
        CoM method instead.
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
                 theta: float, terms: int, n_crit: int, max_level: int) -> None:
        super().__init__(centre, size, None, terms, n_crit)
        
        self.particles: List[Particle] = particles
        self.n_particles: int = len(particles)
        self.theta: float = theta
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

    def populate_multipoles(self) -> None:
        """Calculate the multipole for every cell in the tree. Using direct
        calculation for leaf cells, and M2M for branches.
        """
        for cell in reversed(self.cells):
            if cell.bit_children == 0: # leaf
                cell._calculate_multipole()
            else: # branch
                cell._M2M()


    def evaluate_particle_potentials(self, use_CoM_far: bool = False) -> None:
        """Calculate the value of the potential felt by every particle in the
        tree. Using the Barnes-Hut theta condition to decide if far or near
        field interaction.

        Parameters
        ----------
        far_method : bool, optional
            Controls what method of approximation to use for far-field cell
            interaction. If unspecified (or `False`) with use multipole
            expansion. If `True` will use CoM approximation of the cells.
            Requires the relevant attributes for each cell to have been
            calculated.
        """
        
        def _pairwise(particle: Particle, other: Particle) -> None:
            z0 = particle.centre - other.centre
            particle.potential -= other.charge * math.log(abs(z0))
            # over_r the 1/r term, or force per self*other charge
            over_r = np.array((z0.real, z0.imag)) / abs(z0)**2
            particle.force_per += other.charge \
                                * np.array((z0.real, z0.imag)) / abs(z0)**2

        def _CoM_z0(particle: Particle, cell: Cell) -> complex:
            return particle.centre - cell.CoM
        
        def _multipole_z0(particle: Particle, cell: Cell) -> complex:
            return particle.centre - cell.centre

        def _CoM_far(particle: Particle, cell: Cell, z0: complex) -> None:
            particle.potential -= cell.total_mass * math.log(abs(z0))

            particle.force_per += cell.total_mass \
                                * np.array((z0.real, z0.imag)) / abs(z0)**2
        
        def _multipole_far(particle: Particle, cell: Cell, z0: complex
                           ) -> None:
            k_vals = np.arange(1, self.terms)
            particle.potential -= (cell.multipole[0] * math.log(abs(z0)) \
                            + np.sum(cell.multipole[1:] / z0**k_vals)).real
            
            w_prime = cell.multipole[0] / z0 - np.sum(k_vals * cell.multipole[1:] / z0**(k_vals+1))
            particle.force_per += np.array((w_prime.real, -w_prime.imag))
        
        if use_CoM_far:
            _get_z0 = _CoM_z0
            _far_method = _CoM_far
        else:
            _get_z0 = _multipole_z0
            _far_method = _multipole_far

        def _interact_with_cell(particle: Particle, cell: Cell) -> None:
            """Uses BH algorithm and theta condition to decide if approximate
            force from a cell, or calculate directly pairwise."""

            # don't interact with cells particle is in, unless leaf
            if particle in cell.particles:
                if cell.bit_children: # can go deeper
                    for child in cell.children:
                        if child:
                            _interact_with_cell(particle, child)
                # if leaf cell, need to pairwise interact with all but self
                elif cell.n_particles > 1: # not just the particle in the leaf
                    for other in cell.particles:
                        if other != particle:
                            _pairwise(particle, other)
                return # if a leaf cell with only particle

            # check theta to see if should go deeper
            z0 = _get_z0(particle, cell) # relevant z0, for cell centre or CoM
            if cell.size < self.theta * abs(z0): # far cell
                # using relevant method, multipole or CoM
                _far_method(particle, cell, z0)
                return
            
            # near cell, go deeper if children
            if cell.bit_children:
                for child in cell.children:
                    if child:
                        _interact_with_cell(particle, child)
            # no children then need to pairwise interact
            else:
                for other in cell.particles:
                    _pairwise(particle, other)

        for particle in self.particles:
            # use the BH algorithm to decide which cells to interact with
            _interact_with_cell(particle, self)
