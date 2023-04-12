import math

from typing import List

from ...general import Particle
from .cell import Cell

class RootCell(Cell):
    """Wrapper for total tree operations
    
    Attributes
    ----------
    max_level : int
        the max depth the tree is allowed to go to
    cells : List[Cell]
        list of all cells in the tree, in order they were created
    level_matricies : List[Array]
        matrix for each level containing reference (as to cells) of the index
        of each of the cells on that level
    """

    def __init__(self,
                 centre:complex,
                 size:float,
                 max_level:int,
                 theta:float) -> None:
        
        super().__init__(centre, size)

        self.max_level:int = max_level
        self.theta: float = theta
        self.cells:List[Cell] = [self]

    
    def populate_with_particles(self, particles:List[Particle], n_crit:int=2):
        self.particles = particles
        self.n_particles = len(particles)
        
        if self.n_particles >= n_crit:
            self._split_cell(n_crit, self.max_level, self.cells)

        # for particle in particles:
        #     self._add_particle(particle,
        #                       n_crit,
        #                       self.max_level,
        #                       self.cells)

    def populate_mass_CoM(self):
        # iterate from leaf nodes first
        for cell in reversed(self.cells):
            cell._get_Mass_and_CoM()
    
    def print_tree_CoMs(self):
        def _print_CoM(cell:Cell, level:int):
            print('\t'*level, cell.total_mass, cell.CoM, cell)
            for child in cell.children:
                if child:
                    _print_CoM(child, level+1)
        
        _print_CoM(self, 0)

    def calculate_particle_potentials(self):

        def _interact_with_cell(particle:Particle, cell:Cell):
            
            # don't interact with cells particle is in, unless they are leaf
            if particle in cell.particles:
                if cell.bit_children:
                    # look through the children
                    for child in cell.children:
                        if child:
                            _interact_with_cell(particle, child)
                # if leaf cell, then need to pairwise interact with all but self
                elif cell.n_particles > 1: # not just the particle
                    for other in cell.particles:
                        if other != particle:
                            particle.potential -= other.charge * math.log(abs(particle.centre - other.centre))
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
                    particle.potential -= other.charge * math.log(abs(particle.centre - other.centre))

        # consider each particle in the system
        for particle in self.particles:
            # start from the top of the tree and then work recursively down
            # compare to theta at each point
            _interact_with_cell(particle, self)