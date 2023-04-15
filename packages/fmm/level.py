"""Level to be used to contain level operations and expansions for the FMM
method
"""
from typing import List, Set

import numpy as np
from scipy.special import binom

from ..general import Particle

from .cell import Cell

__all__ = ['Level', 'FinestLevel']

class Level():
    def __init__(self, level_num: int, precision: int) -> None:
        self.level_num: int = level_num
        self.precision: int = precision

        axis_amount = 2**level_num
        self.axis_amount: int = axis_amount

        # fill in particle coords
        self.array = np.zeros((axis_amount, axis_amount, 2*precision+1),
                                        dtype=complex)
        first_val = 1 / (2**(level_num+1))
        # stop of 1 as this is the so called max value, but will never appear
        vals = np.arange(first_val, 1, 2*first_val)
        X, Y = np.meshgrid(vals, vals, indexing='ij')
        self.array[:,:,0] = X + 1j*Y

        # pre-made arrays for M2M, etc
        # M2L
        self.minus_and_plus = np.ones(self.precision-1)
        self.minus_and_plus[::2] = -1
        self.k_M2L = np.arange(1, self.precision)
        self.l_M2L = np.arange(1, self.precision)
    
    def zero_expansions(self) -> None:
        self.array[:,:,1:] = 0

    def _cell_M2M(self, cell: Cell, child_level: 'Level') -> None:
        for child in cell.children():
            child_multipole = \
                child_level.array[child.index][1:self.precision+1]

            self.array[cell.index][1] += child_multipole[0]

            z0 = child_level.array[child.index][0] - self.array[cell.index][0]

            for l in range(1,self.precision):
                k_vals = np.arange(1,l+1)
                self.array[cell.index][l+1] += \
                    - (child_multipole[0] * z0**l / l) \
                    + np.sum(child_multipole[1:l+1] * z0**(l-k_vals) \
                                * binom(l-1, k_vals-1))

    def M2M(self, child_level: 'Level') -> None:
        """Perform M2M to calculate multipoles of a given level
        due to the multipoles in the child level
        
        Parameters
        ----------
        child_level : Level
            The `Level` object of the child level to take expansion
            coefficients from
        """

        # ################################################
        # # Can definitely optimise the way this is done #
        # ################################################
        for x in range(self.axis_amount):
            for y in range(self.axis_amount):
                self._cell_M2M(Cell(x, y, self.level_num), child_level)

    def _cell_M2L(self, cell: Cell, interactor: Cell) -> None:
        # local expansion 'about origin' (so about cell)
        z0 = self.array[interactor.index][0] - self.array[cell.index][0]

        interactor_multipole = self.array[interactor.index][1:self.precision+1]
        
        minus_bk_over_z0k = self.minus_and_plus * interactor_multipole[1:] / z0**self.k_M2L
        
        self.array[cell.index][self.precision+1] += interactor_multipole[0] * np.log(-z0) \
                                            + np.sum(minus_bk_over_z0k)

        self.array[cell.index][self.precision+2:] += \
                -interactor_multipole[0] / (self.l_M2L * z0**self.l_M2L) + (1/z0**self.l_M2L) \
                * np.sum(minus_bk_over_z0k * binom(self.l_M2L[:,np.newaxis] + self.k_M2L - 1, self.k_M2L-1),
                        axis=1)
        
    def M2L(self) -> None:
        """Perform M2L to calculate locals of a given level due to the
        multipoles of interactor cells on the level.
        """

        for x in range(self.axis_amount):
            for y in range(self.axis_amount):
                cell = Cell(x, y, self.level_num)
                for interactor in cell.interaction_list():
                    self._cell_M2L(cell, interactor)

    def _cell_L2L(self, cell: Cell, child_level: 'Level') -> None:
        for child in cell.children():
            z0 = child_level.array[child.index][0] - self.array[cell.index][0]

            # ################################################
            # # Can definitely optimise the way this is done #
            # ################################################
            for l in range(self.precision):
                for k in range(l, self.precision):
                    child_level.array[child.index][self.precision+1+l] += \
                        self.array[cell.index][self.precision+1+k] * binom(k,l) * z0**(k-l)

    def L2L(self, child_level: 'Level') -> None:
        """Perform L2L to distribute local expansions of a given level to
        children level.

        Parameters
        ----------
        child_level : Level
            The `Level` object of the child level to take expansion
            coefficients from
        """

        for x in range(self.axis_amount):
            for y in range(self.axis_amount):
                self._cell_L2L(Cell(x, y, self.level_num), child_level)

    def __repr__(self) -> str:
        return(f'Level: {self.level_num}')


class FinestLevel(Level):
    def __init__(self, level_num: int, precision: int) -> None:
        super().__init__(level_num, precision)
        self.particle_array: List[List[Set[Particle]]] = [[set() for _ in range(self.axis_amount)] for _ in range(self.axis_amount)]

    def _calculate_multipole(self, cell: Cell, particle: Particle) -> None:
        """Update the relevant cell  with multipole due to particle
    
        Parameters
        ----------
        cell : Cell
            Cell in own array to add the effect to
        particle : Particle
            Particle whose effect to add
        """

        # first term
        self.array[cell.index][1] += particle.charge

        # remaining terms
        z0 = particle.centre - self.array[cell.index][0]
        k_vals = np.arange(1, self.precision)
        self.array[cell.index][2:self.precision+1] -= particle.charge * z0**k_vals / k_vals

    def populate_with_particles(self, particles: List[Particle]) -> None:
        for particle in particles:
            cell = Cell.particle_cell(particle, self.level_num)
            self._calculate_multipole(cell, particle)
            # add particle to the array
            self.particle_array[cell.index[0]][cell.index[1]].add(particle)

    def _neighbour_particles(self, cell: Cell) -> Set[Particle]:
        """Returns a set of all particles in the nearest neighbours of the
        given cell.
        
        Parameters
        ----------
        cell : Cell
            The cell object to find neighbour particles of

        Returns
        -------
        neighbour_particles : Set[Particle]
            Set of Particles that are in the cell's neighbours
        """
        
        # near particles are those in the cell and its neighbours
        neighbour_particles: Set[Particle] = set()
        for neighbour in cell.neighbours():
            neighbour_particles.update(
                self.particle_array[neighbour.index[0]][neighbour.index[1]]
            )
        
        return neighbour_particles

    def evaluate_particles(self) -> None:
        l_vals = np.arange(self.precision)

        for x, y_axis in enumerate(self.particle_array):
            for y, cell_particles in enumerate(y_axis):
                
                # if no particles in that cell, pass the cell
                if not cell_particles:
                    continue
                
                # get particles that can interact with particles in the cell
                near_particles: Set[Particle] = cell_particles.union(
                    self._neighbour_particles(Cell(x, y, self.level_num))
                )

                local = self.array[x, y, self.precision+1:]

                for particle in cell_particles:
                    # far field local expansion contribution
                    z0 = particle.centre - self.array[x,y,0]
                    particle.potential -= np.sum(local * z0**l_vals).real

                    w_prime = np.sum(l_vals[1:] * local[1:] * z0**l_vals[:-1])
                    particle.force += particle.charge * np.array((w_prime.real, -w_prime.imag))

                    # near-field
                    # near particles excluding the own particle
                    for other in near_particles-{particle}:
                        z0 = particle.centre - other.centre
                        particle.potential -= other.charge * np.log(abs(z0))
                        
                        particle.force += particle.charge * other.charge \
                            * np.array((z0.real, z0.imag)) / abs(z0)**2

    def __repr__(self) -> str:
        return 'Finest' + super().__repr__()
    