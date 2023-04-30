from typing import List, Set

import numpy as np
from scipy.special import binom

from ..general import Particle

from .cell import Cell

__all__ = ['Level', 'FinestLevel']

class Level():
    """Object to describe the array of a level in the FMM algorithm.

    Parameters
    ----------
    level_num : int
        The index of the given level, starting from 0.
    terms : int
        The numbers of terms to be used in the FMM algorithm.
        Indexing these terms from zero means having terms `0` to `p`,
        where `p = terms - 1` is the precision value as in FMM paper.

    Attributes
    ----------
    level_num : int
        The index of the given level, starting from 0.
    terms : int
        The numbers of terms to be used in the FMM algorithm.
        Indexing these terms from zero means having terms `0` to `p`,
        where `p = terms - 1` is the precision value as in FMM paper.
    axis_amount : int
        Number of cells across one axis in the level.
        Hence there are `axis_amount` by `axis_amount` cells in the level.
    array : 3DArray
        The array of the level to store the expansion coefficeints.
        First term is the complex value of the centre of the cell.
        Next comes the multipole expansion, then the local expansion.
    """

    def __init__(self, level_num: int, terms: int) -> None:
        self.level_num: int = level_num
        self.terms: int = terms

        self.axis_amount: int = 2**level_num

        # expansion array
        self.array = np.zeros((self.axis_amount, self.axis_amount,
                               2*terms+1), dtype=complex)
        first_val = 1 / (2**(level_num+1))
        # stop of 1 as this is the so called max value, but will never appear
        vals = np.arange(first_val, 1, 2*first_val)
        X, Y = np.meshgrid(vals, vals, indexing='ij')
        self.array[:,:,0] = X + 1j*Y

        # pre-made arrays for M2M, etc
        # M2L
        self._minus_and_plus = np.ones(self.terms-1)
        self._minus_and_plus[::2] = -1
        self._k_M2L = np.arange(1, self.terms)
        self._l_M2L = np.arange(1, self.terms)
    
    def zero_expansions(self) -> None:
        """Zero all terms in the expansion arrays."""
        self.array[:,:,1:] = 0

    def _cell_M2M(self, cell: Cell, child_level: 'Level') -> None:
        """Use the M2M operation to generate the multipole of a cell in the
        level matrix due to its children.
        
        cell : Cell
            The cell to perform the M2M on.
        child_level: Level
            The child level to the current level being acted on. To get the
            child cell multipoles from.
        """
        for child in cell.children():
            child_multipole = \
                child_level.array[child.index][1:self.terms+1]

            self.array[cell.index][1] += child_multipole[0]

            z0 = child_level.array[child.index][0] - self.array[cell.index][0]

            for l in range(1,self.terms):
                k_vals = np.arange(1,l+1)
                self.array[cell.index][l+1] \
                    += -(child_multipole[0] * z0**l / l) \
                        + np.sum(child_multipole[1:l+1] * z0**(l-k_vals) \
                                 * binom(l-1, k_vals-1))

    def M2M(self, child_level: 'Level') -> None:
        """Perform M2M to calculate multipoles of a given level
        due to the multipoles in the child level
        
        Parameters
        ----------
        child_level: Level
            The child level to the current level being acted on. To get the
            child cell multipoles from.
        """

        for x in range(self.axis_amount):
            for y in range(self.axis_amount):
                self._cell_M2M(Cell(x, y, self.level_num), child_level)

    def _cell_M2L(self, cell: Cell, interactor: Cell) -> None:
        """Use the M2L operation to generate the local expansion for the cell,
        due to the multipole of the interacting cell.
        
        Parameters
        ----------
        cell : Cell
            The cell location for which to generate the local for.
        Interactor : Cell
            The cell location of the multipole to generate the local expansion
            from.
        """

        # local expansion 'about origin' (so about cell)
        z0 = self.array[interactor.index][0] - self.array[cell.index][0]

        interactor_multipole = self.array[interactor.index][1:self.terms+1]
        
        minus_bk_over_z0k = self._minus_and_plus * interactor_multipole[1:] \
                            / z0**self._k_M2L
        
        self.array[cell.index][self.terms+1] += interactor_multipole[0] \
                                                    * np.log(-z0) \
                                                    + np.sum(minus_bk_over_z0k)

        self.array[cell.index][self.terms+2:] \
            += -interactor_multipole[0] / (self._l_M2L * z0**self._l_M2L) \
                + (1/z0**self._l_M2L) * np.sum(
                        minus_bk_over_z0k * binom(
                                self._l_M2L[:,np.newaxis] + self._k_M2L - 1,
                                self._k_M2L-1
                        ), axis=1)
        
    def M2L(self) -> None:
        """Perform M2L to calculate the local expansions in the level due to
        the multipoles of interactor cells (those in a cell's interaction list)
        on the level.
        """

        for x in range(self.axis_amount):
            for y in range(self.axis_amount):
                cell = Cell(x, y, self.level_num)
                for interactor in cell.interaction_list():
                    self._cell_M2L(cell, interactor)

    def _cell_L2L(self, cell: Cell, child_level: 'Level') -> None:
        """Use the L2L operation to shift the local expansion of a cell to its
        children cells.

        Parameters
        ----------
        cell : Cell
            The cell for which to distribute the local expansion to its
            children from.
        child_level : Level
            `Level` object that contains the child of the cell.
        """
        for child in cell.children():
            z0 = child_level.array[child.index][0] - self.array[cell.index][0]

            for l in range(self.terms):
                for k in range(l, self.terms):
                    child_level.array[child.index][self.terms+1+l] \
                        += self.array[cell.index][self.terms+1+k] \
                            * binom(k,l) * z0**(k-l)

    def L2L(self, child_level: 'Level') -> None:
        """Perform L2L to distribute local expansions of a given level to
        children in the children level.

        Parameters
        ----------
        child_level : Level
            `Level` object that contains the children of the cell.
        """

        for x in range(self.axis_amount):
            for y in range(self.axis_amount):
                self._cell_L2L(Cell(x, y, self.level_num), child_level)

    def __repr__(self) -> str:
        return(f'Level: {self.level_num}')


class FinestLevel(Level):
    """Describes the lowest level (finest precision) within the FMM method.
    Inherits from the `Level` class, with additional methods to populate with
    and evaluate partciles within the method.
    
    Parameters
    ----------
    level_num : int
        The index of the given level, starting from 0.
    terms : int
        The numbers of terms to be used in the FMM algorithm.
        Indexing these terms from zero means having terms `0` to `p`,
        where `p = terms - 1` is the precision value as in FMM paper.

    Attributes (Additional)
    ----------
    particle_array : List[List[Set[Particle]]]
        2D Array - like of the sets of particles within each cell at this
        level.
    """

    def __init__(self, level_num: int, terms: int) -> None:
        super().__init__(level_num, terms)
        self.particle_array: List[List[Set[Particle]]] \
            = [[set() for _ in range(self.axis_amount)]
                for _ in range(self.axis_amount)]
        
        self._k_multi = np.arange(1, self.terms)
        self._l_eval = np.arange(self.terms)

    def _calculate_multipole(self, cell: Cell, particle: Particle) -> None:
        """Update the relevant cell's multipole due to the presence of the
        particle.
    
        Parameters
        ----------
        cell : Cell
            Cell the particle exists in on the level, to update the multipole
            of.
        particle : Particle
            Particle whose effect to add.
        """

        # first term
        self.array[cell.index][1] += particle.charge

        # remaining terms
        z0 = particle.centre - self.array[cell.index][0]
        self.array[cell.index][2:self.terms+1] \
                    -= particle.charge * z0**self._k_multi / self._k_multi

    def populate_with_particles(self, particles: List[Particle]) -> None:
        """Place the particles in `particle_array` and calculate their
        contributions to each cell's multipole.

        Parameters
        ----------
        particles : List[Particle]
            List of the particle objects to compute multipoles from.
        """

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
            The cell object to find neighbour particles of.

        Returns
        -------
        neighbour_particles : Set[Particle]
            Set of all Particles in the cell's neighbours.
        """
        
        # near particles are those in the cell and its neighbours
        neighbour_particles: Set[Particle] = set()
        for neighbour in cell.neighbours():
            neighbour_particles.update(
                self.particle_array[neighbour.index[0]][neighbour.index[1]]
            )
        
        return neighbour_particles

    def evaluate_particles(self) -> None:
        """Compute the potentials and force felt by all of the particles in
        the system due to the sum of the effects of far field (local expansion)
        and near field (pairwise) effects.
        """

        for x, y_axis in enumerate(self.particle_array):
            for y, cell_particles in enumerate(y_axis):
                
                # if no particles in that cell, pass the cell
                if not cell_particles:
                    continue
                
                # get particles that can interact with particles in the cell
                near_particles: Set[Particle] = cell_particles.union(
                    self._neighbour_particles(Cell(x, y, self.level_num))
                )

                local = self.array[x, y, self.terms+1:]

                for particle in cell_particles:
                    # far field local expansion contribution
                    z0 = particle.centre - self.array[x,y,0]
                    particle.potential -= np.sum(local * z0**self._l_eval).real

                    w_prime = np.sum(
                        self._l_eval[1:] * local[1:] * z0**self._l_eval[:-1])
                    particle.force_per \
                                += np.array((w_prime.real, -w_prime.imag))

                    # near-field
                    # near particles excluding the own particle
                    for other in near_particles-{particle}:
                        z0 = particle.centre - other.centre
                        particle.potential -= other.charge * np.log(abs(z0))
                        
                        particle.force_per += other.charge \
                            * np.array((z0.real, z0.imag)) / abs(z0)**2

    def __repr__(self) -> str:
        return 'Finest' + super().__repr__()
    