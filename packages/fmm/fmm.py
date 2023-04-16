from typing import List

import math
import numpy as np

from ..general import Particle

from .level import Level, FinestLevel

__all__ = ['FMM']

class FMM():
    """Class to hold the Fast Multipole Method for N-Body Interaction.
    Calculates multipole expansions due the particles in cells, to then
    calculate local expansions for every cell due to the far-field interaction
    of particles. This can then be used to evaluate the action due to those far
    fild particles.

    Parameters
    ----------
    particles : List[Particle]
        List of `Particle` object to act on with the method.
    terms : int
        The numbers of terms to be used in the FMM algorithm.
        Indexing these terms from zero means having terms `0` to `p`,
        where `p = terms - 1` is the precision value as in FMM paper.
    max_level : int, optional
        The maximum depth of arrays to produce in the method.
        Default value of -1 chooses value of log2(number of particles),
        (approx one particle per cell).

    Attributes
    ----------
    particles : List[Particle]
        List of `Particle` object to act on with the method.
    terms : int
        The precision to be used in the FMM algorithm.
    max_level : int
        The maximum depth of arrays to produce in the method.
        Default value of -1 chooses value of log2(number of particles),
        (approx one particle per cell).
    levels : List[Level]
        Collection of the progressively finer level arrays, wrapped as `Level`
        objects.
    finest_level : FinestLevel
        The finest grained level used in the method.
        Equivalent to `levels[-1]`.
    """

    def __init__(self, particles: List[Particle], terms: int,
                 max_level: int = -1) -> None:
        self.particles: List[Particle] = particles
        self.terms: int = terms
        if max_level > -1:
            self.max_level: int = max_level
        else:
            self.max_level: int = int(0.5 * math.log2(len(particles)))

        self.levels: List[Level] \
            = [Level(lvl, terms) for lvl in range(self.max_level)]
        self.finest_level: FinestLevel = FinestLevel(self.max_level, terms)
        self.levels.append(self.finest_level)

    def upward_pass(self) -> None:
        """Perform the upward pass on the array structre. 
        Starting from the second finest grained array perform M2M to propogate
        the multipoles up the structure. Does not perform on finest grained,
        as this is where the multipoles should have been calculated for.
        """

        # Moves backwards through levels, from 2nd to last level
        for level_num in range(self.max_level-1, -1, -1):
            self.levels[level_num].M2M(self.levels[level_num+1])

    def downward_pass(self) -> None:
        """Perform downward pass on the expansion arrays to calculate all of
        the local expansions for each cell.
        M2L is used to get local contributions for each cell from its
        interaction list, and the L2L used to shift the local expansion to the
        children.

        L2L not done from final level as no children.
        M2L and L2L not done on/from coarsest two levels for open boundary
        conditions, as there is no interaction list for either.
        """

        # first two levels have zero local, as no interaction list
        for i, level in enumerate(self.levels[2:-1], start=2):
            # interaction list contributions
            level.M2L()
            # distribute to children
            level.L2L(self.levels[i+1])

        # no L2L for finest level, no children
        self.finest_level.M2L()

    def do_fmm(self, zero_expansions: bool = False,
               zero_potentials: bool = False, zero_forces: bool = False
               ) -> None:
        """Updates particle potentials using the full FMM method, with the
        given parameters
        
        Parameters
        ----------
        zero_potentials : bool
            Controls if particle potentials are explicitly reset to zero in the
            process. Default of False (does not change the potentials)
        zero_forces : bool
            Controls if particle forces are explicitly reset to zero in the
            process. Default of False (does not change the potentials)
        """
        if zero_expansions:
            for level in self.levels:
                level.zero_expansions()
        if zero_potentials:
            for particle in self.particles:
                particle.potential = 0.0
        if zero_forces:
            for particle in self.particles:
                particle.force = np.zeros(2, dtype=float)

        self.finest_level.populate_with_particles(self.particles)
        self.upward_pass()
        self.downward_pass()
        self.finest_level.evaluate_particles()
