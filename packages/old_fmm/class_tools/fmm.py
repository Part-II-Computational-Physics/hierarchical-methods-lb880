from typing import List

import numpy as np

from ...general import Particle

from .level import Level, FinestLevel

__all__ = ['FMM']

class FMM():
    def __init__(self, precision: int, max_level: int, particles: List[Particle]) -> None:
        self.precision: int = precision
        self.max_level: int = max_level
        self.particles: List[Particle] = particles
        self.levels: List[Level] = [Level(lvl, precision) for lvl in range(max_level)]
        self.finest_level: FinestLevel = FinestLevel(max_level, precision)
        self.levels.append(self.finest_level)

    def upward_pass(self) -> None:
        """Perform the upward pass on the array structre. 
        Starting from the second finest grained array perform M2M to propogate the
        multipoles up the structure. Does not perform on finest grained, as this is
        where the multipoles should have been calculated for.
        """

        # Moves backwards through levels, from 2nd to last level
        for level_num in range(self.max_level-1, -1, -1):
            self.levels[level_num].M2M(self.levels[level_num+1])

    def downward_pass(self) -> None:
        """Perform downward pass on the expansion arrays to calculate all of the
        local expansions for each cell.
        M2L is used to get local contributions for each cell from its interaction
        list, and the L2L used to shift the local expansion to the children.

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

    def do_fmm(self, zero_potentials: bool = False, zero_forces: bool = False) -> None:
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
