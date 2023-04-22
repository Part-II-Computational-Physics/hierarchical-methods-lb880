"""
Pairwise
--------
Methods to calculate direct potentials or forces on a list of `Particle`
objects, through pairwise interaction (Order N squared methods).
"""

from typing import List

import math
import numpy as np

from . import Method, Particle

__all__ = ['Pairwise']

class Pairwise(Method):
    def __init__(self, particles: List[Particle]):
        super().__init__(particles)

    @staticmethod
    def potentials(particles: List[Particle], zero_potentials: bool = True
               ) -> None:
        """Calculate through pairwise interactions the particle potentials and
        store in `Particle` object attribute.
        
        Parameters
        ----------
        particles : List[Particle]
            List of the `Particle` objects to calculate the direct potentials
            for.
        zero_potentials : bool, default True
            Controls if particle potentials are reset to zero in the process.
            Default of `True` resets direct potentials first.
        """

        if zero_potentials:
            for particle in particles:
                particle.potential = 0.0

        for i, particle in enumerate(particles):
            for other in particles[i+1:]:
                potential = - math.log(abs(particle.centre - other.centre))
                particle.potential += other.charge * potential
                other.potential += particle.charge * potential

    @staticmethod
    def forces(particles: List[Particle], zero_forces: bool = True) -> None:
        """Calculate through pairwise interactions the particle potentials and
        store in `Particle` object attribute.
        
        Parameters
        ----------
        particles : List[Particle]
            List of the `Particle` objects to calculate the direct forces for.
        zero_forces : bool, default True
            Controls if particle forces are reset to zero in the process.
            Default of `True` resets direct forces first.
        """

        if zero_forces:
            for particle in particles:
                particle.force_per = np.zeros(2, dtype=float)

        for i, particle in enumerate(particles):
            for other in particles[i+1:]:
                z0 = particle.centre - other.centre
                # over_r the 1/r term, or force per self*other charge
                over_r = np.array((z0.real, z0.imag)) / abs(z0)**2
                particle.force_per += other.charge * over_r
                other.force_per -= particle.charge * over_r
    
    def do_method(self, zero_potentials: bool = True, zero_forces: bool = True
                  ) -> None:
        self.potentials(self.particles, zero_potentials)
        self.forces(self.particles, zero_forces)
