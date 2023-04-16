"""
Pairwise
--------
Methods to calculate direct potentials or forces on a list of `Particle`
objects, through pairwise interaction (Order N squared methods).
"""

from typing import List

import math
import numpy as np

from . import Particle

__all__ = ['potentials', 'forces']

def potentials(particles: List[Particle], zero_potentials: bool = False
               ) -> None:
    """Calculate through pairwise interactions the particle potentials and
    store in `Particle` object attribute.
    
    Parameters
    ----------
    particles : List[Particle]
        List of the `Particle` objects to calculate the direct potentials for.
    zero_potentials : bool, default False
        Controls if particle potentials are reset to zero in the process.
        Default of False leaves potentials unchanged.
    """

    if zero_potentials:
        for particle in particles:
            particle.direct_potential = 0.0

    for i, particle in enumerate(particles):
        for other in particles[i+1:]:
            potential = - math.log(abs(particle.centre - other.centre))
            particle.direct_potential += other.charge * potential
            other.direct_potential += particle.charge * potential

def forces(particles: List[Particle], zero_forces: bool = False) -> None:
    """Calculate through pairwise interactions the particle potentials and
    store in `Particle` object attribute.
    
    Parameters
    ----------
    particles : List[Particle]
        List of the `Particle` objects to calculate the direct forces for.
    zero_forces : bool, default False
        Controls if particle forces are reset to zero in the process.
        Default of False leaves potentials unchanged.
    """

    if zero_forces:
        for particle in particles:
            particle.direct_force = np.zeros(2, dtype=float)

    for i, particle in enumerate(particles):
        for other in particles[i+1:]:
            z0 = particle.centre - other.centre
            force = particle.charge * other.charge \
                * np.array((z0.real, z0.imag)) / abs(z0)**2
            particle.direct_force += force
            other.direct_force -= force
