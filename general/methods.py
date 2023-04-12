import math

from typing import List
from . import Particle

__all__ = ['direct_particle_potentials']

def direct_particle_potentials(particles: List[Particle],
                               zero_potentials: bool = False) -> None:
    """Calculate through pairwise interactions the particle potentials and store
    
    Parameters
    ----------
    particles : List[Particle]
        List of the `Particle` objects to calculate the direct potentials for
    zero_potentials : bool
        Controls if particle potentials are explicitly reset to zero in the
        process. Default of False (does not change the potentials)
    """
    if zero_potentials:
        for particle in particles:
            particle.direct_potential = 0.0

    for i, particle in enumerate(particles):
        for other in particles[i+1:]:
            potential = - math.log(abs(particle.centre-other.centre))
            particle.direct_potential += other.charge * potential
            other.direct_potential += particle.charge * potential