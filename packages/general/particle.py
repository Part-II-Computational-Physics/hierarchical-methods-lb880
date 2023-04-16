from numpy.typing import NDArray

import numpy as np

from .point import Point

__all__ = ['Particle']

class Particle(Point):
    """Have charge and position, used to calculate interactions with other
    particles.

    Inherits from Point class.
    
    Parameters
    ----------
    centre : complex, optional
        Complex coordinates of point, random in box size 1 if no argument
        given.
    charge : float, optional
        Charge associated with the point, real number,
        if no argument given, random in range [-1,1).
    mass_like: bool, default False
        If true limits charge to range [0,1) instead.

    Attributes
    ----------
    centre : complex
        Complex coordinates of point, random in box size 1 if no argument
        given.
    charge : float
        Charge associated with the point, real number,
        if no argument given, random in range [-1,1).
    potential : float
        To store evaluated potential.
    force : 2DArray
        To store evaluated force.
    """

    def __init__(self, charge: float = None, centre: complex = None,
                 mass_like: bool = False) -> None:
        super().__init__(centre)

        if charge:
            self.charge:float = charge
        else:
            if mass_like: 
                self.charge: float = 1 - np.random.random()
            else:
                # random in range -1 to 1
                self.charge:float = 2*np.random.random() - 1

        self.potential: float = 0.0
        self.force: NDArray = np.zeros(2)

    def __repr__(self) -> str:
        return f'Particle: {self.centre}, charge {self.charge}'
    