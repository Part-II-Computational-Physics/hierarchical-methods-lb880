import numpy as np

from .point import Point

class Particle(Point):
    """Sources to calculate multipoles from

    Inherits from Point class
    
    Attributes
    ----------
    centre : complex
        Coords of point, random if no argument given
    charge : float
        Charge associated with the point, real number,
        if no argument given, random in range [-1,1)
    potential : float
        To store evaluated potential
    """

    def __init__(self, charge:float=None, centre:complex=None, mass_like:bool=False) -> None:
        super().__init__(centre)

        if charge:
            self.charge:float = charge
        else:
            if mass_like: 
                self.charge: float = 1 - np.random.random()
            else:
                # random in range -1 to 1
                self.charge:float = 2*np.random.random() - 1

        self.potential:float = 0.0

    def __repr__(self) -> str:
        return f'Particle: {self.centre}, charge {self.charge}'