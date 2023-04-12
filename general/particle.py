import numpy as np

from .point import Point

class Particle(Point):
    """Sources to calculate multipoles from

    Inherits from Point class
    
    Attributes
    ----------
    centre : complex
        coords of point,
        random if no argument given
    charge : float
        charge associated with the point, real number,
        if no argument given, random in range [-1,1)

    Methods
    -------
    distance : float
        return distance to another point
    """

    def __init__(self, charge:float=None, centre:complex=None) -> None:
        super().__init__(centre)
        if charge:
            self.charge:float = charge
        else:
            # # random in range -1 to 1
            # self.charge:float = 2*np.random.random() - 1
            self.charge: float = 1 - np.random.random()

        self.potential:float = 0.0

    def __repr__(self) -> str:
        return f'Particle: {self.centre}, charge {self.charge}'