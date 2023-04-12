import numpy as np

__all__ = ['Particle']

class Particle():
    """Sources to calculate multipoles from
    
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

    def __init__(self, charge: float = None, centre: complex = None) -> None:
        if centre:
            self.centre: complex = centre
        else:
            self.centre: complex = np.random.random() + 1j*np.random.random()

        if charge:
            self.charge: float = charge
        else:
            # # random in range -1 to 1
            # self.charge: float = 2*np.random.random() - 1
            self.charge: float = np.random.random()

        self.potential: float = 0.0

    def __repr__(self) -> str:
        return f'Particle: {self.centre}, charge {self.charge}'