import numpy as np

class Point():
    """General point class
    
    Attributes
    ----------
    centre : complex
        Coords of point, random if no argument given

    Methods
    -------
    distance : float
        Return distance to another point
    """

    def __init__(self, centre:complex=None) -> None:
        if centre:
            self.centre:complex = centre
        else:
            self.centre:complex = np.random.random() + 1j*np.random.random()
    
    def __repr__(self) -> str:
        return f'Point: {self.centre}'

    def distance(self, other:'Point'):
        return abs(self.centre - other.centre)