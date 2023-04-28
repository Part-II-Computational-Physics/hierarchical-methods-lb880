import numpy as np

__all__ = ['Point']

class Point():
    """Class to describe points in the 2D space. Centres described with complex
    numbers.
    
    Parameters
    ----------
    centre : complex, optional
        Complex coordinates of point, random in box size 1 if no argument
        given.

    Attributes
    ----------
    centre : complex
        Complex coordinates of point, random in box size 1 if no argument
        given.
    """

    def __init__(self, centre: complex = None) -> None:
        if centre:
            self.centre: complex = centre
        else:
            self.centre: complex = complex(
                np.random.random(), np.random.random()
            )
    
    def __repr__(self) -> str:
        return f'Point: {self.centre}'
    