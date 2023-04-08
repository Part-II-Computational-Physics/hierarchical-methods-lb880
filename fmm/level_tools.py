from numpy.typing import NDArray

from . import coord_tools
from . import cell_tools

def M2M(precision: int, level: int,
              array: NDArray, child_array: NDArray) -> None:
    """Perform M2M to calculate multipoles of a given level
    due to the multipoles in the child level
    
    Parameters
    ----------
    precision : int
        Precision for the multipole expansion
    level : int
        The level to perform M2M on, 
        indexed from 0 for the coarsest level
    array : 3DArray
        Expansion array of the level to perform on
    child_array : 3DArray
        Expansion array in which the children exist (level below)
    """

    # ################################################
    # # Can definitely optimise the way this is done #
    # ################################################
    for x in range(2**level):
        for y in range(2**level):
            cell_tools.M2M(precision, (x,y), array, child_array)

def M2L(precision: int, level: int, array: NDArray) -> None:
    """Perform M2L to calculate locals of a given level due to the multipoles of
    interactors on the level
    
    Parameters
    ----------
    precision : int
        Precision for the multipole expansion
    level : int
        The level to perform M2L on, 
        indexed from 0 for the coarsest level
    array : 3DArray
        Expansion array of the level to perform on
    """

    for x in range(2**level):
        for y in range(2**level):
            for interactor in coord_tools.get_interaction_list((x,y), level):
                cell_tools.M2L(precision,(x,y),interactor,array)

def L2L(precision: int, level: int,
              matrix: NDArray, child_matrix:NDArray) -> None:
    """Perform L2L to distribute local expansions of a given level to children
    level
    
    Parameters
    ----------
    precision : int
        Precision for the local expansion
    level : int
        The level to perform L2L on, 
        indexed from 0 for the coarsest level
    array : 3DArray
        Expansion array of the level to perform on
    child_array : 3DArray
        Expansion array in which the children exist (level below)
    """

    for x in range(2**level):
        for y in range(2**level):
            cell_tools.L2L(precision, (x,y), matrix, child_matrix)