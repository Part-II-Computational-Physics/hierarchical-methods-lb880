"""Cell operation tools"""

from typing import Tuple
from numpy.typing import NDArray

import numpy as np
from scipy.special import binom

from .. import tools
from ...general import Particle

__all__ = ['calculate_multipole', 'M2M', 'M2L', 'L2L']


def calculate_multipole(precision: int, particle: Particle, cell: Tuple[int],
                        array: NDArray) -> None:
    """Update the relevant cell in given level with multipole due to particle
    
    Parameters
    ----------
    precision : int
        Precision for the multipole expansion
    particle : Particle
        Particle whose effect to add
    cell : Tuple[int]
        Coordinates of cell to add effect to
    array : 3DArray
        Expansion array the cell to update is within
    """

    # first term
    array[cell][1] += particle.charge

    # remaining terms
    z0 = particle.centre - array[cell][0]
    k_vals = np.arange(1, precision)
    array[cell][2:precision+1] -= particle.charge * z0**k_vals / k_vals


def M2M(precision: int, cell: Tuple[int], array: NDArray, child_array: NDArray
        ) -> None:
    """Perform M2M from children for a given cell to calculate its multipole
    
    Parameters
    ----------
    precision : int
        Precision for the multipole expansion
    cell : Tuple[int]
        Coordinates of cell to calculate multipole from M2M
    array : 3DArray
        Expansion array in which the cell resides
    child_array : 3DArray
        Expansion array in which the children exist
    """

    for child in tools.coord.children(cell):
        child_multipole = child_array[child][1:precision+1]

        array[cell][1] += child_multipole[0]

        z0 = child_array[child][0] - array[cell][0]

        for l in range(1,precision):
            k_vals = np.arange(1,l+1)
            array[cell][l+1] += -(child_multipole[0] * z0**l / l) \
                                + np.sum(child_multipole[1:l+1] \
                                         * z0**(l-k_vals) \
                                         * binom(l-1, k_vals-1))
            
def M2L(precision: int, cell: Tuple[int], interactor: Tuple[int],
        array: NDArray) -> None:
    """Use M2L method to get the local expansion for a cell due to an 
    interactor's multipole
    
    Parameters
    ----------
    precision : int
        Precision for the local expansion
    cell : Tuple[int]
        Coordinates of cell to calculate local expansion from M2L
    interactor : Tuple[int]
        Coordinates of cell of whose multipole to calculate from
    array : 3DArray
        Expansion array in which the cell resides
    """

    # local expansion 'about origin' (so about cell)
    z0 = array[interactor][0] - array[cell][0]

    # arrays for calculations
    minus_and_plus = np.ones(precision-1)
    minus_and_plus[::2] = -1
    k_vals = np.arange(1, precision)
    l_vals = np.arange(1, precision)

    interactor_multipole = array[interactor][1:precision+1]

    minus_bk_over_z0k = minus_and_plus * interactor_multipole[1:] / z0**k_vals

    array[cell][precision+1] += interactor_multipole[0] * np.log(-z0) \
                              + np.sum(minus_bk_over_z0k)

    array[cell][precision+2:] += \
            -interactor_multipole[0] / (l_vals * z0**l_vals) + (1/z0**l_vals) \
            * np.sum(minus_bk_over_z0k \
                     * binom(l_vals[:,np.newaxis] + k_vals - 1, k_vals-1),
                     axis=1)
    
def L2L(precision: int, cell: Tuple[int], array: NDArray, child_array: NDArray
        ) -> None:
    """Use L2L method to distribute local expansion to a given cell's children
    
    Parameters
    ----------
    precision : int
        Precision for the local expansion
    cell : Tuple[int]
        Coordinates of cell to calculate local expansion from M2L
    array : 3DArray
        Expansion array in which the cell resides
    child_array : 3DArray
        Expansion array in which the children exist
    """
    
    for child in tools.coord.children(cell):
        z0 = child_array[child][0] - array[cell][0]

        # ################################################
        # # Can definitely optimise the way this is done #
        # ################################################
        for l in range(precision):
            for k in range(l, precision):
                child_array[child][precision+1+l] += \
                    array[cell][precision+1+k] * binom(k,l) * z0**(k-l)
                