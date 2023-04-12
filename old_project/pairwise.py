import numpy as np
import numpy.typing as npt

def calculate_accelerations(
        properties:dict,
        masses:npt.NDArray,
        positions:npt.NDArray
    ) -> npt.NDArray:
    """Calculate the accelerations of all bodies in system

    Takes positions as arguments and then returns accelerations
    
    No approx in method
    
    Runs using python for loops

    Parameters
    ----------
    properties : dict
        dictionary of system properties
    masses : NDArray
        array of masses of all bodies
    positions : NDArray
        array of positions of all bodies

    Returns
    -------
    accelerations : NDArray
        array of velocites of all bodies
    """

    accelerations = np.zeros((properties['num_bodies'], 2))

    for i,position1 in enumerate(positions[:-1]):
        for j,position2 in enumerate(positions[i+1:], i+1):
            # r points object 1 to object 2
            r = position1 - position2
            mag_r = np.linalg.norm(r)
            dir_r = r / mag_r
            # force felt by 1 points at 2
            # can later multiply in masses
            # softening factor ensures that distance is never close to zero => inverse finite
            F = - (properties['G'] * dir_r) / (mag_r**2 + properties['softening']**2)
            # calculate accelerations
            accelerations[i] += F * masses[j]
            accelerations[j] -= F * masses[i]

    return accelerations

def calculate_accelerations_np(
        properties:dict,
        masses:npt.NDArray,
        positions:npt.NDArray
    ) -> npt.NDArray:
    """Calculate the accelerations of all bodies in system
    
    No approx in method
    
    Runs using numpy vectors

    O(N) for up to 400 bodies (assumed due to numpy optimisation)

    Parameters
    ----------
    properties : dict
        dictionary of system properties
    masses : NDArray
        array of masses of all bodies
    positions : NDArray
        array of positions of all bodies

    Returns
    -------
    accelerations : NDArray
        array of velocites of all bodies
    """
    
    accelerations = np.zeros((properties['num_bodies'],2))

    for i, pos in enumerate(positions):
        # points towards current body
        r = pos - positions[1:]
        mag_r = np.linalg.norm(r, axis=1).reshape((-1,1))
        dir_r = r / mag_r
        # force felt by current body
        # can later multiply in masses
        # softening factor ensures that distance is never close to zero => inverse finite
        F = - (properties['G'] * dir_r) / (mag_r**2 + properties['softening']**2)
        # calculate accelerations
        accelerations[i] = np.sum(F * masses[1:], axis=0)
        # roll on the body_x
        positions = np.roll(positions, -1, axis=0)
        masses = np.roll(masses, -1, axis=0)

    return accelerations
