from typing import List, Set, Tuple
from numpy.typing import NDArray

import numpy as np

from ..general import Particle

from . import tools

__all__ = ['create_expansion_arrays', 'insert_particles', 'upward_pass',
           'downward_pass', 'evaluate_particles', 'do_fmm']

def create_expansion_arrays(precision: int, max_level: int) -> List[NDArray]:
    """Returns a list of arrays for the expansion coefficients to be placed
    into.
    
    For each cell the first value also stores the complex centre of that cell

    Parameters
    ----------
    precision : int
        The precision for which expansions are to be calculated to
    max_level : int
        The max level number to be generated up to.
        e.g. `max_level = 3` will generate 4 arrays, the finest level of
        which will be `expansion_arrays[3]`

    Returns
    -------
    expansion_arrays : List[3DArray]
        List of the arrays at each level, with first value preloaded with
        the complex coordinate of the cell, and later values initialied to zero
    """

    expansion_arrays = [np.zeros((2**l, 2**l, 2*precision + 1), dtype=complex)
                        for l in range(max_level+1)]

    # initialise the complex coordinates
    for l, array in enumerate(expansion_arrays):
        first_val = 1/(2**(l+1))
        # stop of 1 as this is the so called max value, but will never appear
        vals = np.arange(first_val, 1, 2*first_val)
        X,Y = np.meshgrid(vals, vals, indexing='ij')
        array[:,:,0] = X + 1j*Y
    
    return expansion_arrays

def insert_particles(precision: int, level: int, particles:List[Particle],
                     array: NDArray) -> List[List[Set[Particle]]]:
    """Insert particles into the given matrix of the given level.
    Inserted particles will update the relevant multipole expansions of only
    the cell they are inserted into. Returns a 2D array (list of list) of sets
    of the particles that reside in each of the cells.

    It is expected that this should only be used to insert particles into the
    finest level of the arrays. 

    Parameters
    ----------
    precision : int
        Precision for the multipole expansion
    level : int
        The level to insert on,
        indexed from 0 for the coarsest level
        (expected to be the `max_level` of the arrays of expansions)
    array : 3DArray
        Expansion array of the level to perform on
        (expected to be the finest grained array)

    Returns
    -------
    level_particles : List[List[Set[Particle]]]
        '2D Array' of sets of `Particle` objects of the relevant particles in
        each cell
    """

    # create the '2D array' of the sets
    level_particles = [[set() for _ in range(2**level)]
                       for _ in range(2**level)]

    for particle in particles:
        cell = tools.coord.particle_cell(particle, level)
        tools.cell.calculate_multipole(precision, particle, cell, array)
        # add particle to the array
        level_particles[cell[0]][cell[1]].add(particle)
    
    return level_particles

def upward_pass(precision: int, expansion_arrays: List[NDArray]) -> None:
    """Perform the upward pass on the array structre. 
    Starting from the second finest grained array perform M2M to propogate the
    multipoles up the structure. Does not perform on finest grained, as this is
    where the multipoles should have been calculated for.

    Parameters
    ----------
    precision : int
        Precision for the multipole expansions
    expansion_arrays : List[3DArray]
        List of the expansion arrays for all of the levels.
        Expected that the finest grained level has already had multipoles
        filled
    """

    # Starts at second to max level, and then goes upwards up to the 0th level
    for level in range(len(expansion_arrays)-2, -1, -1):
        tools.level.M2M(precision, level,
                        expansion_arrays[level], expansion_arrays[level+1])

def downward_pass(precision: int, expansion_arrays: List[NDArray]) -> None:
    """Perform downward pass on the expansion arrays to calculate all of the
    local expansions for each cell.
    M2L is used to get local contributions for each cell from its interaction
    list, and the L2L used to shift the local expansion to the children.

    L2L not done from final level as no children.
    M2L and L2L not done on/from coarsest two levels for open boundary
    conditions, as there is no interaction list for either. 

    Parameters
    ----------
    precision : int
        Precision for the multipole expansions
    expansion_arrays : List[3DArray]
        List of the expansion arrays for all of the levels
    """

    max_level = len(expansion_arrays) - 1

    # first two levels have zero local, as no interaction list
    for level in range(2, max_level):
        # interaction list contributions
        tools.level.M2L(precision, level, expansion_arrays[level])
        # distribute to children
        tools.level.L2L(precision, level, expansion_arrays[level],
                        expansion_arrays[level+1])

    # no L2L for finest level, no children
    tools.level.M2L(precision, max_level, expansion_arrays[max_level])

def _neighbour_particles(cell: Tuple, level: int, array_particles: List[List[Set[Particle]]]
                         ) -> Set[Particle]:
    """Returns a set of all particles in the nearest neighbours of the given
    cell.
    
    Parameters
    ----------
    cell : Tuple
        The array indicies of the relevant cell
    level : int
        The level of the array being considered
    array_particles: 2DArray
        Array of sets of Particles that are in each cell

    Returns
    -------
    neighbour_particles : Set[Particle]
        Set of Particles that are in the cell's neighbours
    """
    
    # near particles are those in the cell and its neighbours
    neighbour_particles: Set[Particle] = set()
    for neighbour_set in [array_particles[xn][yn] \
                          for xn,yn in tools.coord.neighbours(cell, level)]:
        neighbour_particles.update(neighbour_set)
    
    return neighbour_particles

def evaluate_particles(precision: int, max_level: int, 
                       finest_particles: List[List[Set[Particle]]],
                       finest_array: NDArray) -> None:
    """Evaluate the potentials for each particle in the '2D Array' of sets of
    `Particle`, `finest_particles`. Potential is a sum of the far-field (from
    calculated local expansions) and near-field (from pairwise interaction of 
    all particles in the cell's interaction list of cells).

    Potentials must first be zeroed for correct output.

    Parameters
    ----------
    precision : int
        Precision for the expansions
    max_level : int
        The level at which the particles potentials are to be calculated from.
        Must match the level of `finest_particles` and `finest_matrix`.
    finest_particles : List[List[Set[Particle]]]
        '2D Array' of sets of `Particle` objects of the relevant particles in
        each cell
    finest_array : 3DArray
        Expansion array for the relevant `max_level`
    """
    
    l_vals = np.arange(precision)

    for x, ys in enumerate(finest_particles):
        for y, cell_particles in enumerate(ys):
            
            # if no particles in that cell, pass the cell
            if not cell_particles:
                continue
            
            # get particles that can interact with particles in the cell
            near_particles: Set[Particle] = cell_particles.union(
                _neighbour_particles((x,y), max_level, finest_particles)
            )
            local = finest_array[x,y,precision+1:]

            for particle in cell_particles:
                # far field local expansion contribution
                z0 = particle.centre - finest_array[x,y,0]
                particle.potential -= np.sum(local * z0**l_vals).real

                w_prime = np.sum(l_vals[1:] * local[1:] * z0**l_vals[:-1])
                particle.force += particle.charge * np.array((w_prime.real, -w_prime.imag))

                # near-field
                # near particles excluding the own particle
                for other in near_particles-{particle}:
                    z0 = particle.centre - other.centre
                    particle.potential -= other.charge * np.log(abs(z0))
                    
                    particle.force += particle.charge * other.charge \
                        * np.array((z0.real, z0.imag)) / abs(z0)**2


def do_fmm(precision: int, particles: List[Particle],
           expansion_arrays: List[NDArray],
           zero_potentials: bool = False, zero_forces: bool = False
           ) -> None:
    """Updates particle potentials in the given particle list using the full
    FMM method, with the given parameters
    
    Parameters
    ----------
    precision : int
        The precision for which expansions are to be calculated to
    particles : List[Particle]
        List of the `Particle` objects to calculate potentials for
    expansion_arrays : List[3DArray]
        List of the (empty) arrays of expansions to be filled
    zero_potentials : bool
        Controls if particle potentials are explicitly reset to zero in the
        process. Default of False (does not change the potentials)
    """

    max_level = len(expansion_arrays) - 1

    if zero_potentials:
        for particle in particles:
            particle.potential = 0.0
    if zero_forces:
        for particle in particles:
            particle.force = np.zeros(2, dtype=float)

    finest_particles = insert_particles(precision, max_level, particles,
                                        expansion_arrays[max_level])
    upward_pass(precision, expansion_arrays)
    downward_pass(precision, expansion_arrays)
    evaluate_particles(precision, max_level, finest_particles,
                                 expansion_arrays[max_level])
