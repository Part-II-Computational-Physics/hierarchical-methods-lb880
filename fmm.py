import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt

from typing import List, Tuple, Set
from numpy.typing import NDArray
    

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

    def __init__(self, charge: float=None, centre: complex=None) -> None:
        if centre:
            self.centre: complex = centre
        else:
            self.centre: complex = np.random.random() + 1j*np.random.random()

        if charge:
            self.charge: float = charge
        else:
            # random in range -1 to 1
            self.charge: float = 2*np.random.random() - 1

        self.potential: float = 0.0

    def __repr__(self) -> str:
        return f'Particle: {self.centre}, charge {self.charge}'


def get_children(parent_coords: Tuple[int]) -> List[Tuple[int]]:
    """Returns children coordinates given parent coordinates
    
    (Coordinates are matrix indicies)
    
    Parameters
    ----------
    parent_coords : Tuple[int]
        Matrix indicies of parent
    
    Returns
    -------
    child_coords : List[Tuple[int]]
        List of matrix indicies of children
    """

    x, y = parent_coords
    return [
        (x*2, y*2),
        (x*2+1, y*2),
        (x*2, y*2+1),
        (x*2+1, y*2+1)
    ]

def get_parent(child_coords: Tuple[int]) -> Tuple[int]:
    """Returns parent coordinates given child coordinates
    
    (Coordinates are matrix indicies)
    
    Parameters
    ----------
    child_coords : Tuple[int]
        Matrix indicies of child
    
    Returns
    -------
    parent_coords : Tuple[int]
        Matrix indicies of parent
    """

    return (child_coords[0]//2, child_coords[1]//2)

def get_neighbours(cell_coords: Tuple[int], level: int) -> Set[Tuple[int]]:
    """Returns nearest neighbour coordinates given cell coordinates
    
    (Coordinates are matrix indicies)
    
    Parameters
    ----------
    cell_coords : Tuple[int]
        Matrix indicies of cell to find nearest_neighbours
    level : int
        The level the nearest neighbours are to be calculated for, 
        indexed from 0 for the coarsest level
    
    Returns
    -------
    neighbour_coords : Set[Tuple[int]]
        Set of matrix indicies of the nearest neighbours
    """

    if level == 0:
        return set()

    x,y = cell_coords
    max_coord = 2**level - 1
    neighbours = set()

    # middle row x_n = x
    if y != 0:
        neighbours.add((x,y-1))
    if y != max_coord:
        neighbours.add((x,y+1))

    # left row x_n = x-1
    if x !=0:
        neighbours.add((x-1,y))
        if y != 0:
            neighbours.add((x-1,y-1))
        if y != max_coord:
            neighbours.add((x-1,y+1))

    # right row x_n = x+1
    if x != max_coord:
        neighbours.add((x+1,y))
        if y != 0:
            neighbours.add((x+1,y-1))
        if y != max_coord:
            neighbours.add((x+1,y+1))
    
    return neighbours

def get_interaction_list(cell_coords: Tuple[int], level: int
                         ) -> Set[Tuple[int]]:
    """Returns nearest neighbour coordinates given cell coordinates
    
    (Coordinates are matrix indicies)
    
    Parameters
    ----------
    cell_coords : Tuple[int]
        Matrix indicies of cell to find interaction list
    level : int
        The level the nearest neighbours are to be calculated for, 
        indexed from 0 for the coarsest level
    
    Returns
    -------
    interactor_coords : Set[Tuple[int]]
        Set of matrix indicies of the cells in interaction list (interactors)
    """

    if level <= 1:
        return set()
    
    neighbours = get_neighbours(cell_coords, level)
    parent_neighbours = get_neighbours(get_parent(cell_coords), level-1)
    all_possible = set()
    for p_n in parent_neighbours:
        all_possible.update(get_children(p_n))

    return all_possible - neighbours

def get_particle_cell(particle: Particle, level: int) -> Tuple[int]:
    """Returns cell coordinates in which the given particle lies
    
    (Coordinates are matrix indicies)
    
    Parameters
    ----------
    particle : Particle
        Particle for which to find coordinates
    level : int
        The level the nearest neighbours are to be calculated for, 
        indexed from 0 for the coarsest level
    
    Returns
    -------
    cell_coords : Tuple[int]
        Matrix indicies of the cell the particle is in
    """

    return (
        int(particle.centre.real * 2**level),
        int(particle.centre.imag * 2**level)
    )


def create_expansion_arrays(precision: int, max_level: int) -> List[NDArray]:
    """Returns a list of arrays for the expansion coefficients to be placed into
    
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
    k_vals = np.arange(1,precision)
    array[cell][2:precision+1] -= particle.charge * z0**k_vals / k_vals


def _cell_M2M(precision: int, cell: Tuple[int],
             array: NDArray, child_array: NDArray) -> None:
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

    for child in get_children(cell):
        child_multipole = child_array[child][1:precision+1]

        array[cell][1] += child_multipole[0]

        z0 = child_array[child][0] - array[cell][0]

        for l in range(1,precision):
            k_vals = np.arange(1,l+1)
            array[cell][l+1] += -(child_multipole[0] * z0**l / l) \
                                + np.sum(child_multipole[1:l+1] \
                                         * z0**(l-k_vals) \
                                         * binom(l-1, k_vals-1))

def level_M2M(precision: int, level: int,
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
            _cell_M2M(precision, (x,y), array, child_array)

def cell_M2L(precision: int, cell: Tuple[int], interactor: Tuple[int],
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

    minus_and_plus = np.empty(precision-1)
    minus_and_plus[::2] = -1
    minus_and_plus[1::2] = 1

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
    
def level_M2L(precision: int, level: int, array: NDArray) -> None:
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
            for interactor in get_interaction_list((x,y), level):
                cell_M2L(precision,(x,y),interactor,array)

def cell_L2L(precision: int, cell: Tuple[int],
             array: NDArray, child_array: NDArray) -> None:
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
    
    for child in get_children(cell):
        z0 = child_array[child][0] - array[cell][0]

        # ################################################
        # # Can definitely optimise the way this is done #
        # ################################################
        for l in range(precision):
            for k in range(l, precision):
                child_array[child][precision+1+l] += \
                    array[cell][precision+1+k] * binom(k,l) * z0**(k-l)
                
        # l_vals = np.arange(precision)
        # k_vals = np.arange(1,precision)

        # child_matrix[child][precision+1:] += np.sum(
        #         matrix[cell][precision+2] * \
        #           binom(k_vals, l_vals[:,np.newaxis]) \
        #           * z0**(k_vals - l_vals[:,np.newaxis])
        #     ,axis=1)

            
    
def level_L2L(precision: int, level: int,
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
            cell_L2L(precision, (x,y), matrix, child_matrix)


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
        cell = get_particle_cell(particle, level)
        calculate_multipole(precision, particle, cell, array)
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
        Expected that the finest grained level has already had multipoles filled
    """

    # Starts at second to max level, and then goes upwards up to the 0th level
    for level in range(len(expansion_arrays)-2, -1, -1):
        level_M2M(precision, level,
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
        level_M2L(precision, level, expansion_arrays[level])
        # distribute to children
        level_L2L(precision, level, expansion_arrays[level], expansion_arrays[level+1])

    # no L2L for finest level, no children
    level_M2L(precision, max_level, expansion_arrays[max_level])


def evaluate_particle_potentials(precision: int, max_level: int,
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
    
    for x, ys in enumerate(finest_particles):
        for y, cell_particles in enumerate(ys):
            
            # if no particles in that cell, pass the cell
            if not cell_particles:
                continue

            # near particles are those in the cell and its neighbours
            near_particles = set()
            for neighbour_set in [finest_particles[xn][yn] 
                                  for xn,yn in get_neighbours((x,y), max_level)
                                 ]:
                near_particles.update(neighbour_set)
            near_particles.update(cell_particles)

            cell_centre = finest_array[x,y,0]
            l_vals = np.arange(precision)

            for particle in cell_particles:
                # far field expansion contribution
                z0 = particle.centre - cell_centre
                particle.potential -= np.sum(
                        finest_array[x,y,precision+1:] * z0**l_vals
                    ).real

                # near-field
                # near particles excluding the own particle
                for other_particle in near_particles-{particle}:
                    particle.potential -= other_particle.charge \
                        * np.log(abs(particle.centre - other_particle.centre))


def direct_particle_potentials(particles:List[Particle]):
    """Calculate through pairwise interactions the particle potentials and store
    
    Parameters
    ----------
    particles : List[Particle]
        List of the `Particle` objects to calculate the direct potentials for
    """
    for particle in particles:
        particle.direct_potential = 0.0

    for i, particle in enumerate(particles):
        for other in particles[i+1:]:
            potential = - np.log(abs(particle.centre-other.centre))
            particle.direct_potential += other.charge * potential
            other.direct_potential += particle.charge * potential


def main():
    max_level = 5
    precision = 10
    num_particles = 500

    particles = [Particle() for _ in range(num_particles)]


    expansion_arrays = create_expansion_arrays(precision, max_level)
    for particle in particles:
        particle.potential = 0.0

    finest_particles = insert_particles(precision, max_level, particles, expansion_arrays[max_level])
    upward_pass(precision, expansion_arrays)
    downward_pass(precision, max_level, expansion_arrays)
    evaluate_particle_potentials(precision, max_level, finest_particles, expansion_arrays[max_level])


    direct_particle_potentials(particles)

    potentials = [particle.potential for particle in particles]
    direct_potentials = [particle.direct_potential for particle in particles]

    potential_err = list((np.array(potentials) - np.array(direct_potentials)) / np.array(direct_potentials))
    percents = [abs(round(deci, 3))*100 for deci in potential_err]

    print([particle.centre for particle in particles])
    print()
    print('Dir:', direct_potentials)
    print('FMM:', potentials)
    print()
    print('Err:', potential_err)
    print('  %:', percents)
    print([abs(err) > 0.01 for err in potential_err])
    print('Max ', max(percents), '%')

    X = [particle.centre.real for particle in particles]
    Y = [particle.centre.imag for particle in particles]
    ticks = np.arange(0,1,2*expansion_arrays[max_level][0,0,0].real)

    fig, ax = plt.subplots()
    ax.plot(X,Y, 'o')
    ax.set_aspect('equal')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks(ticks, minor=True)
    ax.set_yticks(ticks, minor=True)
    ax.grid(True, 'minor')
    plt.show()

if __name__ == '__main__':
    main()