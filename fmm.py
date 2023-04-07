import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt

from typing import List, Tuple, Set
from numpy.typing import NDArray
    

class Particle():
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
    """

    def __init__(self, charge:float=None, centre:complex=None) -> None:
        if centre:
            self.centre:complex = centre
        else:
            self.centre:complex = np.random.random() + 1j*np.random.random()

        if charge:
            self.charge:float = charge
        else:
            # random in range -1 to 1
            self.charge:float = 2*np.random.random() - 1

        self.potential:float = 0.0

    def __repr__(self) -> str:
        return f'Particle: {self.centre}, charge {self.charge}'


def get_children(parent_coords:Tuple[int]) -> List[Tuple[int]]:
    x, y = parent_coords
    return [
        (x*2, y*2),
        (x*2+1, y*2),
        (x*2, y*2+1),
        (x*2+1, y*2+1)
    ]

def parent(child_coords:Tuple[int]) -> Tuple[int]:
    return (child_coords[0]//2, child_coords[1]//2)

def nearest_neighbours(coords:Tuple[int], level:int) -> Set[Tuple[int]]:
    if level == 0:
        return set()

    x,y = coords
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

def interaction_list(cell_coords:Tuple[int], level:int) -> Set[Tuple[int]]:
    if level <= 1:
        return set()
    
    own_neighbours = nearest_neighbours(cell_coords, level)
    parent_neighbours = nearest_neighbours(parent(cell_coords), level-1)
    all_possible = set()
    for p_n in parent_neighbours:
        all_possible.update(get_children(p_n))

    return all_possible - own_neighbours


def create_expansion_matricies(max_level:int, precision:int):
    """Returns a list of matricies for the expansion coefficients to be placed into
    
    For each 'cell' the first value also stores the complex centre of that cell
    """

    expansion_matricies = [np.zeros((2**l,2**l,2*precision+1), dtype=complex) for l in range(max_level+1)]
    for l, matrix in enumerate(expansion_matricies):
        first_val = 1/(2**(l+1))
        # stop of 1 as this is the so called max value, but will never appear
        vals = np.arange(first_val,1,2*first_val)
        X,Y = np.meshgrid(vals, vals, indexing='ij')
        matrix[:,:,0] = X + 1j*Y
    
    return expansion_matricies

def get_particle_cell(particle:Particle, level:int) -> Tuple[int]:
    return (
        int(particle.centre.real * 2**level),
        int(particle.centre.imag * 2**level)
    )


def calculate_multipole(particle:Particle, cell:Tuple[int], precision:int, level:int, matrix) -> None:
    """Update the relevant cell in given level with multipole due to particle
    
    Parameters
    ----------
    particle : Particle
        particle whose effect to add
    cell : Tuple[int]
        particles cell coords
    precision : int
        precision in multipole expansion
    level : int
        the level at which to add the particle's effect
    matrix : NDArray
        the full matrix for the relevant level
    """

    matrix[cell][1] += particle.charge
    k = np.arange(1,precision,1)
    matrix[cell][2:precision+1] += -particle.charge * (particle.centre - matrix[cell[0],cell[1],0])**k / k

def cell_M2M(cell:Tuple[int], child:Tuple[int], precision:int, matrix, children_matrix):
    """M2M from children for a single cell"""

    child_multipole = children_matrix[child][1:precision+1]

    matrix[cell][1] += child_multipole[0]

    z0 = children_matrix[child][0] - matrix[cell][0]

    for l in range(1,precision):
        k = np.arange(1,l+1)
        matrix[cell][l+1] += -(child_multipole[0] * z0**l / l) \
            + np.sum(child_multipole[1:l+1] \
                        * z0**(l-k) * binom(l-1,k-1))

def level_M2M(precision:int, level:int, matrix, children_matrix) -> None:
    """Perform M2M on a given level due to the multipoles in the child level"""
    # Can definitely optimise the way this is done
    for x in range(2**level):
        for y in range(2**level):
            for child in get_children((x,y)):
                cell_M2M((x,y),child,precision,matrix,children_matrix)

def cell_M2L(cell:Tuple[int], interactor:Tuple[int], precision:int, matrix) -> None:
    """Local expansion of a cell due to interactor"""

    z0 = matrix[interactor][0] - matrix[cell][0] # local expansion 'about origin' (so about cell)

    minus_and_plus = np.empty(precision-1)
    minus_and_plus[::2] = -1
    minus_and_plus[1::2] = 1

    k_vals = np.arange(1, precision)
    l_vals = np.arange(1, precision)

    interactor_multipole = matrix[interactor][1:precision+1]

    minus_bk_over_z0k = minus_and_plus * interactor_multipole[1:] / z0**k_vals

    matrix[cell][precision+1] += interactor_multipole[0] * np.log(-z0) + np.sum(minus_bk_over_z0k)
    matrix[cell][precision+2:] += -interactor_multipole[0] / (l_vals * z0**l_vals) \
                    + (1/z0**l_vals) * np.sum(minus_bk_over_z0k * binom(l_vals[:,np.newaxis] + k_vals - 1, k_vals-1), axis=1)
    
def level_M2L(precision:int, level:int, matrix) -> None:
    """Do M2L for a given level"""

    for x in range(2**level):
        for y in range(2**level):
            for interactor in interaction_list((x,y),level):
                cell_M2L((x,y),interactor,precision,matrix)

def cell_L2L(cell:Tuple[int], child:Tuple[int], precision:int, matrix, child_matrix) -> None:
    """Distribute local expansion to child cells"""
    
    z0 = child_matrix[child][0] - matrix[cell][0]

    for l in range(precision):
        for k in range(l, precision):
            child_matrix[child][precision+1+l] += matrix[cell][precision+1+k] * binom(k,l) * z0**(k-l)
            
    # l_vals = np.arange(precision)
    # k_vals = np.arange(1,precision)

    # child_matrix[child][precision+1:] += np.sum(
    #         matrix[cell][precision+2] * binom(k_vals, l_vals[:,np.newaxis]) * z0**(k_vals - l_vals[:,np.newaxis])
    #     ,axis=1)

            
    
def level_L2L(precision:int, level:int, matrix, child_matrix) -> None:
    """Distribute all locals in a given level to children level"""

    for x in range(2**level):
        for y in range(2**level):
            for child in get_children((x,y)):
                cell_L2L((x,y), child, precision, matrix, child_matrix)


def insert_particles(finest_matrix, max_level, particles:List[Particle], precision:int) -> List[List[Set[Particle]]]:
    finest_particles = [[set() for _ in range(2**max_level)] for _ in range(2**max_level)]
    for particle in particles:
        cell = get_particle_cell(particle, max_level)
        calculate_multipole(particle, cell, precision, max_level, finest_matrix)
        finest_particles[cell[0]][cell[1]].add(particle)
    
    return finest_particles

def upward_pass(precision:int, max_level:int, expansion_matricies):
    for level in range(max_level-1, -1, -1):
        level_M2M(precision, level, expansion_matricies[level], expansion_matricies[level+1])

def downward_pass(precision:int, max_level, expansion_matricies):
    """Perform downward pass"""
    # first two levels have zero local, as no interaction list
    for level in range(2, max_level):
        # interaction list contributions
        level_M2L(precision, level, expansion_matricies[level])
        # distribute to children
        level_L2L(precision, level, expansion_matricies[level], expansion_matricies[level+1])
    # don't want L2L for finest level, no children
    level_M2L(precision, max_level, expansion_matricies[max_level])

def get_particle_potentials(precision:int, max_level:int, finest_particles:List[List[Set[Particle]]], finest_matrix):
    for x, ys in enumerate(finest_particles):
        for y, elem in enumerate(ys):
            
            if not finest_particles[x][y]:
                continue

            cell_centre = finest_matrix[x,y,0]
            local = finest_matrix[x,y,precision+1:]
            l_vals = np.arange(len(local))

            near_particles = set()
            for neighbour_set in [finest_particles[xn][yn] for xn,yn in nearest_neighbours((x,y), max_level)]:
                near_particles.update(neighbour_set)
            near_particles.update(elem)

            for particle in elem:
                # far field expansion contribution
                z0 = particle.centre - cell_centre
                particle.potential -= np.sum(local * z0**l_vals).real

                # near-field
                for other_particle in near_particles-{particle}:
                    particle.potential -= other_particle.charge * np.log(abs(particle.centre - other_particle.centre))


def direct_particle_potentials(particles:List[Particle]):
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


    expansion_matricies = create_expansion_matricies(max_level, precision)
    for particle in particles:
        particle.potential = 0.0

    finest_particles = insert_particles(expansion_matricies[max_level], max_level, particles, precision)
    upward_pass(precision, max_level, expansion_matricies)
    downward_pass(precision, max_level, expansion_matricies)
    get_particle_potentials(precision, max_level, finest_particles, expansion_matricies[max_level])


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
    ticks = np.arange(0,1,2*expansion_matricies[max_level][0,0,0].real)

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