"""Array index coordinate operations"""

from typing import List, Tuple, Set

from ..classes import Particle

__all__ = ['children', 'parent', 'neighbours', 'interaction_list',
           'particle_cell']

def children(parent_coords: Tuple[int]) -> List[Tuple[int]]:
    """Returns children coordinates given parent coordinates
    
    (Coordinates are matrix indicies)
    
    Parameters
    ----------
    parent_coords : Tuple[int]
        Matrix indicies of parent
    
    Returns
    -------
    child_coords : Set[Tuple[int]]
        List of matrix indicies of children
    """

    x, y = parent_coords
    return [
        (x*2, y*2),
        (x*2+1, y*2),
        (x*2, y*2+1),
        (x*2+1, y*2+1)
    ]

def parent(child_coords: Tuple[int]) -> Tuple[int]:
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

def neighbours(cell_coords: Tuple[int], level: int) -> Set[Tuple[int]]:
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

def interaction_list(cell_coords: Tuple[int], level: int
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
    
    neighbours = neighbours(cell_coords, level)
    parent_neighbours = neighbours(parent(cell_coords), level-1)
    all_possible = set()
    for p_n in parent_neighbours:
        all_possible.update(children(p_n))

    return all_possible - neighbours

def particle_cell(particle: Particle, level: int) -> Tuple[int]:
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
