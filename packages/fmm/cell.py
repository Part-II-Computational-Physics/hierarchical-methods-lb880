from typing import List, Tuple, Set

from ..general import Particle

__all__ = ['Cell']

class Cell():
    """Class for operation on cell indices for the FMM method. 

    Attributes
    ----------
    index : Tuple[int]
        The `(x,y)` of the cell in the level.
        Referencing the array as `array[x,y]`.
    level : int
        The depth of the array the cell exists in, starting from 0.
    """

    def __init__(self, x: int, y: int, level: int) -> None:
        self.index: Tuple[int] = (x, y)
        self.level: int = level

    def children(self) -> List['Cell']:
        """Returns the children cells' locations, ordered from bottom left.

        Returns
        -------
        child_index : List[Cell]
            List of cell locations of the children.
        """

        x, y = self.index
        return [
            Cell(x*2,   y*2,   self.level+1),
            Cell(x*2+1, y*2,   self.level+1),
            Cell(x*2,   y*2+1, self.level+1),
            Cell(x*2+1, y*2+1, self.level+1)
        ]

    def parent(self) -> 'Cell':
        """Returns the parent cell's location.
        
        Returns
        -------
        parent_index : Index
            Cell location of the parent to the cell.
        """

        return Cell(self.index[0]//2, self.index[1]//2, self.level - 1)
    
    def neighbours(self) -> Set['Cell']:
        """Returns nearest neighbours to the index, as a set.
        
        Returns
        -------
        neighbours : Set[Index]
            Set of locations of the cell's neighbours.
        """

        if self.level == 0:
            return set()

        x, y = self.index
        max_coord = 2**self.level - 1
        neighbours = set()

        # middle row x_n = x
        if y != 0:
            neighbours.add(Cell(x, y-1, self.level))
        if y != max_coord:
            neighbours.add(Cell(x, y+1, self.level))

        # left row x_n = x-1
        if x !=0:
            neighbours.add(Cell(x-1, y, self.level))
            if y != 0:
                neighbours.add(Cell(x-1, y-1, self.level))
            if y != max_coord:
                neighbours.add(Cell(x-1, y+1, self.level))

        # right row x_n = x+1
        if x != max_coord:
            neighbours.add(Cell(x+1, y, self.level))
            if y != 0:
                neighbours.add(Cell(x+1, y-1, self.level))
            if y != max_coord:
                neighbours.add(Cell(x+1, y+1, self.level))
        
        return neighbours
    
    def interaction_list(self) -> Set['Cell']:
        """Returns the interaction list of the cell.
        
        Returns
        -------
        interactors : Set[Cell]
            Set of the cell locations in the interation list.
        """

        # top two levels have no interaction list
        if self.level <= 1:
            return set()
        
        all_possible = set()
        for parent_neighbour in self.parent().neighbours():
            all_possible.update(parent_neighbour.children())

        return all_possible - self.neighbours()
    
    @classmethod
    def particle_cell(self, particle: Particle, level: int) -> 'Cell':
        """Returns cell location in which the given particle lies, in the
        given level.
        
        Parameters
        ----------
        particle : Particle
            Particle for which to find location.
        level : int
            The level the location of the particle is to be calculated for, 
            indexed from 0 for the coarsest level.
        
        Returns
        -------
        Cell
            `Cell` object of the particle's location.
        """

        return Cell(
            int(particle.centre.real * 2**level),
            int(particle.centre.imag * 2**level),
            level
        )
    
    def __repr__(self) -> str:
        return f'Cell: {self.index}, lvl {self.level})'
    
    def __key(self) -> Tuple:
        return (self.index, self.level)

    def __hash__(self) -> int:
        return hash(self.__key())
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Cell):
            return self.__key() == other.__key()
        else:
            return False
