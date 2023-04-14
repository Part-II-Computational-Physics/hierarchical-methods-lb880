"""Cell class does operations on index and level of cell

Focused on cell relations, not actions
"""

from typing import List, Tuple, Set

from ..general import Particle

__all__ = ['Cell']

class Cell():
    def __init__(self, x: int, y: int, level: int) -> None:
        self.index: Tuple[int] = (x, y)
        self.level: int = level

    def children(self) -> List['Cell']:
        """Returns children `Index`s

        Returns
        -------
        child_index : List[Index]
            List of Index of children
        """

        x, y = self.index
        return [
            Cell(x*2,   y*2,   self.level+1),
            Cell(x*2+1, y*2,   self.level+1),
            Cell(x*2,   y*2+1, self.level+1),
            Cell(x*2+1, y*2+1, self.level+1)
        ]

    def parent(self) -> 'Cell':
        """Returns parent `Index`
        
        Returns
        -------
        parent_index : Index
            Index of parent
        """

        return Cell(self.index[0]//2, self.index[1]//2, self.level - 1)
    
    def neighbours(self) -> Set['Cell']:
        """Returns nearest neighbour `Index`s
        
        Returns
        -------
        neighbours : Set[Index]
            Set of indicies of neighbours
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
        """Returns interaction list
        
        Returns
        -------
        interactors : Set[Index]
            Set of `Index`s for cells in the interation list
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
        cell_index : Index
            `Index` object for the cell the particle is in
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
