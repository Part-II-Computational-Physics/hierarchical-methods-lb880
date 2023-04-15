from typing import List, Tuple

import numpy as np

from ...general import Point, Particle

__all__ = ['Cell']

class Cell(Point):
    """Class for cells that tree constructed from
    
    Inherits from Point
    
    Attributes
    ----------
    centre : complex
        Coords of point. Random if no argument given
    size : float
        Size of the side of a box
    level : int
        Level number (start at 0) in tree
    parent : Cell
        The parent cell
    level_coords : tuple of int
        Index coords of where the cell sits in its level


    n_particles : int
        Number of particles in cell
    particles : List[Particle]
        List of particles contained

    bit_children : bitwise
        Child locations eg, child in 2 and 4 is 1010
    children : List[Cell]
        Cells children

    Methods
    -------
    print_tree
        Print the tree from this cell as the root
    """

    def __init__(self, centre: complex, size: float, parent: 'Cell', n_crit: int = 2) -> None:
        
        super().__init__(centre)

        self.size: float = size

        if parent:
            self.level:int = parent.level + 1
        else:
            self.level:int = 0

        self.parent: 'Cell' = parent

        self.n_crit: int = n_crit

        self.n_particles: int = 0
        self.particles: List[Particle] = []

        self.bit_children: int = 0 # 0000, for bitwise operations
        self.children: List['Cell'] = [None]*4

    def _add_child(self, quadrant: int, cells: List['Cell']) -> None:
        """Add new child to the cell in given quadrant.
        Create relevant references in cell object, and in given cells list.

        Parameters
        ----------
        quadrant : int
            The quadrant to add the child to
        cells : List[Cell]
            List of all cells in the tree to append the child to
        """

        # bitwise operations to determine the coords of the child centre
        #   if left or right
        #   then if up or down, and apply appropriate shift
        # (if both yeild 0, then bottom left, otherwise relative to there) 
        child_centre =  self.centre + 0.25 * self.size \
            * ((-1+2*(quadrant & 1)) + 1j*(-1+2*((quadrant & 2)>>1)))
        
        # add child to array of children, in correct location
        self.children[quadrant] = Cell(centre = child_centre,
                                       size = self.size/2,
                                       parent = self,
                                       n_crit =  self.n_crit)
        # store in bitchildren
        self.bit_children += (1<<quadrant)

        # add child to cells list
        cells.append(self.children[quadrant])

    def _quadrant(self, centre: complex) -> int:
        """Return int 0 to 3 corresponding to the given `centre`'s quadrant
        relative to self. 
        E.g. the quadrant to add a child too due to an added particle.

        Parameters
        ----------
        centre : complex
            Complex coordinates of the location to find quadrant

        Returns
        -------
        quadrant : int
            Integer 0-3 corresponding to the quadrant.
            0 bottom left, 1 bottom right, 2 top left, 3 top right
        """

        # int 0 to 3
        return (centre.real > self.centre.real) \
                | (centre.imag > self.centre.imag)<<1
    
    def _split_cell(self, max_level: int, cells: List['Cell']
                    ) -> None:
        """Splits self cell, distributing children and creating child cells as
        needed.
        
        Parameters
        ----------
        n_crit : int
            Number of particles in a cell that should be then split at.
            (Used to initialise children)
        max_level : int
            Maximum level to recurse to in the tree. 
            (Used to initialise children)
        cells : List[Cell]
            List of all cells in the tree.
            (Used to initalise children)
        """

        for particle in self.particles:
            quadrant = self._quadrant(particle.centre)

            # make sure a child cell is avaliable for insertion
            # need one if there is no match between the bit children and the
            #   quadrant bit
            if not self.bit_children & (1 << quadrant):
                self._add_child(quadrant, cells)

            # add particle to child
            self.children[quadrant]._add_particle(particle, max_level,
                                                  cells)
        
    def _add_particle(self, particle: Particle, max_level: int, cells: List['Cell']) -> None:
        """Add a particle to the given cell, splitting the cell if required
        
        Parameters
        ----------
        particle : Particle
            The particle to be add
        n_crit : int
            Number of particles in a cell that should be then split at
        max_level : int
            Maximum level to recurse to in the tree
        cells : List[Cell]
            List of all cells in the tree.
            (Used to add any newly created cells)
        """

        self.n_particles += 1
        self.particles.append(particle)

        # still a leaf or at the max level, no splitting or children
        if (self.n_particles < self.n_crit) or (self.level == max_level):
            return
        
        elif self.n_particles == self.n_crit: # just become not leaf
            self._split_cell(self.n_crit, max_level, cells)

        else: # already branch
            quadrant = self._quadrant(particle.centre)

            # make sure a child cell is avaliable for insertion
            # need one if there is no match between the bit children and the
            #   quadrant bit
            if not self.bit_children & (1 << quadrant):
                self._add_child(quadrant, cells)

            # add particle to child
            self.children[quadrant]._add_particle(particle, self.n_crit, max_level,
                                                  cells)

    def _get_mass_CoM(self) -> None:
        """Calculate the total mass and centre of mass of the cell.
        If a leaf this is calculated due to the contribution of all the cells.
        If a branch this is calulated due to the contribution of children.
        """
        if not self.bit_children: # leaf
            masses = np.array(
                [particle.charge for particle in self.particles])
            centres = np.array(
                [particle.centre for particle in self.particles])
        
        else: # has children to 'take' CoM from
            masses = np.array(
                [child.total_mass for child in self.children if child])
            centres = np.array(
                [child.CoM for child in self.children if child])

        # calculate total mass and centre of mass
        self.total_mass = np.sum(masses)
        self.CoM = np.sum(masses * centres) / self.total_mass

    def print_tree(self, level: int = 0) -> None:
        level_coords = (
            int(self.centre.real * 2**(self.level) -0.5),
            int(self.centre.imag * 2**(self.level) -0.5)
        )
        print('\t'*level, level_coords, self)
        for child in self.children:
            if child:
                child.print_tree(level+1)

    def __repr__(self) -> str:
        return f'Cell lvl{self.level}: {self.centre} \
            {self.n_particles} particle'
