from typing import List

import numpy as np
from scipy.special import binom

from ...general import Point, Particle

__all__ = ['Cell']

class Cell(Point):
    """Cells used to construct the Barnes-Hut tree.
    
    Inherits from `Point`.
    
    Parameters
    ----------
    centre : complex
        Complex coordinates of the centre of the cell.
    size : float
        Size of the side of the box.
    parent : Cell
        The parent cell.
    terms : int
        Number of terms in the 'expansion of the multipole'. If `1` then uses
        CoM method instead.
    n_crit : int
        Number of particles in a cell to split at.
        Default value of `2` will leave one particle per leaf cell (as each
        cell splits when it has 2 particles in it).

    Attributes
    ----------
    centre : complex
        Complex coordinates of the centre of the cell.
    size : float
        Size of the side of the box.
    level : int
        Level number (starting from 0) where the cell is in the tree.
    parent : Cell
        The parent cell.
    terms : int
        Number of terms in the 'expansion of the multipole'. If `0` then uses
        CoM method instead.
    n_crit : int
        Number of particles in a cell to split at.
        e.g. value of `2` will leave one particle per leaf cell (as each cell
        splits when it has 2 particles in it).
    n_particles : int
        Number of particles in the cell.
    particles : List[Particle]
        List of particles contained with in the cell. 
    bit_children : 4-bit
        Child location storage.
        eg. child in 2 and 4 is 1010
    children : List[Cell]
        List of the children cells (ordered so the first item is bitwise 0001).
    """

    def __init__(self, centre: complex, size: float, parent: 'Cell',
                 terms: int, n_crit: int) -> None:
        
        super().__init__(centre)

        self.size: float = size

        if parent:
            self.level: int = parent.level + 1
        else:
            self.level: int = 0

        self.parent: 'Cell' = parent

        self.terms: int = terms
        if terms > 0:
            self.multipole: int = np.zeros(terms, dtype=complex)

        self.n_crit: int = n_crit

        self.n_particles: int = 0
        self.particles: List[Particle] = []

        self.bit_children: int = 0 # 0000, for bitwise operations
        self.children: List['Cell'] = [None]*4

    def _add_child(self, quadrant: int, cells: List['Cell']) -> None:
        """Add new child to the cell in given quadrant.
        Creates relevant references in cell object, and in given cells list.

        Parameters
        ----------
        quadrant : int
            The quadrant of the parent the child is added to.
        cells : List[Cell]
            List of all cells in the tree to append the child to.
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
                                       terms = self.terms,
                                       n_crit =  self.n_crit)
        # store in bitchildren
        self.bit_children += (1<<quadrant)

        # add child to cells list
        cells.append(self.children[quadrant])

    def _quadrant(self, centre: complex) -> int:
        """Return integer 0 to 3 corresponding to the given `centre`'s quadrant
        relative to the cell. 
        Eg. the quadrant to add a new child too due to an added particle.

        Parameters
        ----------
        centre : complex
            Complex coordinates of the location to find the quadrant for.

        Returns
        -------
        quadrant : int
            Integer 0-3 corresponding to the quadrant.
            0 bottom left, 1 bottom right, 2 top left, 3 top right.
        """

        return (centre.real > self.centre.real) \
                | (centre.imag > self.centre.imag)<<1
    
    def _split_cell(self, max_level: int, cells: List['Cell']) -> None:
        """Splits the cell, distributing particles to children with children
        cells created as needed.
        
        Parameters
        ----------
        max_level : int
            Maximum level to recurse to in the tree.
        cells : List[Cell]
            List of all cells in the tree.
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
        
    def _add_particle(self, particle: Particle, max_level: int,
                      cells: List['Cell']) -> None:
        """Add a particle to the given cell, splitting the cell or distributing
        child to children if required.
        
        Parameters
        ----------
        particle : Particle
            The particle to be added to the cell.
        max_level : int
            Maximum level to recurse to in the tree.
        cells : List[Cell]
            List of all cells in the tree.
        """

        self.n_particles += 1
        self.particles.append(particle)

        # still a leaf or at the max level, no splitting or children
        if (self.n_particles < self.n_crit) or (self.level == max_level):
            return
        
        elif self.n_particles == self.n_crit: # just become not leaf
            self._split_cell(max_level, cells)

        else: # already branch
            quadrant = self._quadrant(particle.centre)

            # make sure a child cell is avaliable for insertion
            # need one if there is no match between the bit children and the
            #   quadrant bit
            if not self.bit_children & (1 << quadrant):
                self._add_child(quadrant, cells)

            # add particle to child
            self.children[quadrant]._add_particle(particle, max_level, cells)

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

    def _calculate_multipole(self) -> None:
        """Update the cell's multipole due to the presence of the particles
        within.
        """

        for particle in self.particles:
            # first term
            self.multipole[0] += particle.charge

            # remaining terms
            z0 = particle.centre - self.centre
            k_vals = np.arange(1, self.terms)
            self.multipole[1:self.terms+1] -= particle.charge \
                                            * z0**k_vals / k_vals
            
    def _M2M(self) -> None:
        """Use the M2M operation to generate the multipole the cell due to its
        children.
        """
        for child in self.children:
            if child:

                self.multipole[0] += child.multipole[0]

                z0 = child.centre - self.centre

                for l in range(1,self.terms):
                    k_vals = np.arange(1,l+1)
                    self.multipole[l] += -(child.multipole[0] * z0**l / l) \
                        + np.sum(child.multipole[1:l+1] \
                                 * z0**(l-k_vals) * binom(l-1, k_vals-1))

    def print_tree(self, level: int = 0) -> None:
        """Print the tree from this cell downwards. 
        Best used for a `RootCell`.

        Parameters
        ----------
        level : int, default 0
            Controls number of tabs print is indented by.
        """
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
