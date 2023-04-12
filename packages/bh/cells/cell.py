import numpy as np

from typing import List, Tuple
from numpy.typing import NDArray

from ...general import Point, Particle

class Cell(Point):
    """Class for cells that tree constructed from
    
    Inherits from Point
    
    Attributes
    ----------
    centre : complex
        coords of point,
        random if no argument given
    size : float
        size of the side of a box
    level : int
        level number (start at 0) in tree
    level_coords : tuple of int
        index coords of where the cell sits in its level
    parent : Cell
        the parent cell


    n_particles : int
        number of particles in cell
    particles : list of Particle
        list of particles contained

    bit_children : bitwise
        child locations eg, child in 2 and 4 is 1010
    children : array of Cells
        cells children

    Methods
    -------
    distance : float
        return distance to another point
    create_multipole : None
        generate multipole due to bodies contained
    """

    def __init__(self,
                 centre:complex,
                 size:float,
                 parent:'Cell'=None) -> None:
        
        super().__init__(centre)

        self.size:float = size
        if parent:
            self.level:int = parent.level + 1
        else:
            self.level:int = 0

        self.level_coords:Tuple[int] = (
            int(self.centre.real * 2**(self.level) -0.5),
            int(self.centre.imag * 2**(self.level) -0.5)
        )

        self.parent:'Cell' = parent

        self.n_particles:int = 0
        self.particles:List[Particle] = []

        self.bit_children:int = 0 # 0000, bitwise operations
        self.children:NDArray[+'Cell'] = np.zeros(4, dtype=object)

    def __repr__(self) -> str:
        return f'Cell lvl{self.level}: {self.centre} {self.n_particles} particle'
    
    def print_tree(self, level=0):
        print('\t'*level, self.level_coords, self)
        for child in self.children:
            if child:
                child.print_tree(level+1)


    def _add_child(self, quadrant:int, cells:List['Cell']):
        """Add new child in given octant
        
        Create relevant references in cell object, and in cells list
        """

        # bitwise operations to determine if left or right
        #   then if up or down, and apply appropriate shift
        # (if both yeild 0, then bottom left, otherwise relative to there) 
        child_centre =  self.centre + 0.25*self.size * (
                (-1+2*(quadrant & 1)) + 1j*(-1+2*((quadrant & 2)>>1)))
        
        # add child to array of children, in correct location
        self.children[quadrant] = Cell(child_centre,
                                        self.size/2,
                                        self)
        self.bit_children += (1<<quadrant)

        # add child to cells list
        cells.append(self.children[quadrant])


    def _particle_quadrant(self, particle_centre:complex) -> int:
        """Return int 0 to 3 corresponding to the particles quadrant"""
        return (particle_centre.real > self.centre.real) | \
                    (particle_centre.imag > self.centre.imag)<<1 # int 0 to 3
    

    def _split_cell(self, n_crit:int, max_level:int, cells:List['Cell']):
        """Splits self, distributing children and creating cells as needed"""

        for particle in self.particles:
            quadrant = self._particle_quadrant(particle.centre)

            # check for no child child
            #   if there is no match between the bit children and the quadrant bit
            if not self.bit_children & (1 << quadrant):
                self._add_child(quadrant, cells)

            # add particle to child
            self.children[quadrant]._add_particle(particle,n_crit,max_level,cells)

        
    def _add_particle(self, particle:Particle, n_crit:int, max_level:int, cells:List['Cell']):
        self.n_particles += 1
        self.particles.append(particle)

        if (self.n_particles < n_crit) or (self.level == max_level): # still leaf or max level
            return
        
        elif self.n_particles == n_crit: # just become not leaf
            self._split_cell(n_crit, max_level, cells)

        else: # already branch
            quadrant = self._particle_quadrant(particle.centre)

            # check for no child child
            #   if there is no match between the bit children and the quadrant bit
            if not self.bit_children & (1 << quadrant):
                self._add_child(quadrant, cells)

            # add particle to child
            self.children[quadrant]._add_particle(particle,n_crit,max_level,cells)

    
    def _get_Mass_and_CoM(self) -> None:
        if not self.bit_children: # leaf
            masses = np.array([particle.charge for particle in self.particles])
            centres = np.array([particle.centre for particle in self.particles])
        
        else: # has children to 'take' CoM from
            masses = np.array([child.total_mass for child in self.children if child])
            centres = np.array([child.CoM for child in self.children if child])

        # calculate total mass and centre of mass
        self.total_mass = np.sum(masses)
        self.CoM = np.sum(masses * centres) / self.total_mass