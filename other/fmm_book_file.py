import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from typing import List
from numpy.typing import NDArray

class Point():
    """General point class
    
    Attributes
    ----------
    centre : complex
        coords of point,
        random if no argument given

    Methods
    -------
    distance : float
        return distance to another point
    """

    def __init__(self, centre:complex=None) -> None:
        if centre:
            self.centre:complex = centre
        else:
            self.centre:complex = np.random.random() + 1j*np.random.random()

    def distance(self, other:'Point'):
        return abs(self.centre - other.centre)
    
class Particle(Point):
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

    Methods
    -------
    distance : float
        return distance to another point
    """

    def __init__(self, charge:float=None, centre:complex=None) -> None:
        super().__init__(centre)
        if charge:
            self.charge:float = charge
        else:
            # random in range -1 to 1
            self.charge:float = 2*np.random.random() - 1

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
    
    precision : int
        number of terms in multipole expansion
    multipole : NDArray of complex
        coefficients of the multipole expansion
    local : NDArray of complex
        coefficients of the multipole local

    Methods
    -------
    distance : float
        return distance to another point
    create_multipole : None
        generate multipole due to bodies contained
    """

    def __init__(self, centre:complex, size:float, precision:int=1, parent:'Cell'=None) -> None:
        super().__init__(centre)

        self.size:float = size

        self.parent:'Cell' = parent

        self.n_particles:int = 0
        self.particles:List[Particle] = []

        self.bit_children:int = 0 # 0000, bitwise operations
        self.children:NDArray[+'Cell'] = np.zeros(4, dtype=object)

        self.precision:int = precision
        self.multipole:NDArray[+complex] = np.zeros(precision, dtype=complex)
        self.local:NDArray[+complex] = np.zeros(precision, dtype=complex)

    def __repr__(self) -> str:
        return f'Cell: {self.centre} {self.n_particles} particle'
    
    def print_tree(self, level=0):
        print('\t'*level, f'({int(self.centre.real*2**(level) -0.5)},{int(self.centre.imag*2**(level) -0.5)})', self)
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
                                        self.precision,
                                        self)
        self.bit_children += (1<<quadrant)

        # add child to cells list
        cells.append(self.children[quadrant])


    def _particle_quadrant(self, particle_centre:complex) -> int:
        """Return int 0 to 3 corresponding to the particles quadrant"""
        return (particle_centre.real > self.centre.real) | \
                    (particle_centre.imag > self.centre.imag)<<1 # int 0 to 3
    

    def _split_cell(self, n_crit:int, cells:List['Cell']):
        """Splits self, distributing children and creating cells as needed"""

        for particle in self.particles:
            quadrant = self._particle_quadrant(particle.centre)

            # check for no child child
            #   if there is no match between the bit children and the quadrant bit
            if not self.bit_children & (1 << quadrant):
                self._add_child(quadrant, cells)

            # add particle to child
            self.children[quadrant].add_particle(particle,n_crit,cells)

        
    def add_particle(self, particle:Particle, n_crit:int, cells:List['Cell']):
        self.n_particles += 1
        self.particles.append(particle)

        if self.n_particles < n_crit: # still leaf
            return
        
        elif self.n_particles == n_crit: # just become not leaf
            self._split_cell(n_crit, cells)

        else: # already branch
            quadrant = self._particle_quadrant(particle.centre)

            # check for no child child
            #   if there is no match between the bit children and the quadrant bit
            if not self.bit_children & (1 << quadrant):
                self._add_child(quadrant, cells)

            # add particle to child
            self.children[quadrant].add_particle(particle,n_crit,cells)


    def _calculate_multipole(self) -> None:
        """Explicit calculation of multipole coeffs due to constituent particles"""
        charges = np.array([particle.charge for particle in self.particles])
        positions = np.array([particle.centre for particle in self.particles])
        # Q
        self.multipole[0] = np.sum(charges)
        # a_k
        for k in range(1, len(self.multipole)):
            self.multipole[k] = np.sum(-charges * (positions-self.centre)**k / k)
    

    def _M2M(self, child:'Cell') -> None:
        """Perform M2M method"""

        z0 = child.centre - self.centre

        print(child.multipole[0])
        self.multipole[0] += child.multipole[0]

        for l in range(1, self.precision):
            self.multipole[l] += \
                -(child.multipole[0] * z0**l / l) \
                    + np.sum(child.multipole[1:l] \
                             * z0**(l-np.arange(1,l,1)) \
                             * sp.special.binom(l-1, np.arange(0,l-1,1)))


    def get_multipole(self) -> None:
        """Either use M2M or calculation to get multipole of cell"""

        if self.bit_children == 0: # leaf
            self._calculate_multipole()
        else: # branch
            for child in self.children:
                if child:
                    self._M2M(child)

def main():
    num_particles = 10
    particles = [ Particle(1) for _ in range(num_particles) ]
    p=4

    root = Cell(0.5*(1+1j),1,p)
    cells = [root]

    for particle in particles:
        root.add_particle(particle,n_crit=10,cells=cells)

    root._calculate_multipole()
    print(root.multipole)

    root.multipole = np.zeros(p)

    for cell in reversed(cells):
        cell.get_multipole()
    print(root.multipole)

if __name__ == '__main__':
    main()