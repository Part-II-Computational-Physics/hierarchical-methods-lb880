import numpy as np
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
        self.children:NDArray[+object] = np.zeros(4, dtype=object)

        self.precision:int = precision
        self.multipole:NDArray[+complex] = np.zeros(precision, dtype=complex)
        self.local:NDArray[+complex] = np.zeros(precision, dtype=complex)

    def __repr__(self) -> str:
        return f'Cell: {self.centre} {self.n_particles} particle'
    
    def print_tree(self, level=0):
        print('\t'*level, self)
        for child in self.children:
            if child:
                child.print_tree(level+1)


    def _particle_quadrant(self, particle_centre:complex) -> int:
        """Return int 0 to 3 corresponding to the particles quadrant
        
        Will create child quadrant if not existing already
        """
        quadrant = (particle_centre.real > self.centre.real) | \
                    (particle_centre.imag > self.centre.imag)<<1 # int 0 to 3
        
        # check for no child child
        #   create if none
        # if there is no match between the bit children and the quadrant bit
        if not self.bit_children & (1 << quadrant):
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
            
        return quadrant
    

    def _split_cell(self, n_crit:int):
        """Splits self, distributing children and creating cells as needed"""

        for particle in self.particles:
            quadrant = self._particle_quadrant(particle.centre)
            # add particle to child
            self.children[quadrant].add_particle(particle,n_crit)

        
    def add_particle(self, particle:Particle, n_crit:int):
        self.n_particles += 1
        self.particles.append(particle)

        if self.n_particles < n_crit: # still leaf
            return
        
        elif self.n_particles == n_crit: # just become not leaf
            self._split_cell(n_crit)

        else: # already branch
            quadrant = self._particle_quadrant(particle.centre)
            # add particle to child
            self.children[quadrant].add_particle(particle,n_crit)


    def get_multipole(self):
        # get multipole coefficients
        charges = np.array([particle.charge for particle in self.particles])
        positions = np.array([particle.centre for particle in self.particles])
        # Q
        self.multipole[0] = np.sum(charges)
        # a_k
        for k in range(1, len(self.multipole)):
            self.multipole[k] = np.sum(-charges * (positions-self.centre)**k / k)

def main():
    num_particles = 4
    particles = [ Particle(1) for _ in range(num_particles) ]

    root = Cell(0.5*(1+1j),1,4)

    for particle in particles:
        root.add_particle(particle,4)

    root.print_tree()

if __name__ == '__main__':
    main()