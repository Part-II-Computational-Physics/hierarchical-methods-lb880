import math
import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt

from typing import List, Tuple, Set
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

        self.potential:float = 0.0

    def __repr__(self) -> str:
        return f'Particle: {self.centre}, charge {self.charge}'
    
# class Particle():
#     """Sources to calculate multipoles from
    
#     Attributes
#     ----------
#     centre : complex
#         Coords of point, random if no argument given
#     charge : float
#         Charge associated with the point, real number,
#         if no argument given, random in range [-1,1)
#     potential : float
#         To store evaluated potential
#     """

#     def __init__(self, charge: float = None, centre: complex = None) -> None:
#         if centre:
#             self.centre: complex = centre
#         else:
#             self.centre: complex = np.random.random() + 1j*np.random.random()

#         if charge:
#             self.charge: float = charge
#         else:
#             # random in range -1 to 1
#             self.charge: float = 2*np.random.random() - 1

#         self.potential: float = 0.0

#     def __repr__(self) -> str:
#         return f'Particle: {self.centre}, charge {self.charge}'

    
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


class RootCell(Cell):
    """Wrapper for total tree operations
    
    Attributes
    ----------
    max_level : int
        the max depth the tree is allowed to go to
    cells : List[Cell]
        list of all cells in the tree, in order they were created
    level_matricies : List[Array]
        matrix for each level containing reference (as to cells) of the index
        of each of the cells on that level
    """

    def __init__(self,
                 centre:complex,
                 size:float,
                 max_level:int,
                 theta:float) -> None:
        
        super().__init__(centre, size)

        self.max_level:int = max_level
        self.theta: float = theta
        self.cells:List[Cell] = [self]

    
    def populate_with_particles(self, particles:List[Particle], n_crit:int=2):
        self.particles = particles
        self.n_particles = len(particles)
        
        if self.n_particles >= n_crit:
            self._split_cell(n_crit, self.max_level, self.cells)

        # for particle in particles:
        #     self._add_particle(particle,
        #                       n_crit,
        #                       self.max_level,
        #                       self.cells)

    def populate_mass_CoM(self):
        # iterate from leaf nodes first
        for cell in reversed(self.cells):
            cell._get_Mass_and_CoM()
    
    def print_tree_CoMs(self):
        def _print_CoM(cell:Cell, level:int):
            print('\t'*level, cell.total_mass, cell.CoM, cell)
            for child in cell.children:
                if child:
                    _print_CoM(child, level+1)
        
        _print_CoM(self, 0)

    def calculate_particle_potentials(self):

        def _interact_with_cell(particle:Particle, cell:Cell):
            
            # definitely cannot interact if particle in the cell
            if particle in cell.particles:
                if cell.bit_children:
                    # look through the children
                    for child in cell.children:
                        if child:
                            _interact_with_cell(particle, child)
                return

            # check theta to see if should go deeper
            z0_abs = abs(particle.centre - cell.CoM)
            if cell.size < self.theta * z0_abs: # far cell, CoM interact
                particle.potential -= cell.total_mass * math.log(z0_abs)
                return
            
            # near cell, go deeper if children
            if cell.bit_children:
                for child in cell.children:
                    if child:
                        _interact_with_cell(particle, child)
            else:
                for other in cell.particles:
                    particle.potential -= particle.charge * math.log(abs(particle.centre - other.centre))

        # consider each particle in the system
        for particle in self.particles:
            # start from the top of the tree and then work recursively down
            # compare to theta at each point
            _interact_with_cell(particle, self)

def direct_particle_potentials(particles:List[Particle]):
    for particle in particles:
        particle.direct_potential = 0.0

    for i, particle in enumerate(particles):
        for other in particles[i+1:]:
            potential = - np.log(abs(particle.centre-other.centre))
            particle.direct_potential += other.charge * potential
            other.direct_potential += particle.charge * potential


def plot(root:RootCell):
    fig,ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    points = [source.centre for source in root.particles]
    X,Y = np.real(points), np.imag(points)

    ax.scatter(X,Y)

    import matplotlib.patches as patches

    def draw_rectangles(cell:Cell):
        corner = cell.centre - cell.size*(0.5+0.5j)
        p = patches.Rectangle((corner.real,corner.imag),cell.size,cell.size, fill=False, color='red')
        ax.add_patch(p)
        if cell.bit_children == 0:
            return
        else:
            for child in cell.children:
                if child:
                    draw_rectangles(child)

    draw_rectangles(root)

    plt.show()


def main():
    num_particles = 10

    particles = [Particle(1) for _ in range(num_particles)]

    max_level = 10
    n_crit = 2

    root = RootCell(0.5+0.5j, 1, max_level, 0.5)

    root.populate_with_particles(particles, n_crit)
    plot(root)
    root.populate_mass_CoM()

    for particle in particles:
        particle.potential = 0.0

    root.calculate_particle_potentials()

    direct_particle_potentials(particles)

    bh_pot = np.array([particle.potential for particle in particles])
    dir_pot = np.array([particle.direct_potential for particle in particles])
    diff_pot = bh_pot - dir_pot
    frac_err = diff_pot / dir_pot

    print(frac_err)


if __name__ == '__main__':
    main()
