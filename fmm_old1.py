import numpy as np
import scipy as sp

class Source():
    def __init__(self, charge, pos) -> None:
        self.charge = charge
        self.pos = pos

class Ibox():
    """For 1/r force (log(r) potentials)
    
    Attributes
    ----------
    multipole_coefficients : list
        index 0 is Q, rest are the a_k
    """

    def __init__(self, p, l, side, centre) -> None:
        """level l
        
        Parameters
        ----------
        p : int
            precision of expansions (has log term then p more)
            
        l : int
            level the ibox is on
        """
        self.l = l
        self.side = side
        self.centre = centre

        self.children = []

        self.multipole_coefficients = np.zeros(p+1, dtype=complex)

    def __repr__(self) -> str:
        string = f'Ibox lvl {self.l}'
        if len(self.children) != 0:
            string += ' (children)'
        return string
    
    def print_tree_coefficients(self, level=0):
        string = level*'\t' + self.__repr__()
        string += f' Coeffs: {[coeff for coeff in self.multipole_coefficients]}'
        print(string)
        for child in self.children:
            child.print_tree_coefficients(level+1)

    def upward_pass(self, p:int, l_max:int, sources) -> None:
        """Create children recursively for the given ibox at the next level, until l_max
        
        Parameters
        ----------
        p : int
            precision of multipole expansion
        l_max : int
            will stop creating children when l=l_max
        """
        if self.l == l_max:
            self.create_multipole_expansion(sources)
            return
        
        centres = [self.centre + (-1-1j) * self.side/4,
                   self.centre + (1-1j)  * self.side/4,
                   self.centre + (-1+1j) * self.side/4,
                   self.centre + (1+1j)  * self.side/4]
        self.children = [Ibox(p, self.l+1, self.side/2, centre) for centre in centres]
        
        for child in self.children:
            child.upward_pass(p, l_max, sources)
            self.M2M(child)

    def create_multipole_expansion(self, sources) -> None:
        """
        Parameters
        ----------
        charges_and_positions : NDArray
            2D array of charges and then positions coords (x,y) of all sources
        """
        # get relevant sources
        sources = np.array(
            [source for source in sources
                if  (np.real(self.centre) - self.side/2 <= np.real(source[1]) < np.real(self.centre) + self.side/2) 
                and (np.imag(self.centre) - self.side/2 <= np.imag(source[1]) < np.imag(self.centre) + self.side/2)]
        )

        if sources.size == 0:
            # coeffs are default of zero
            return

        # get multipole coefficients
        # Q
        self.multipole_coefficients[0] = np.sum(sources[:,0])
        # a_k
        for k in range(1, len(self.multipole_coefficients)):
            self.multipole_coefficients[k] = np.sum(-sources[:,0] * (sources[:,1]-self.centre)**k / k)

    def M2M(self, child):
        """Perform M2M method"""

        z0 = child.centre - self.centre

        self.multipole_coefficients[0] += child.multipole_coefficients[0]

        for l in range(1, len(self.multipole_coefficients)):
            self.multipole_coefficients[l] += \
                -(child.multipole_coefficients[0] * z0**l / l) \
                    + np.sum(child.multipole_coefficients[1:l] \
                             * z0**(l-np.arange(1,l,1)) \
                             * sp.special.binom(l-1, np.arange(0,l-1,1)))
            
    def downward_pass():
        pass


def main():
    N = 10
    n = np.ceil(0.5*np.log2(N))
    precision = 0.1
    p = np.ceil(np.log2(precision))
    p = 4

    sources = np.zeros((N,2), dtype=complex)

    for i in range(N):
        # charge
        sources[i,0] = 2 * np.random.randint(2) - 1
        # position (complex)
        sources[i,1] = np.random.rand() + 1j*np.random.rand()
    
    # sources[0] = np.array([1, 0.25*(1+1j)])
    # sources[1] = np.array([2, 0.25*(3+1j)])
    # sources[2] = np.array([-2, 0.25*(1+3j)])
    # sources[3] = np.array([-3, 0.25*(3+3j)])
    # print(sources)

    top_ibox = Ibox(p, 0, 1, 0.5+0.5j)
    
    top_ibox.upward_pass(p, n, sources)
    # top_ibox.print_tree_coefficients()

    # plot points
    import matplotlib.pyplot as plt

    colors = ['r' if charge==1 else 'b' for charge in sources[:,0]]
    complex_points = [position for position in sources[:,1]]
    X = np.real(complex_points)
    Y = np.imag(complex_points)

    fig, ax = plt.subplots()
    print(colors)
    ax.scatter(X, Y,c=colors)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    ax.set_xticks(np.arange(0,1,1/(2**n)), minor=True)
    ax.set_yticks(np.arange(0,1,1/(2**n)), minor=True)
    ax.grid(True, 'minor', 'both')
    
    plt.show()


if __name__ == '__main__':
    main()