import numpy as np
import matplotlib.pyplot as plt

from typing import List

from ..general import Particle
from . import cells

__all__ = ['do_bh', 'plot']


def do_bh(particles:List[Particle], max_level, theta, n_crit=2, zero_potentials=False):

    if zero_potentials:
        for particle in particles:
            particle.potential = 0.0

    root = cells.RootCell(0.5+0.5j, 1, max_level, theta)

    root.populate_with_particles(particles, n_crit)
    root.populate_mass_CoM()
    root.calculate_particle_potentials()
    

def plot(root:cells.RootCell):
    fig,ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    points = [source.centre for source in root.particles]
    X,Y = np.real(points), np.imag(points)

    ax.scatter(X,Y)

    import matplotlib.patches as patches

    def draw_rectangles(cell:cells.Cell):
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
    from ..general import direct_particle_potentials

    num_particles = 1000

    particles = [Particle() for _ in range(num_particles)]
    # print([particle.charge for particle in particles])

    max_level = 10
    theta = 0
    n_crit = 2

    do_bh(particles, max_level, theta, n_crit)

    direct_particle_potentials(particles)

    bh_pot = np.array([particle.potential for particle in particles])
    dir_pot = np.array([particle.direct_potential for particle in particles])
    diff_pot = bh_pot - dir_pot
    frac_err = np.abs(diff_pot / dir_pot)

    print(frac_err)
    print(np.max(frac_err))
    print(np.average(frac_err))


if __name__ == '__main__':
    main()
