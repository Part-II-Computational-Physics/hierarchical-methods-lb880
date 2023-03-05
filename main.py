import numpy as np
import matplotlib

matplotlib.use('TkAgg')

from universe import Universe
from animator import Animator

if __name__ == '__main__':
    num_bodies = 10
    size = 2
    G = 20
    dt = 0.001

    universe = Universe(num_bodies, size, dt, G, softening=0.1)

    universe.initialise_positions_velocities('random')

    animator = Animator(universe)
    animator.create_figure_basic()
    animator.produce_animation_with_momentum_energy()
