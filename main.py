import numpy as np
import matplotlib

matplotlib.use('TkAgg')

from universe import Universe
from animator import Animator

if __name__ == '__main__':
    num_bodies = 5
    size = 2
    G = 1
    dt = 0.01

    universe = Universe(num_bodies, size, dt, G, softening=0.02)

    universe.initialise_positions_velocities('circle')

    animator = Animator(universe)
    animator.create_figure_for_animation()
    animator.produce_animation_with_momentum_energy()
