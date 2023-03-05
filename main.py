import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.animation import FuncAnimation
import random

matplotlib.use('TkAgg')

from universe import Universe
from animator import Animator

if __name__ == '__main__':
    num_bodies = 100
    size = 2
    G = 10
    dt = 0.001

    universe = Universe(num_bodies, size, dt, G, softening=0.05)

    universe.initialise_positions_velocities('circle')

    animator = Animator(universe)
    animator.create_figure_basic()
    animator.produce_animation_with_momentum_energy()
