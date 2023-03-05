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
    num_bodies = 50
    size = 2
    G = 10
    dt = 0.001

    universe = Universe(num_bodies, size, dt, G, softening=0.05)

    # r = 1
    # v = 0.2 * np.sqrt(G * (num_bodies-1) / r)
    # for i in range(num_bodies):
    #     theta = (i/num_bodies) * 2 * np.pi
    #     universe.body_x[i] = r * np.array([np.cos(theta),np.sin(theta)])
    #     universe.body_v[i] = v * np.array([-np.sin(theta),np.cos(theta)])

    for i in range(num_bodies):
        r = random.uniform(0,1)
        theta = random.uniform(0,2*np.pi)
        universe.body_x[i] = r * np.array([np.cos(theta),np.sin(theta)])
        universe.body_v[i] = 1 * np.sqrt(num_bodies * G * r) * np.array([-np.sin(theta),np.cos(theta)])

    animator = Animator(universe)
    animator.create_figure_basic()
    animator.produce_animation_basic()
    