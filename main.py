import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.animation import FuncAnimation
import random

matplotlib.use('TkAgg')

from classes import Universe

if __name__ == '__main__':
    num_bodies = 100
    size = 2
    G = 10
    dt = 0.001

    universe = Universe(num_bodies, size, dt, G, softening=0.04)

    r = 1
    v = 0.2 * np.sqrt(G * (num_bodies-1) / r)
    for i in range(num_bodies):
        theta = (i/num_bodies) * 2 * np.pi
        universe.body_x[i] = r * np.array([np.cos(theta),np.sin(theta)])
        universe.body_v[i] = v * np.array([-np.sin(theta),np.cos(theta)])

    # for i in range(num_bodies):
    #     r = random.uniform(0,1)
    #     theta = random.uniform(0,2*np.pi)
    #     universe.body_x[i] = r * np.array([np.cos(theta),np.sin(theta)])
    #     universe.body_v[i] = 1.2 * np.sqrt(num_bodies * G * r) * np.array([-np.sin(theta),np.cos(theta)])

    fig, ax = plt.subplots()
    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)
    ax.set_aspect('equal')

    points, = ax.plot(universe.body_x[:,0], universe.body_x[:,1], 'o')

    def animate(i):
        universe.update_positions()

        points.set_data(universe.body_x[:,0], universe.body_x[:,1])

        return [points]

    anim = FuncAnimation(fig, animate, frames=1000, interval=5*universe.dt, blit=True)

    plt.show()