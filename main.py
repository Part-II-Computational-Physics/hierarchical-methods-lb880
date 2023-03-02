import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.animation import FuncAnimation
import random

matplotlib.use('TkAgg')

from classes import Universe

if __name__ == '__main__':
    num_bodies = 2
    size = 2
    G = 0.1
    dt = 0.01

    universe = Universe(num_bodies, size, dt, G)

    for i in range(num_bodies):
        x = random.uniform(-size/2, size/2)
        y = random.uniform(-size/2, size/2)
        universe.body_x[i] = np.array([x,y])
        theta = np.arctan2(y,x)
        universe.body_v[i] = 1e-2 * random.uniform(0.3,1) * np.array([-np.sin(theta), np.cos(theta)])

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