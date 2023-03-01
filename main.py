import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.animation import FuncAnimation

matplotlib.use('TkAgg')

from classes import Universe

if __name__ == '__main__':
    num_bodies = 2
    size = 2
    G = 1

    universe = Universe(num_bodies, size, G)

    universe.body_x[0] = np.array([-1,0])
    universe.body_x[1] = np.array([1, 0])
    universe.body_v[0] = np.array([1,0])
    universe.body_v[1] = np.array([-1,0])

    fig, ax = plt.subplots()
    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)
    ax.set_aspect('equal')

    points = ax.plot(universe.body_x[:,0], universe.body_x[:,1], 'o')

    def animate(i):
        universe.update_positions()

        points.set_data(universe.body_x[:,0], universe.body_x[:,1])

        return points

    anim = FuncAnimation(fig, animate, frames=100, blit=True)

    plt.show()