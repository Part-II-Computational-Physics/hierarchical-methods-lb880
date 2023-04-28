import numpy as np
import matplotlib.pyplot as plt

def plot(particles):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    points = [particle.centre for particle in particles]
    X,Y = np.real(points), np.imag(points)

    ax.scatter(X,Y)

    return fig
