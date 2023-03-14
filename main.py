import matplotlib

matplotlib.use('TkAgg')

from universe import Universe
from animator import Animator
import pairwise

def main():
    num_bodies = 10
    size = 1
    G = .1
    dt = 0.01
    softening = 0.05

    universe = Universe(num_bodies, size, dt, G, softening)

    universe.initialise_bodies('circle')

    animator = Animator(universe, pairwise.calculate_accelerations)
    animator.create_figure_for_animation()
    animator.produce_animation(True)

if __name__ == '__main__':
    main()
