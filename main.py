import matplotlib

matplotlib.use('TkAgg')

from universe import Universe
from animator import Animator
import pairwise

def main():
    num_bodies = 200
    size = 1
    G = .00001
    dt = 0.01
    softening = 0.05

    universe = Universe(num_bodies, size, dt, G, softening)

    universe.initialise_bodies('orbital')

    animator = Animator(universe, pairwise.calculate_accelerations_np)
    animator.create_figure_for_animation()
    animator.produce_animation(True)

if __name__ == '__main__':
    main()
