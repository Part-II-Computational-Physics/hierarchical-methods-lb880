import matplotlib

matplotlib.use('TkAgg')

from universe import Universe
from animator import Animator
import pairwise
import barnes_hut

def main():
    num_bodies = 100
    size = 1
    G = .0001
    dt = 0.01
    softening = 0.05

    universe = Universe(num_bodies, size, dt, G, softening, theta=0.5)

    universe.initialise_bodies('orbital')

    animator = Animator(universe, barnes_hut.calculate_accelerations)
    animator.create_figure_for_animation()
    animator.produce_animation(
        with_momentum_energy=True,
        draw_barnes_hut=False,
    )

if __name__ == '__main__':
    main()
