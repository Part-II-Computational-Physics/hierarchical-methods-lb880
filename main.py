import matplotlib

matplotlib.use('TkAgg')

from universe import Universe
from animator import Animator

def main():
    num_bodies = 10
    size = 2
    G = .1
    dt = 0.01
    softening = 0.1

    universe = Universe(num_bodies, size, dt, G, softening)

    universe.initialise_bodies('circle')

    animator = Animator(universe)
    animator.create_figure_for_animation()
    animator.produce_animation_with_momentum_energy()

if __name__ == '__main__':
    main()
