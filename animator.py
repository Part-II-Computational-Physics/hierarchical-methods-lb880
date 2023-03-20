import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from typing import Callable

from universe import Universe
import barnes_hut

class Animator():
    """Class to hold various animation routines
    
    Parameters
    ----------
    universe : Universe
        the universe to animate from
    accelerations : Callable
        Call signature of (properties, masses, positions)
    """

    def __init__(self,
                 universe: Universe,
                 accelerations: Callable
                ) -> None:
        self.universe = universe
        self.accelerations = accelerations

    def create_figure_for_animation(self) -> None:
        """Setup figure for animation"""
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.universe.properties['size'])
        self.ax.set_ylim(0, self.universe.properties['size'])
        self.ax.set_aspect('equal')

        self.points, = self.ax.plot(self.universe.positions[:,0],
                                    self.universe.positions[:,1],
                                    'o')

    def animate(self, i):
        self.universe.update_positions_RK4(self.accelerations)

        self.universe.calculate_system_momentum()
        self.universe.calculate_system_kinetic_energy()
        self.universe.calculate_system_potential_energy()

        self.points.set_data(self.universe.positions[:,0],
                             self.universe.positions[:,1])

        return self.points,

    def animate_draw_barnes_hut(self, i):
        patch_artists = barnes_hut.draw_rectangles(self.universe.properties, self.universe.positions, self.ax)

        self.universe.update_positions_RK4(self.accelerations)

        self.universe.calculate_system_momentum()
        self.universe.calculate_system_kinetic_energy()
        self.universe.calculate_system_potential_energy()

        self.points.set_data(self.universe.positions[:,0],
                             self.universe.positions[:,1])

        return self.points, *patch_artists, 
    
    def produce_animation(self,
                          with_momentum_energy:bool,
                          draw_barnes_hut:bool):
        """Produce animation, then plot p, KE, PE after"""

        if draw_barnes_hut:
            animate = self.animate_draw_barnes_hut
        else:
            animate = self.animate

        self.anim = FuncAnimation(self.fig,
                                  animate,
                                  100,
                                  interval=5*self.universe.dt,
                                  blit=True)
        
        plt.show()

        if with_momentum_energy:
            fig_plots, [ax_p, ax_E] = plt.subplots(2)

            ax_p.plot(self.universe.momentum)
            ax_p.set_ylabel('Momentum')
            
            total_energy = np.array(self.universe.kinetic_energy) + np.array(self.universe.potential_energy).reshape(-1)
            ax_E.plot(self.universe.kinetic_energy, label='Kinetic')
            ax_E.plot(self.universe.potential_energy, label='Potential')
            ax_E.plot(total_energy, label='Total')
            ax_E.set_ylabel('Energy')
            ax_E.set_xlabel('Frames')
            ax_E.legend()

            plt.show()
