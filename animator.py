import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from universe import Universe

class Animator():
    def __init__(self,
                 universe: Universe,
                ) -> None:
        self.universe = universe

    def create_figure_basic(self) -> None:
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-self.universe.size, self.universe.size)
        self.ax.set_ylim(-self.universe.size, self.universe.size)
        self.ax.set_aspect('equal')

        self.points, = self.ax.plot(self.universe.body_x[:,0],
                                    self.universe.body_x[:,1],
                                    'o')
    
    def animate_function_basic(self, i):
        self.universe.update_positions()

        self.points.set_data(self.universe.body_x[:,0],
                             self.universe.body_x[:,1])

        return self.points,

    def produce_animation_basic(self):
        self.anim = FuncAnimation(self.fig,
                                  self.animate_function_basic,
                                  100,
                                  interval=5*self.universe.dt,
                                  blit=True)
        
        plt.show()

    def animate_function_with_momentum_energy(self, i):
        self.universe.update_positions()

        self.universe.calculate_system_momentum()
        self.universe.calculate_system_kinetic_energy()
        self.universe.calculate_system_potential_energy()

        self.points.set_data(self.universe.body_x[:,0],
                             self.universe.body_x[:,1])

        return self.points,
    
    def produce_animation_with_momentum_energy(self):
        self.anim = FuncAnimation(self.fig,
                                  self.animate_function_with_momentum_energy,
                                  100,
                                  interval=5*self.universe.dt,
                                  blit=True)
        
        plt.show()

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
