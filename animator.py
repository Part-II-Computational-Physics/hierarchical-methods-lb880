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
        