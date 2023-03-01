import numpy as np

class Universe():
    """Wrapper class for all items to be simulated
    """

    def __init__(self,
                 num_bodies,
                 size = 100,
                 dt = 1,
                 G = 1,
                ) -> None:
        self.size = size
        self.dt = dt
        self.G = G
        self.body_x = np.zeros((num_bodies, 2))
        self.body_v = np.zeros((num_bodies, 2))

    def update_positions(self):
        self.body_x += self.dt * self.body_v


class Body():
    """Body class for all objects
    """

    def __init__(self,
                 x = (0,0),
                 v = (0,0),
                 m = 1,
                ) -> None:
        self.x = x
        self.v = v
        self.a = (0,0)
        self.m = m

    def update_position():
        pass