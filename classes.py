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
        self.num_bodies = num_bodies
        self.size = size
        self.dt = dt
        self.G = G
        self.body_x = np.zeros((num_bodies, 2))
        self.body_v = np.zeros((num_bodies, 2))
        self.body_a = np.zeros((num_bodies, 2))
        self.body_m = np.ones(num_bodies)

    def update_positions(self):
        self.calculate_accelerations()

        self.body_v += self.dt * self.body_a
        self.body_x += self.dt * self.body_v

    def calculate_accelerations(self):
        self.body_a = np.zeros((self.num_bodies, 2))

        for i,x1 in enumerate(self.body_x[:-1]):
            for j,x2 in enumerate(self.body_x[i+1:], i+1):
                # r points object 1 to object 2
                r = x1 - x2
                mag_r = np.abs(r)
                norm_r = r / mag_r
                mag_F = self.G / (mag_r**2) # can later multiply in masses
                # force felt by 1 points at 2
                F = - mag_F * norm_r
                self.body_a[i] += F * self.body_m[j]
                self.body_a[j] -= F * self.body_m[i]

        print(self.body_a)
    