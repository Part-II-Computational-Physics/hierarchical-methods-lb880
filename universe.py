import numpy as np
import random

class Universe():
    """Wrapper class for all items to be simulated
    """

    def __init__(self,
                 num_bodies,
                 size = 100,
                 dt = 1,
                 G = 1,
                 softening = 0.01,
                ) -> None:
        self.num_bodies = num_bodies
        self.size = size
        self.dt = dt
        self.G = G
        self.softening = softening

        self.body_x = np.zeros((num_bodies, 2))
        self.body_v = np.zeros((num_bodies, 2))
        self.body_a = np.zeros((num_bodies, 2))
        self.body_m = np.ones(num_bodies).reshape((-1,1)) # make column vector

        self.momentum = []
        self.kinetic_energy = []
        self.potential_energy = []

    def initialise_positions_velocities(self, setup:str) -> None:
        """Define the initial particle positions and velocites

        Parameters
        ----------
        setup : str
            'random'
                randomised positions and velocites
            'circle'
                start in a circle traveling anticlockwise
        """
        if setup == 'random':
            for i in range(self.num_bodies):
                r = random.uniform(0,1)
                theta = random.uniform(0,2*np.pi)
                self.body_x[i] = r * np.array([np.cos(theta),np.sin(theta)])
                self.body_v[i] = 1 * np.sqrt(self.num_bodies * self.G * r) * np.array([-np.sin(theta),np.cos(theta)])
        elif setup == 'circle':
            r = 1
            v = 0.3 * np.sqrt(self.G * (self.num_bodies-1) / r)
            for i in range(self.num_bodies):
                theta = (i/self.num_bodies) * 2 * np.pi
                self.body_x[i] = r * np.array([np.cos(theta),np.sin(theta)])
                self.body_v[i] = v * np.array([-np.sin(theta),np.cos(theta)])
        else:
            raise ValueError('Not a valid initial position setup')
        
        # Record initial values of momentum and energy
        self.calculate_system_momentum()
        self.calculate_system_kinetic_energy()
        self.calculate_system_potential_energy()

    def update_positions(self) -> None:
        self.calculate_accelerations_np_naive()

        self.body_v += self.dt * self.body_a
        self.body_x += self.dt * self.body_v

    def update_positions_RK4(self) -> None:
        # Using RK4 integration
        # self.body_x/v is the x_n v_n respective

        # calculate intermediates
        k1v = self.calculate_accelerations_with_return(self.body_x) * self.dt
        k1x = self.body_v * self.dt

        k2v = self.calculate_accelerations_with_return(self.body_x + k1x/2) * self.dt
        k2x = (self.body_v + k1v/2) * self.dt

        k3v = self.calculate_accelerations_with_return(self.body_x + k2x/2) * self.dt
        k3x = (self.body_v + k2v/2) * self.dt

        k4v = self.calculate_accelerations_with_return(self.body_x + k3x) * self.dt
        k4x = (self.body_v + k3v) * self.dt

        # update to next values
        self.body_v += 0.5 * (1/6)*(k1v + 2*k2v + 2*k3v + k4v)
        self.body_x += 0.5 * (1/6)*(k1x + 2*k2x + 2*k3x + k4x)

    def calculate_accelerations_with_return(self, body_x):
        body_a = np.zeros((self.num_bodies, 2))

        for i,position1 in enumerate(body_x[:-1]):
            for j,position2 in enumerate(body_x[i+1:], i+1):
                # r points object 1 to object 2
                r = position1 - position2
                mag_r = np.linalg.norm(r)
                dir_r = r / mag_r
                # force felt by 1 points at 2
                # can later multiply in masses
                # softening factor ensures that distance is never close to zero => inverse finite
                F = - (self.G * dir_r) / np.maximum(mag_r,self.softening)**2
                # calculate accelerations
                body_a[i] += F * self.body_m[j]
                body_a[j] -= F * self.body_m[i]

        return body_a

    def calculate_accelerations(self) -> None:
        self.body_a = np.zeros((self.num_bodies, 2))

        for i,position1 in enumerate(self.body_x[:-1]):
            for j,position2 in enumerate(self.body_x[i+1:], i+1):
                # r points object 1 to object 2
                r = position1 - position2
                mag_r = np.linalg.norm(r)
                dir_r = r / mag_r
                # force felt by 1 points at 2
                # can later multiply in masses
                # softening factor ensures that distance is never close to zero => inverse finite
                F = - (self.G * dir_r) / (mag_r**2 + self.softening**2)
                # calculate accelerations
                self.body_a[i] += F * self.body_m[j]
                self.body_a[j] -= F * self.body_m[i]

    def calculate_accelerations_np_naive(self) -> None:
        self.body_a = np.zeros((self.num_bodies,2))

        body_x = self.body_x
        body_m = self.body_m

        for i, pos in enumerate(body_x):
            # points towards current body
            r = pos - body_x[1:]
            mag_r = np.linalg.norm(r, axis=1).reshape((-1,1))
            dir_r = r / mag_r
            # force felt by current body
            # can later multiply in masses
            # softening factor ensures that distance is never close to zero => inverse finite
            F = - (self.G * dir_r) / (mag_r**2 + self.softening**2)
            # calculate accelerations
            self.body_a[i] = np.sum(F * body_m[1:], axis=0)
            # roll on the body_x
            body_x = np.roll(body_x, -1, axis=0)
            body_m = np.roll(body_m, -1, axis=0)

    def calculate_system_momentum(self) -> None:
        momenta_array = self.body_v * self.body_m
        total_momentum = np.sum(momenta_array)

        self.momentum.append(total_momentum)

    def calculate_system_kinetic_energy(self) -> None:
        kinetic_array = 0.5 * self.body_m * \
                        np.linalg.norm(self.body_v, axis=1).reshape((-1,1))**2
        total_kinetic_energy = np.sum(kinetic_array)

        self.kinetic_energy.append(total_kinetic_energy)

    def calculate_system_potential_energy(self) -> None:
        """Calculate the systems potential energy at a point in time
            Then add value to potential energy array
        
        Does so through sum of all pairwise potentials"""

        total_potential_energy = 0

        for i,position1 in enumerate(self.body_x[:-1]):
            for j,position2 in enumerate(self.body_x[i+1:], i+1):
                potential = - self.G * self.body_m[i] * self.body_m[j] / \
                                np.linalg.norm(position1 - position2)
                
                total_potential_energy += potential


        # body_x = self.body_x
        # body_m = self.body_m

        # for i, pos in enumerate(body_x):
        #     r = pos - body_x[1:]
        #     mag_r = np.linalg.norm(r, axis=1).reshape((-1,1))
        #     potential = - self.G * body_m[0] * body_m[1:] / mag_r

        #     total_potential_energy += 0.5 * np.sum(potential)

        #     body_x = np.roll(body_x, -1, axis=0)
        #     body_m = np.roll(body_m, -1, axis=0)

        self.potential_energy.append(total_potential_energy)

    