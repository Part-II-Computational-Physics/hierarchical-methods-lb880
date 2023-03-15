import numpy as np
import random

from typing import Callable

class Universe():
    """Wrapper class for simulated system

    Attributes
    ----------
    num_bodies
        Number of simulated bodies
    size
        xlim and ylim right, from 0 to size in each axis
    dt
        Timestep
    G
        Gravitational constant
    softening
        Factor added in quadrature to denom of force calculation to avoid singulrities
    
    positions, velocities, masses
        Arrays of body positions, velocites, etc.
    momentum, kinetic_energy, potential_energy
        Lists of these values computed on each frame

    Methods
    -------
    initialise_bodies

    update_positions
    
    accelerations

    calculate_system_...
        Calc the various properies
    """

    def __init__(self,
                 num_bodies,
                 size = 1,
                 dt = 1,
                 G = 1,
                 softening = 0.01,
                ) -> None:
        self.dt = dt

        self.properties = {
            'num_bodies': num_bodies,
            'size': size,
            'G': G,
            'softening': softening
        }

        self.positions = np.zeros((num_bodies, 2))
        self.velocities = np.zeros((num_bodies, 2))
        self.masses = np.ones(num_bodies).reshape((-1,1)) # make column vector

        self.momentum = []
        self.kinetic_energy = []
        self.potential_energy = []

    def initialise_bodies(self, setup:str) -> None:
        """Define the initial particle positions and velocites

        Parameters
        ----------
        setup : str
            'random'
                randomised positions and velocites
            'circle'
                start in a circle traveling anticlockwise
            'orbit
                one central heavy mass, with the remaining masses set to orbit round
        """
        if setup == 'random':
            for i in range(self.properties['num_bodies']):
                r = random.uniform(0,self.properties['size']/3)
                theta = random.uniform(0,2*np.pi)
                self.positions[i] = r * np.array([np.cos(theta),np.sin(theta)]) + np.full(2, self.properties['size']/2)
                self.velocities[i] = np.sqrt(self.properties['num_bodies'] * self.properties['G'] / r) * np.array([-np.sin(theta),np.cos(theta)])
        elif setup == 'circle':
            r = self.properties['size']/4
            v = 0.3 * np.sqrt(self.properties['G'] * (self.properties['num_bodies']-1) / r)
            for i in range(self.properties['num_bodies']):
                theta = (i/self.properties['num_bodies']) * 2 * np.pi
                self.positions[i] = r * np.array([np.cos(theta),np.sin(theta)]) + np.full(2, self.properties['size']/2)
                self.velocities[i] = v * np.array([-np.sin(theta),np.cos(theta)])
        elif setup == 'orbital':
            M = 10000
            self.masses[0] = 10000
            self.positions[0] = np.full(2, self.properties['size']/2)
            for i in range(1, self.properties['num_bodies']):
                r = random.uniform(0.6,1.5)
                theta = random.uniform(0,2*np.pi)
                self.positions[i] = r * np.array([np.cos(theta),np.sin(theta)]) + np.full(2, self.properties['size']/2)
                self.velocities[i] = np.sqrt(self.properties['G'] * M / r) * np.array([-np.sin(theta),np.cos(theta)])
        else:
            raise ValueError('Not a valid initial position setup')
        
        # Record initial values of momentum and energy
        self.calculate_system_momentum()
        self.calculate_system_kinetic_energy()
        self.calculate_system_potential_energy()

    def update_positions_RK4(self, accelerations:Callable) -> None:
        """Update the positions of all bodies in the universe using RK4 integration

        masses, positions is the x_n v_n respective

        Parameters
        ----------
        accelerations : Callable
            Call signature of (properties, masses, positions)
        """

        # calculate intermediates
        k1v = accelerations(self.properties, self.masses, self.positions) * self.dt
        k1x = self.velocities * self.dt

        k2v = accelerations(self.properties, self.masses, self.positions + k1x/2) * self.dt
        k2x = (self.velocities + k1v/2) * self.dt

        k3v = accelerations(self.properties, self.masses, self.positions + k2x/2) * self.dt
        k3x = (self.velocities + k2v/2) * self.dt

        k4v = accelerations(self.properties, self.masses, self.positions + k3x) * self.dt
        k4x = (self.velocities + k3v) * self.dt

        # update to next values
        self.velocities += (1/6) * (k1v + 2*k2v + 2*k3v + k4v)
        self.positions += (1/6) * (k1x + 2*k2x + 2*k3x + k4x)

    def update_positions_euler(self, accelerations:Callable) -> None:
        """Update the positions of all bodies in the universe using Euler integration

        masses, positions is the x_n v_n respective

        Parameters
        ----------
        accelerations : Callable
            Call signature of (properties, masses, positions)
        """
        self.velocities += self.dt * accelerations(self.properties, self.masses, self.positions)
        self.positions += self.dt * self.velocities

    def calculate_system_momentum(self) -> None:
        """Calculate total momentum of system and add to class list"""

        momenta_array = self.velocities * self.masses
        total_momentum = np.sum(momenta_array)

        self.momentum.append(total_momentum)

    def calculate_system_kinetic_energy(self) -> None:
        """Calculate total kinetic energy of system and add to class list"""
        kinetic_array = 0.5 * self.masses * \
                        np.linalg.norm(self.velocities, axis=1).reshape((-1,1))**2
        total_kinetic_energy = np.sum(kinetic_array)

        self.kinetic_energy.append(total_kinetic_energy)

    def calculate_system_potential_energy(self) -> None:
        """Calculate total potential energy of system and add to class list
        
        Uses numpy functionality
        """

        body_x = self.positions
        body_m = self.masses

        total_potential_energy = 0

        for i, pos in enumerate(body_x):
            r = pos - body_x[1:]
            mag_r = np.linalg.norm(r, axis=1).reshape((-1,1))
            potential = - self.properties['G'] * body_m[0] * body_m[1:] / mag_r

            total_potential_energy += 0.5 * np.sum(potential)

            body_x = np.roll(body_x, -1, axis=0)
            body_m = np.roll(body_m, -1, axis=0)

        self.potential_energy.append(total_potential_energy)
    