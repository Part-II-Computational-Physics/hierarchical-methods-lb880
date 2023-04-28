from typing import List
from numpy.typing import NDArray

import numpy as np

from packages import general
from packages.general import Particle

class Universe():

    def __init__(self, method, dt: float, k: float,
                 masses: NDArray = None) -> None:
        """Defaults to all masses 1 if no argument supplied"""
        self.method = method
        self.dt = dt
        self.k = k

        self.particles: List[Particle] = method.particles
        self.charges = np.array([p.charge for p in self.particles])
        if np.any(masses) != None:
            assert len(masses) == len(self.particles)
            self.masses = masses
        else:
            self.masses = np.ones_like(self.charges)
        self.charge_per_mass = self.charges / self.masses

        centres = np.array([particle.centre for particle in self.particles])
        self.positions = np.array([centres.real, centres.imag]).transpose()
        self.velocities = np.zeros_like(self.positions)

    def get_particle_accelerations(self) -> NDArray:
        """Get particle accelerations for their current positions"""
        self.method.do_method()
        force_pers = self.k * np.array([particle.force_per for particle in self.particles])
        accelerations =  np.multiply(force_pers, self.charge_per_mass[:,np.newaxis])

        return accelerations
    
    def set_particle_positions(self) -> None:
        """Sets particle positions to those in `self.positions`"""
        for particle, update in zip(self.particles, self.positions):
            particle.centre = complex(update[0], update[1])
    
    def update_particle_positions(self, pos_update: NDArray) -> None:
        """Update all particles with `pos_update`. Additive to `self.positions`"""
        new_positions = self.box_confine(self.positions + pos_update)
        for particle, update in zip(self.particles, new_positions):
            particle.centre = complex(update[0], update[1])
    
    def store_positions_velocities(self, pos_update, vel_update):
        self.positions += pos_update
        self.velocities += vel_update

        self._box_confine()
        self.set_particle_positions()

    def _box_confine(self) -> None:
        self.positions = self.positions % 2
        for i, pos in enumerate(self.positions):
            if pos[0] > 1:
                self.positions[i, 0] = 2 - pos[0]
                self.velocities[i, 0] *= -1
            if pos[1] > 1:
                self.positions[i, 1] = 2 - pos[1]
                self.velocities[i, 1] *= -1
    
    @staticmethod
    def box_confine(positions) -> None:
        """Just confines the given positions to a box"""
        positions = positions % 2
        for i, pos in enumerate(positions):
            if pos[0] > 1:
                positions[i, 0] = 2 - pos[0]
            if pos[1] > 1:
                positions[i, 1] = 2 - pos[1]
        return positions

    def update_system_euler(self) -> None:
        """First test to get updates working"""
        self.velocities += self.get_particle_accelerations() * self.dt
        self.positions = self.positions + self.velocities * self.dt
        self._box_confine()
        self.set_particle_positions()

    def update_system_RK4(self) -> None:
        """Update and store all positions and velocities"""

        # calculate intermediates
        k1v = self.get_particle_accelerations() * self.dt
        k1x = self.velocities * self.dt
        
        self.update_particle_positions(k1x/2)
        k2v = self.get_particle_accelerations() * self.dt
        k2x = (self.velocities + k1v/2) * self.dt

        self.update_particle_positions(k2x/2)
        k3v = self.get_particle_accelerations() * self.dt
        k3x = (self.velocities + k2v/2) * self.dt

        self.update_particle_positions(k3x)
        k4v = self.get_particle_accelerations() * self.dt
        k4x = (self.velocities + k3v) * self.dt

        # update to next values
        # self.positions += (1/6) * (k1x + 2*k2x + 2*k3x + k4x)
        # self.velocities += (1/6) * (k1v + 2*k2v + 2*k3v + k4v)
        self.store_positions_velocities(
            (1/6) * (k1x + 2*k2x + 2*k3x + k4x),
            (1/6) * (k1v + 2*k2v + 2*k3v + k4v)
        )

    def kinetic_energy(self):
        """Currently for all unit mass"""
        return 0.5 * np.sum(np.linalg.norm(self.velocities, axis=1)**2)
    
    def potential_energy(self):
        # general.Pairwise.potentials(self.particles)
        return 0.5 * self.k * np.sum([p.charge * p.potential for p in self.particles])