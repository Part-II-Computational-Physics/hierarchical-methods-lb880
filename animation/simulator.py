import numpy as np
import matplotlib.pyplot as plt

import packages
    
np.random.seed(0)

dt = 0.001
k=1

num_protons_side = 32
num_protons = num_protons_side ** 2

protons = [packages.general.Particle(1) for _ in range(num_protons)]
electrons = [packages.general.Particle(-1) for _ in range(num_protons)]

hack_finest_level = packages.fmm.FinestLevel(int(np.log2(num_protons_side)), 1)
centres = hack_finest_level.array[:,:,0].reshape(-1)
half_box_size = centres[0].real
for p, c in zip(protons, centres):
    noise_x = half_box_size * np.random.random() - 0.5*half_box_size
    noise_y = half_box_size * np.random.random() - 0.5*half_box_size
    p.centre = c + complex(noise_x, noise_y)

particles = protons + electrons

masses = np.ones_like(particles, dtype=float)
masses[:masses.size//2] *= 1000

pair_method = packages.general.Pairwise(particles)
bh_method = packages.bh.BH(particles, 0.5, terms=3)
fmm_method = packages.fmm.FMM(particles, 4)

universe = packages.animation.Universe(fmm_method, dt, k, masses=masses)

charges = [p.charge for p in universe.particles]

pots = []
kins = []

frames = int(input('Frames: '))

positions_store = np.zeros((frames+1, len(particles), 2))

for i in range(frames+1):
    through = (10*i)//frames
    print(i, '='*through + '-'*(10-through), end='\r')
    universe.update_system_RK4()
    positions_store[i] = universe.positions
    pots.append(universe.potential_energy())
    kins.append(universe.kinetic_energy())


FILE_PATH = 'sim_data_local/106.npz'
pots_arr, kins_arr = np.array(pots), np.array(kins)
np.savez(FILE_PATH, positions=positions_store, pots=pots_arr, kins=kins_arr, charges=charges)

with np.load(FILE_PATH) as sim_data:
    positions = sim_data['positions']
    pots = sim_data['pots']
    kins = sim_data['kins']

fig, ax = plt.subplots()
ax.plot(pots_arr, label='pots')
ax.plot(kins_arr, label='kins')
totals = kins_arr + pots_arr
ax.plot(totals, label='totals')
fig.legend()

plt.show()
