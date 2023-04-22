import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import packages

# def animate(i):
#     print(i*10)
#     for _ in range(10):
#         universe.update_system_RK4()

#         x = [p.centre.real for p in universe.particles]
#         y = [p.centre.imag for p in universe.particles]

#         pots.append(universe.potential_energy())
#         kins.append(universe.kinetic_energy())
#     scatter.set_offsets(np.transpose((x,y)))
#     return scatter,
    
np.random.seed(0)

dt = 0.0001
k=1
num_protons_side = 16
num_protons = num_protons_side ** 2

protons = [packages.general.Particle(1) for _ in range(num_protons)]
electrons = [packages.general.Particle(-1) for _ in range(num_protons)]

hack_finest_level = packages.fmm.FinestLevel(int(np.log2(num_protons_side)), 1)
centres = hack_finest_level.array[:,:,0].reshape(-1)
for p, c in zip(protons, centres):
    p.centre = c

particles = protons + electrons

masses = np.ones_like(particles, dtype=float)
masses[:masses.size//2] *= 1000

pair_method = packages.general.Pairwise(particles)
bh_method = packages.bh.BH(particles, 0.5, terms=3)
fmm_method = packages.fmm.FMM(particles, 10)
universe = packages.animation.Universe(bh_method, dt, k, masses=masses)


charges = [p.charge for p in universe.particles]
# x = [p.centre.real for p in universe.particles]
# y = [p.centre.imag for p in universe.particles]
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
# scatter = ax.scatter(x,y, c=charges, cmap='plasma')

pots = []
kins = []

# anim = FuncAnimation(fig, animate, 100, blit=True)
# plt.show()

frames = int(input('Frames: '))

positions_store = np.zeros((frames+1, len(particles), 2))

for i in range(frames+1):
    through = (10*i)//frames
    print(i, '='*through + '-'*(10-through), end='\r')
    universe.update_system_RK4()
    positions_store[i] = universe.positions
    pots.append(universe.potential_energy())
    kins.append(universe.kinetic_energy())

FILE_PATH = 'sim_data_local/006.npz'
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
