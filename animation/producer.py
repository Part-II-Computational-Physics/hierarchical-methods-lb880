import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

FILE_PATH = 'sim_data_local/105.npz'
up_to_frame: int = -1

print(f'Loading {FILE_PATH}...')
with np.load(FILE_PATH) as sim_data:
    positions = sim_data['positions'][:up_to_frame]
    pots = sim_data['pots'][:up_to_frame]
    kins = sim_data['kins'][:up_to_frame]
    charges= sim_data['charges']
print(f'{FILE_PATH} Loaded')

totals = kins + pots
max_val = max(np.max(pots), np.max(kins), np.max(totals))
min_val = min(np.min(pots), np.min(kins), np.min(totals))

def animate(i):
    i *= 1
    scatter.set_offsets(positions[i,:,:])
    pot_points.set_data(frame_nums[:i], pots[:i])
    kin_points.set_data(frame_nums[:i], kins[:i])
    tot_points.set_data(frame_nums[:i], totals[:i])

    return scatter, pot_points, kin_points, tot_points,

fig, (sim, energy) = plt.subplots(1,2)

sim.set_aspect('equal')
sim.set_xlim(0,1)
sim.set_ylim(0,1)
scatter = sim.scatter(positions[0,:,0],positions[0,:,1], marker='.', c=charges, cmap='plasma')

frame_nums = np.arange(len(totals))
energy.set_xlim(0, frame_nums[-1])
energy.set_ylim(min_val, max_val)
pot_points, = energy.plot(pots[0], label='Potential')
kin_points, = energy.plot(kins[0], label='Kinetic')
tot_points, = energy.plot(totals[0], label='Total')
fig.legend()
fig.suptitle(r'Lattice like, FMM, dt=0.001, k=1, $2 \times 32^2$ particles')

anim = FuncAnimation(fig, animate, len(totals)//1, blit=True, interval=10, repeat=True)
# SAVE_PATH = 'sim_data_local/106.mp4'
# writervideo = animation.FFMpegWriter(fps=24, bitrate=6000)
# print(f'Saving animation as {SAVE_PATH}')
# anim.save(SAVE_PATH, writer=writervideo, )
# print('Animation saved')
plt.show()
