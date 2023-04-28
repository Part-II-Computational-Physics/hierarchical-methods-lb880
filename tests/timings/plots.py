import numpy as np
import matplotlib.pyplot as plt
from . import fits

def general(FILE_PATH):
    with np.load(FILE_PATH) as data:
        terms = data['terms']
        theta = data['theta']
        max_time = data['max_time']

        pair_averages = data['pair_averages']
        pair_stdevs = data['pair_stdevs']
        bh_averages = data['bh_averages']
        bh_stdevs = data['bh_stdevs']
        fmm_averages = data['fmm_averages']
        fmm_stdevs = data['fmm_stdevs']
        particle_numbers = data['particle_numbers']

    fig, ax = plt.subplots()

    ax.errorbar(particle_numbers[:len(pair_averages)], pair_averages, 
                pair_stdevs, capsize=3, label='Pairwise')
    ax.errorbar(particle_numbers[:len(bh_averages)],   bh_averages,   
                bh_stdevs,   capsize=3, label='Barnes Hut')
    ax.errorbar(particle_numbers[:len(fmm_averages)],  fmm_averages,  
                fmm_stdevs,  capsize=3, label='FMM')

    # ax.plot(particle_numbers, particle_numbers*10**-3.7)

    ax.set_title(f'Runtimes for the different methods. '
                 f'terms={terms}, theta={theta}', wrap=True)

    # ax.set_aspect('equal')
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Number of Particles')
    ax.set_ylabel('Runtime in Seconds')
    ax.legend()

    return fig


def max_level(FILE_PATH):
    with np.load(FILE_PATH) as data:
        normal_max = data['normal_max']
        num_particles = data['num_particles']
        terms = data['terms']

        averages = data['averages']
        stdevs = data['stdevs']
        max_levels = data['max_levels']

    fig, ax = plt.subplots()

    ax.errorbar(max_levels, averages, stdevs, fmt=':_', capsize=3)

    ax.set_title('Times for one more and one less level than default. '
                 f'num_particles={num_particles}, normal_max={normal_max}, terms={terms}', wrap=True)

    ax.set_xlabel('Max Level')
    ax.set_ylabel('Runtime in Seconds')

    return fig

    
def theta(FILE_PATH):
    with np.load(FILE_PATH) as data:
        num_particles = data['num_particles']
        terms = data['terms']

        averages = data['averages']
        stdevs = data['stdevs']
        thetas = data['thetas']

    fig, ax = plt.subplots()

    ax.errorbar(thetas, averages, stdevs, fmt=':_', capsize=3)

    ax.set_title('Times for varying thetas. '
                 f'num_particles={num_particles}, terms={terms}', wrap=True)

    ax.set_xlabel('Theta')
    ax.set_ylabel('Runtime in Seconds')

    return fig


def n_crit(FILE_PATH):
    with np.load(FILE_PATH) as data:
        num_particles = data['num_particles']
        theta = data['theta']
        terms = data['terms']

        averages = data['averages']
        stdevs = data['stdevs']
        n_crits = data['n_crits']

    fig, ax = plt.subplots()

    ax.errorbar(n_crits, averages, stdevs, fmt=':_', capsize=3)

    ax.set_title('Times for varying n_crit. '
                 f'num_particles={num_particles}, theta={theta}, terms={terms}', wrap=True)

    ax.set_xlabel('n_crit')
    ax.set_ylabel('Runtime in Seconds')

    return fig

    
def terms(FILE_PATH):
    with np.load(FILE_PATH) as data:
        num_particles = data['num_particles']
        theta = data['theta']
        terms = data['terms']

        bh_averages = data['bh_averages']
        bh_stdevs = data['bh_stdevs']
        fmm_averages = data['fmm_averages']
        fmm_stdevs = data['fmm_stdevs']
        terms_vals = np.array(data['terms_vals'])

    fig, axs = plt.subplots(2, sharex='all')
    ax1, ax2 = axs

    ax1.errorbar(terms_vals, bh_averages, bh_stdevs, fmt='_', capsize=3, label='BH')
    ax2.errorbar(terms_vals, fmm_averages, fmm_stdevs, fmt='_', capsize=3, label='FMM')

    fmm_params, fmm_fit = fits.fmm_terms(FILE_PATH)
    ax2.plot(terms_vals, fmm_fit, '-')

    fig.suptitle('Times for varying terms. '
                 f'num_particles={num_particles}, theta={theta}, terms={terms}', wrap=True)

    ax2.set_xlabel('Terms')
    ax2.set_xticks(np.linspace(terms_vals[0], terms_vals[-1], 5, endpoint=True, dtype=int))
    fig.supylabel('Runtime in Seconds')

    ax1.set_title('BH')
    ax2.set_title('FMM')

    fig.tight_layout()

    return fig
