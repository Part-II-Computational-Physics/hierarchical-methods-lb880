import numpy as np
import matplotlib.pyplot as plt

def plot(particles):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    points = [particle.centre for particle in particles]
    X,Y = np.real(points), np.imag(points)

    ax.scatter(X,Y)

    return fig

def bh_com_error(bh_com_pots, pair_pots):
    bh_com_pots_fracs = abs(bh_com_pots - pair_pots) / pair_pots

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey='all', gridspec_kw={'width_ratios':[2,1]})

    ax1.plot(bh_com_pots_fracs, 'o', alpha=0.5)
    ax2.boxplot(bh_com_pots_fracs, notch=True, whis=(0,100))

    fig.suptitle('BH CoM Method')

    ax1.set_yscale('log')
    ax1.set_ylabel('Fractional Errors')
    ax1.set_xlabel('Particle')
    ax1.set_title('Individual errors')
    ax2.set_title('All particles')
    ax2.set_xticks([])

def bh_multi_error(bh_multi_pots, pair_pots, terms):
    bh_multi_pots_fracs = abs(bh_multi_pots - pair_pots) / pair_pots

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey='all', gridspec_kw={'width_ratios':[2,1]})

    ax1.plot(bh_multi_pots_fracs, 'o', alpha=0.5)
    ax2.boxplot(bh_multi_pots_fracs, notch=True, whis=(0,100))

    fig.suptitle(f'BH Multipole method. {terms} terms. ')

    ax1.set_yscale('log')
    ax1.set_ylabel('Fractional Errors')
    ax1.set_xlabel('Particle')
    ax1.set_title('Individual errors')
    ax2.set_title('All particles')
    ax2.set_xticks([])

def fmm_error(fmm_pots, pair_pots, terms):
    fmm_pots_fracs = abs(fmm_pots - pair_pots) / pair_pots

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey='all', gridspec_kw={'width_ratios':[2,1]})

    ax1.plot(fmm_pots_fracs, 'o', alpha=0.5)
    ax2.boxplot(fmm_pots_fracs, notch=True, whis=(0,100))

    fig.suptitle(f'FMM method. {terms} terms. ')

    ax1.set_yscale('log')
    ax1.set_ylabel('Fractional Errors')
    ax1.set_xlabel('Particle')
    ax1.set_title('Individual errors')
    ax2.set_title('All particles')
    ax2.set_xticks([])

    return fig
