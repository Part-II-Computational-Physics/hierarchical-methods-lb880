import math
import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

def linear(x, p, c):
    return p*x + c

def power_law(N, p, A):
    return A * N**p

def NlogN(N, A):
    return A * N * np.log(N)

def lin_with_log(N, a, b, c):
    return a*N + b*np.log(N) + c


def general(FILE_PATH, truncation=5):
    with np.load(FILE_PATH) as data:
        pair_averages = data['pair_averages']
        pair_stdevs = data['pair_stdevs']
        bh_averages = data['bh_averages']
        bh_stdevs = data['bh_stdevs']
        fmm_averages = data['fmm_averages']
        fmm_stdevs = data['fmm_stdevs']
        particle_numbers = data['particle_numbers']

    pair_particles = particle_numbers[:len(pair_averages)]
    bh_particles = particle_numbers[:len(bh_averages)]
    fmm_particles = particle_numbers[:len(fmm_averages)]

    # use log-log scale (gives more accurate fit for the data)
    # x = log(particle_numbers), y = log(times)
    x_pair = np.log(pair_particles)
    y_pair = np.log(pair_averages)
    x_bh = np.log(bh_particles)
    y_bh = np.log(bh_averages)
    x_fmm = np.log(fmm_particles)
    y_fmm = np.log(fmm_averages)
    x_fmm_trun = np.log(fmm_particles[truncation:])
    y_fmm_trun = np.log(fmm_averages[truncation:])

    pair_params, pair_covar = curve_fit(linear, x_pair, y_pair)
    bh_params, bh_covar = curve_fit(linear, x_bh, y_bh)
    bh_NlogN_params, bh_NlogN_covar = curve_fit(NlogN, bh_particles, bh_averages)
    fmm_params, fmm_covar = curve_fit(linear, x_fmm, y_fmm)
    fmm_trun_params, fmm_trun_covar = curve_fit(linear, x_fmm_trun, y_fmm_trun)

    # from linear fit of y=px+c, have fit of T=e^c N^p, A = e^c
    # so p is the order
    print(f'Pair Order: {pair_params[0]}')
    print(f'BH Order:   {bh_params[0]}')
    print(f'FMM Order:  {fmm_params[0]}')
    print(f'FMM Trun:  {fmm_trun_params[0]}')

    pair_fit = power_law(pair_particles, pair_params[0], math.exp(pair_params[1]))
    bh_fit = power_law(bh_particles, bh_params[0], math.exp(bh_params[1]))
    bh_NlogN_fit = NlogN(bh_particles, *bh_NlogN_params)
    fmm_fit = power_law(fmm_particles, fmm_params[0], math.exp(fmm_params[1]))
    fmm_trun_fit = power_law(fmm_particles, fmm_trun_params[0], math.exp(fmm_trun_params[1]))

    fig, axs = plt.subplots(2, 2)
    (ax1, ax2), (ax3, ax4) = axs

    fig.suptitle(r'Power law fits for timing data. Fits of form $A N^p$. ', wrap=True)

    ax1.errorbar(pair_particles, pair_averages, pair_stdevs, fmt='_', capsize=3, label='Data')
    ax1.plot(pair_particles, pair_fit,           label='Fit')

    ax2.errorbar(bh_particles, bh_averages, bh_stdevs, fmt='_', capsize=3, label='Data')
    ax2.plot(bh_particles, bh_fit,           label='Fit')
    ax2.plot(bh_particles, bh_NlogN_fit,     label='NlogN')

    ax3.errorbar(fmm_particles, fmm_averages, fmm_stdevs, fmt='_', capsize=3, label='Data')
    ax3.plot(fmm_particles, fmm_fit,           label='Fit')

    ax4.errorbar(fmm_particles[truncation:], fmm_averages[truncation:], fmm_stdevs[truncation:],
                 fmt='_', capsize=3, label='Data')
    ax4.plot(fmm_particles,  fmm_trun_fit, label='Fit')
    ax4.errorbar(fmm_particles[:truncation],  fmm_averages[:truncation], fmm_stdevs[:truncation],
                 fmt='_r', capsize=3)

    methods = ['Pair', 'BH', 'FMM', f'FMM (from {truncation})']
    params = [pair_params, bh_params, fmm_params, fmm_trun_params]
    for i, ax in enumerate(axs.flat):
        ax.set_title(f'{methods[i]}: p={params[i][0]:.3}, A={math.exp(params[i][1]):.2e}', wrap=True)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()

    fig.tight_layout()

    return fig


fig = general('tests/timings/data/general.npz', 6)
plt.show()
