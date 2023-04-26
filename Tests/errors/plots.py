import numpy as np
import matplotlib.pyplot as plt

def general(FILE_PATH):
    with np.load(FILE_PATH) as data:
        num_particles = data['num_particles']
        theta = data['theta']
        n_crit = data['n_crit']
        terms = data['terms']

        data_pots = list(data['data_pots'])
        data_mag = list(data['data_mag'])
        data_angle = list(data['data_angle'])
        x_labels = data['x_labels']

    fig, axs = plt.subplots(1, 3, sharex='all', sharey='all')
    ax1, ax2, ax3 = axs

    fig.suptitle(f'Errors across all methods for {num_particles} particles. '
                 f'theta={theta}, n_crit={n_crit}, terms={terms}', wrap=True)

    ax1.boxplot(data_pots,  notch=True, whis=(0,100))
    ax2.boxplot(data_mag,   notch=True, whis=(0,100))
    ax3.boxplot(data_angle, notch=True, whis=(0,100))

    full_x_labels = list(x_labels)*3

    ax1.set_yscale('log')
    ax1.set_ylabel('Fractional Error')
    ax1.set_xticks(ax1.get_xticks(), full_x_labels)
    for ax in axs.flat:
        ax.tick_params('x', labelrotation=30)
    ax1.set_title('Potential')
    ax2.set_title('Force Magnitude')
    ax3.set_title('Force Angle')

    return fig


def max_level(FILE_PATH):
    with np.load(FILE_PATH) as data:
        num_particles = data['num_particles']
        terms = data['terms']

        data_pots = list(data['data_pots'])
        data_mag = list(data['data_mag'])
        data_angle = list(data['data_angle'])
        x_labels = data['x_labels']

    fig, axs = plt.subplots(1, 3, sharex='all', sharey='all')
    ax1, ax2, ax3 = axs

    fig.suptitle(f'Errors for FMM (different max_level) for {num_particles} particles. terms={terms}')
    fig.supxlabel('Max Level')

    ax1.boxplot(data_pots,  notch=True, whis=(0,100))
    ax2.boxplot(data_mag,   notch=True, whis=(0,100))
    ax3.boxplot(data_angle, notch=True, whis=(0,100))

    full_x_labels = list(x_labels)*3

    ax1.set_yscale('log')
    ax1.set_ylabel('Fractional Error')
    ax1.set_xticks(ax1.get_xticks(), full_x_labels)
    ax1.set_title('Potential')
    ax2.set_title('Force Magnitude')
    ax3.set_title('Force Angle')

    return fig


def n_crit(FILE_PATH):
    with np.load(FILE_PATH) as data:
        num_particles = data['num_particles']
        theta = data['theta']
        terms = data['terms']

        data_CoM_pots = list(data['data_CoM_pots'])
        data_CoM_mag = list(data['data_CoM_mag'])
        data_CoM_angle = list(data['data_CoM_angle'])
        data_multi_pots = list(data['data_multi_pots'])
        data_multi_mag = list(data['data_multi_mag'])
        data_multi_angle = list(data['data_multi_angle'])
        x_labels = data['x_labels']

    fig, axs = plt.subplots(2, 3, sharey='all', sharex='all')
    (CoM_ax1, CoM_ax2, CoM_ax3), (multi_ax1, multi_ax2, multi_ax3) = axs

    fig.suptitle(f'Errors for CoM and multi BH (different n_crit) for {num_particles} particles. theta={theta}, terms={terms}')
    fig.supxlabel('n_crit')
    fig.supylabel('Fractional Error')

    CoM_ax1.boxplot(data_CoM_pots,  notch=True, whis=(0,100))
    CoM_ax2.boxplot(data_CoM_mag,   notch=True, whis=(0,100))
    CoM_ax3.boxplot(data_CoM_angle, notch=True, whis=(0,100))
    multi_ax1.boxplot(data_multi_pots,  notch=True, whis=(0,100))
    multi_ax2.boxplot(data_multi_mag,   notch=True, whis=(0,100))
    multi_ax3.boxplot(data_multi_angle, notch=True, whis=(0,100))

    full_x_labels = list(x_labels)*6

    CoM_ax1.set_yscale('log')
    CoM_ax1.set_ylabel('CoM')
    multi_ax1.set_ylabel('Multi')
    CoM_ax1.set_xticklabels(full_x_labels)
    CoM_ax1.set_title('Potential')
    CoM_ax2.set_title('Force Magnitude')
    CoM_ax3.set_title('Force Angle')


def terms(FILE_PATH):
    with np.load(FILE_PATH) as data:
        num_particles = data['num_particles']
        theta = data['theta']

        data_bh_pots = list(data['data_bh_pots'])
        data_bh_mag = list(data['data_bh_mag'])
        data_bh_angle = list(data['data_bh_angle'])
        data_fmm_pots = list(data['data_fmm_pots'])
        data_fmm_mag = list(data['data_fmm_mag'])
        data_fmm_angle = list(data['data_fmm_angle'])
        x_labels = data['x_labels']

    fig, axs = plt.subplots(2, 3, sharex='all', sharey='all')
    (bh_ax1, bh_ax2, bh_ax3), (fmm_ax1, fmm_ax2, fmm_ax3) = axs

    fig.suptitle(f'Errors for BH and FMM (different terms) for {num_particles} particles. theta={theta}')
    fig.supxlabel('Terms')
    fig.supylabel('Fractional Error')

    bh_ax1.boxplot(data_bh_pots,  notch=True, whis=(0,100))
    bh_ax2.boxplot(data_bh_mag,   notch=True, whis=(0,100))
    bh_ax3.boxplot(data_bh_angle, notch=True, whis=(0,100))
    fmm_ax1.boxplot(data_fmm_pots,  notch=True, whis=(0,100))
    fmm_ax2.boxplot(data_fmm_mag,   notch=True, whis=(0,100))
    fmm_ax3.boxplot(data_fmm_angle, notch=True, whis=(0,100))

    full_x_labels = list(x_labels)*6

    bh_ax1.set_yscale('log')
    bh_ax1.set_ylable('BH')
    fmm_ax1.set_ylable('FMM')
    bh_ax1.set_xticklabels(full_x_labels)
    bh_ax1.set_title('Potential')
    bh_ax2.set_title('Force Magnitude')
    bh_ax3.set_title('Force Angle')

    return fig


def theta(FILE_PATH):
    with np.load(FILE_PATH) as data:
        num_particles = data['num_particles']
        terms = data['terms']

        data_CoM_pots = list(data['data_CoM_pots'])
        data_CoM_mag = list(data['data_CoM_mag'])
        data_CoM_angle = list(data['data_CoM_angle'])
        data_multi_pots = list(data['data_multi_pots'])
        data_multi_mag = list(data['data_multi_mag'])
        data_multi_angle = list(data['data_multi_angle'])
        x_labels = data['x_labels']

    fig, axs = plt.subplots(2, 3, sharex='all', sharey='all')
    (CoM_ax1, CoM_ax2, CoM_ax3), (multi_ax1, multi_ax2, multi_ax3) = axs

    fig.suptitle(f'Errors for CoM and multi BH (different theta) for {num_particles} particles. terms={terms}')
    fig.supxlabel('Theta')
    fig.supylabel('Fractional Error')

    CoM_ax1.boxplot(data_CoM_pots,  notch=True, whis=(0,100))
    CoM_ax2.boxplot(data_CoM_mag,   notch=True, whis=(0,100))
    CoM_ax3.boxplot(data_CoM_angle, notch=True, whis=(0,100))
    multi_ax1.boxplot(data_multi_pots,  notch=True, whis=(0,100))
    multi_ax2.boxplot(data_multi_mag,   notch=True, whis=(0,100))
    multi_ax3.boxplot(data_multi_angle, notch=True, whis=(0,100))

    full_x_labels = list(x_labels)*6

    CoM_ax1.set_yscale('log')
    CoM_ax1.set_ylabel('CoM')
    multi_ax1.set_ylabel('Multi')
    CoM_ax1.set_xticks(CoM_ax1.get_xticks(), full_x_labels)
    for ax in axs.flat:
        ax.tick_params('x', labelrotation=90)
    CoM_ax1.set_title('Potential')
    CoM_ax2.set_title('Force Magnitude')
    CoM_ax3.set_title('Force Angle')

    return fig
