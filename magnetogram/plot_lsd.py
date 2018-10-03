import numpy as np
import matplotlib.pyplot as plt

import logging
log = logging.getLogger(__name__)


def plot_file(file_name, skip_header=2):
    data = np.genfromtxt(file_name, skip_header=skip_header)
    fig, axs = plt.subplots(6, figsize=(12,9), sharex=True)
    for row_id in range(0,6):
        axs[row_id].plot(data[:,0], data[:,row_id+1], '-')
        axs[row_id].grid(True)
    plt.grid(True)
    axs[0].set_title('Data in %s' % file_name)
    fig.subplots_adjust(hspace=.1)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.show()


def plot_file_triple(file_base_name, skip_header=2, rows=range(0, 6)):
    base_data = np.genfromtxt(file_base_name, skip_header=skip_header)
    norm_data = np.genfromtxt(file_base_name + '.norm', skip_header=skip_header)
    model_data = np.genfromtxt(file_base_name + '.norm.model', skip_header=skip_header)
    data = np.genfromtxt(file_base_name, skip_header=skip_header)

    fig, axs = plt.subplots(len(rows), figsize=(12, 9), sharex=True)
    for plot_id, row_id in enumerate(rows):
        axs[plot_id].plot(base_data[:, 0], base_data[:, row_id + 1], '-')
        axs[plot_id].plot(norm_data[:, 0], norm_data[:, row_id + 1], '-')
        axs[plot_id].plot(model_data[:, 0], model_data[:, row_id + 1], '-')
        axs[plot_id].grid(True)
    plt.grid(True)
    axs[0].set_title('Data in %s' % file_name)
    fig.legend(('profD', 'profD.norm', 'profD.norm.model'))
    fig.subplots_adjust(hspace=.1)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig(file_base_name + '.png')
    plt.show()


def plot_file_interp(file_base_name, skip_header=2):
    base_data = np.genfromtxt(file_base_name, skip_header=skip_header)
    norm_data = np.genfromtxt(file_base_name + '.norm', skip_header=skip_header)
    model_data = np.genfromtxt(file_base_name + '.norm.model', skip_header=skip_header)
    data = np.genfromtxt(file_base_name, skip_header=skip_header)

    fig, axs = plt.subplots(3, figsize=(12, 9), sharex=True)
    kw = {'linewidth': 2, 'elinewidth': .5}

    rows = (3, 5, 1)
    ylabels = ('Stokes $V/V_\mathrm{c}$', 'Null $N/N_\mathrm{c}$', 'Stokes $I/I_\mathrm{c}$')
    for plot_id, row_id in enumerate(rows):
        ax = axs[plot_id]
        ax.errorbar(base_data[:, 0], base_data[:, row_id], yerr=base_data[:, row_id + 1], **kw)
        ax.errorbar(norm_data[:, 0], norm_data[:, row_id], yerr=norm_data[:, row_id + 1], **kw)
        ax.errorbar(model_data[:, 0], model_data[:, row_id], yerr=model_data[:, row_id + 1], **kw)
        ax.grid(True)
        ax.set_ylabel(ylabels[plot_id])

    axs[0].set_title('Data in %s' % file_name)
    axs[2].legend(('profD', 'profD.norm', 'profD.norm.model'))
    fig.subplots_adjust(hspace=.1)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig(file_base_name + '.png')
    plt.show()


def plot_file_columns(file_name, skip_header=2, columns=None):
    data = np.genfromtxt(file_name, skip_header=skip_header)
    if columns is None:
        columns = range(1, data.shape[1])
    print('Using columns:', columns)
    fig, axs = plt.subplots(len(columns), figsize=(12,9), sharex=True)
    for plot_id, col_id in enumerate (columns):
        print('Plotting column %d' % col_id)
        axs[plot_id].plot(data[:,0], data[:,col_id], ',', alpha=.05)
        axs[plot_id].grid(True)
        yrange = (np.percentile(data[:,col_id],3), np.percentile(data[:,col_id],97))
        axs[plot_id].set_ylim(yrange)
    plt.grid(True)
    axs[0].set_title('Data in %s' % file_name)
    fig.subplots_adjust(hspace=.1)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig('lopeg_31aug14_v_09.s.full.png')
    plt.show()


def wl_intervals(wavelengths, treshold=0.2):
    abs_diffs = np.abs(np.diff(wavelengths))
    skips = np.where(abs_diffs > treshold)[0]
    print(skips)
    interval_starts = np.concatenate(([0],skips))
    interval_ends = np.concatenate(
        (skips + 1,
         np.array([len(wavelengths)-1])))
    return zip(interval_starts, interval_ends)


def plot_s_data(full_data, wl_range=None, size=.05, alpha=.15, colormap='viridis'):
    fig, axs = plt.subplots(5, figsize=(12, 9), sharex=True)

    for (id0, id1) in wl_intervals(data[:, 0]):
        color = np.linspace(0, 1, id1 - id0)

    print('Plotting column %d' % 0)
    axs[0].grid(True)

    # print(interval_starts)
    # print(interval_ends)

    names = ('normalised flux I/Ic',
             'circular V',
             '1st null ',
             '2nd null ',
             'error bar')

    for plot_id, name in enumerate(names):
        ax = axs[plot_id]
        col_id = plot_id + 1

        for (id0, id1) in wl_intervals(data[:, 0]):
            color = np.linspace(0, 1, id1 - id0)

            #            ax.plot(data[id0:id1,0], data[id0:id1,col_id], ',', alpha=alpha)
            ax.scatter(data[id0:id1, 0], data[id0:id1, col_id], s=size, c=color,
                       cmap=colormap,
                       alpha=alpha)
            ax.grid(True)
            yrange = (np.percentile(data[:, col_id], 3), np.percentile(data[:, col_id], 97))
            ax.set_ylim(yrange)
            ax.set_ylabel(name)

    axs[0].set_title('Data in %s' % file_name)
    axs[-1].set_xlabel('Wavelength [nm]')
    axs[-1].set_yscale('log')
    fig.subplots_adjust(hspace=.1)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig('lopeg_31aug14_v_09.s.full.png')
    plt.show()


def plot_s_file(file_name, skip_header=2, wl_range=None):
    data = np.genfromtxt(file_name, skip_header=skip_header)
    print(data[:, 0].shape)
    if wl_range is not None:
        id_min = np.searchsorted(data[:,0], wl_range[0])
        id_max = np.searchsorted(data[:,0], wl_range[1])
        data = data[id_min:id_max,...]
    print(data[:, 0].shape)
    fig, axs = plt.subplots(5, figsize=(12,9), sharex=True)
    axs[0].plot(data[:,0], data[:,1], '-')
    axs[0].grid(True)
    axs[1].plot(data[:,0], data[:,2], '-')
    axs[1].grid(True)
    axs[2].plot(data[:,0], data[:,3], '-')
    axs[2].grid(True)
    axs[3].plot(data[:,0], data[:,4], '-')
    axs[3].grid(True)
    axs[4].set_yscale('symlog', linthreshy=1e-2)
    axs[4].plot(data[:,0], data[:,5], '-')
    axs[4].grid(True)
    axs[4].set_yscale('log')

    axs[0].set_title('Data in %s' % file_name)
    fig.subplots_adjust(hspace=.1)
    plt.autoscale(enable=True, axis='x', tight=True)

    plt.savefig('lopeg_31aug14_v_09.s.detail.png')
    plt.show()


def plot_lsd_a(file_name, skip_header=2):

    data = np.genfromtxt(file_name, skip_header=skip_header)

    fig = plt.gcf()
    if len(fig.axes) != 3:
        log.debug('Create new plot with 3 subplots')
        fig, axs = plt.subplots(3, figsize=(12, 9), sharex=True)
    else:
        log.debug('Reuse existing plot (with 3 subplots).')

    err_kw = {'linewidth': 2, 'elinewidth': .5}


    ax_id=0
    ax = fig.axes[ax_id]
    ax.errorbar(data[:, 0], data[:, 1],
                yerr=data[:, 2],
                **err_kw)
    ax.grid(True)
    ax.set_ylabel('$I/I_\mathrm{c}$')


    ax_id = ax_id + 1
    ax = fig.axes[ax_id]
    ax.set_title('Data in %s' % file_name)
    ax.errorbar(data[:, 0], data[:, 3],
                yerr=data[:, 4],
                **err_kw)
    ax.grid(True)
    ax.set_ylabel('$V/V_\mathrm{c}$')


    ax_id = ax_id + 1
    ax = fig.axes[ax_id]
    ax.errorbar(data[:, 0], data[:, 5],
                yerr=data[:, 6],
                **err_kw)
    ax.grid(True)
    ax.set_ylabel('Null $N/N_\mathrm{c}$')


    #    fig.tight_layout()
    fig.subplots_adjust(hspace=.1)
    plt.autoscale(enable=True, axis='x', tight=True)


def plot_lsd_stack(data_stack, name_stack, skip_header=2, ylim=None):
    log.debug('Stack size %d', len(data_stack))

    err_kw = {'linewidth': 2, 'elinewidth': .5}

    fig = plt.gcf()
    axs = fig.axes
    if len(fig.axes) != len(data_stack):
        log.debug('Create new plot with %d subplots' % len(data_stack))
        fig, axs = plt.subplots(len(data_stack), figsize=(12, len(data_stack)), sharex=True)
    else:
        log.debug('Reuse existing plot (with 3 subplots).')


    fig.subplots_adjust(hspace=.0)
    plt.autoscale(enable=True, axis='x', tight=True)

    for data_id, data in enumerate(data_stack):
        ax = axs[data_id]

        ax.errorbar(data[:, 0], data[:, 3],
                    yerr=data[:, 4],
                    **err_kw)

        if name_stack is not None:
            ax.legend((name_stack[data_id],))

        ax.grid(True)
        ax.set_ylabel('$V/V_\mathrm{c}$')

    #
    # Y limit options
    #
    if ylim == 'same':
        ylim = list(axs[0].get_ylim())
        for ax in fig.axes:
            ylim[0] = min(ylim[0], ax.get_ylim()[0])
            ylim[1] = max(ylim[1], ax.get_ylim()[1])

    if ylim is not None:
        for ax in fig.axes:
            ax.set_ylim(ylim)


    ax.set_xlabel('[km/s]')
    #    fig.tight_layout()


if __name__ == "__main__":
    file_name = 'lopeg_31aug14_v_09.profD'
    plot_file_interp(file_name)
    plot_file_triple(file_name)
    s_file_name = '/Users/u1092841/Documents/PHD/toupies-data-colin/LOPEG/lopeg_31aug14_v_09.s'
    plot_file_columns(s_file_name)

    data = np.genfromtxt(s_file_name, skip_header=2)
    deltas = np.abs(np.diff(data[:,0]))
    plt.plot(deltas)
    plt.plot(0*deltas + np.mean(deltas))
    plt.yscale('log')
    print(np.mean(deltas))
    print(deltas[0]-deltas[1])

    data = np.genfromtxt(s_file_name, skip_header=2)
    ids = (data[:, 0] > 500) & (data[:, 0] < 600)
    plot_s_data(data[ids, ...])
    ids = (data[:, 0] > 600) & (data[:, 0] < 604)
    plot_s_data(data[ids, ...], size=.3, alpha=1)

    plot_s_file(s_file_name, wl_range=(600,604))

    plot_lsd('/Users/u1092841/Documents/PHD/toupies-pipeline/lopeg_31aug14_v_09.s.out.lsd')
