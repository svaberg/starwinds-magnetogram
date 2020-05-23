import numpy as np
import matplotlib.pyplot as plt
import logging
log = logging.getLogger(__name__)


def plot_lsd_profile(data, parameters="VNI"):
    """
    Plot data from a LSD profile.
    :param data: LSD profile data
    :param parameters: Which parameters to plot and in which order
    :return: figure handle
    """
    fig = plt.gcf()
    assert(len(fig.get_axes()) >= len(parameters))

    kw = {'linewidth': 2, 'elinewidth': .5}
    row_offsets = {"I": 1, "V": 3, "N": 5}

    for plot_id, parameter in enumerate(parameters):
        ax = fig.get_axes()[plot_id]
        ax.errorbar(data[:, 0], data[:, row_offsets[parameter]], yerr=data[:, row_offsets[parameter] + 1], **kw)
        ax.grid(True)
        ax.set_ylabel('Stokes $%s/I_\mathrm{c}$' % parameter)

    ax.set_xlabel('Doppler velocity')
    fig.subplots_adjust(hspace=.1)
    plt.autoscale(enable=True, axis='x', tight=True)
    return fig


def wl_intervals(wavelengths, threshold=0.2):
    """
    Find wavelength intervals a.k.a "orders" on spectrograph.
    :param wavelengths: vector of wavelengths
    :param threshold: skipping treshold
    :return: tuple of (start, end) wavelength intervals
    """
    abs_diffs = np.abs(np.diff(wavelengths))
    skips = np.where(abs_diffs > threshold)[0]
    print(skips)
    interval_starts = np.concatenate(([0],skips))
    interval_ends = np.concatenate(
        (skips + 1,
         np.array([len(wavelengths)-1])))
    return zip(interval_starts, interval_ends)


def plot_spectrum(data, wl_range=None, size=.05, alpha=.15, colormap='viridis'):
    """
    Plot spectrum data
    :param data: spectrum data
    :param wl_range: wavelength range
    :param size: marker size
    :param alpha: marker opacity
    :param colormap: marker colormap
    :return:
    """
    fig, axs = plt.subplots(5, figsize=(12, 9), sharex=True)

    for (id0, id1) in wl_intervals(data[:, 0]):
        color = np.linspace(0, 1, id1 - id0)

    print('Plotting column %d' % 0)
    axs[0].grid(True)

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
            y_range = (np.percentile(data[:, col_id], 3), np.percentile(data[:, col_id], 97))
            ax.set_ylim(y_range)
            ax.set_ylabel(name)

    axs[-1].set_xlabel('Wavelength [nm]')
    axs[-1].set_yscale('log')
    fig.subplots_adjust(hspace=.1)
    plt.autoscale(enable=True, axis='x', tight=True)

    return fig


def plot_lsd_stack(data_stack, name_stack, y_lim=None):
    """
    Plot stack of lsd files.
    :param data_stack: lsd data list
    :param name_stack: lsd name list
    :param y_lim: y axis limit setting
    :return:
    """
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
    if y_lim == 'same':
        y_lim = list(axs[0].get_ylim())
        for ax in fig.axes:
            y_lim[0] = min(y_lim[0], ax.get_ylim()[0])
            y_lim[1] = max(y_lim[1], ax.get_ylim()[1])

    if y_lim is not None:
        for ax in fig.axes:
            ax.set_ylim(y_lim)

    ax.set_xlabel('[km/s]')
    #    fig.tight_layout()

