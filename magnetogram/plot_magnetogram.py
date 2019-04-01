import numpy as np
import logging
log = logging.getLogger(__name__)


# TODO move stuff from test_plot_magnetogram.plot_test to here.
def pretty_plot(polar, azimuth, z, ax, color_range=(), color_map='RdBu_r'):
    """

    :param polar:
    :param azimuth:
    :param z:
    :param ax:
    :param color_range:
    :param color_map:
    """
    if len(color_range) == 1:
        color_min = -color_range[0]
        color_max = color_range[0]
    elif len(color_range) == 2:
        color_min = color_range[0]
        color_max = color_range[1]
    else:
        color_max = np.absolute(z[~np.isnan(z)]).max()
        color_min = np.absolute(z[~np.isnan(z)]).min()

    log.info('Color map %s, range [%f, %f].' % (color_map, color_min, color_max))

    # Print zero contour
    if np.min(z) < 0 < np.max(z):
        ax.contour(180 / np.pi * azimuth.T, 180 / np.pi * polar.T, z.T, levels=[0], colors=('g',), linewidths=.25)

    q = ax.contourf(180 / np.pi * azimuth.T, 180 / np.pi * polar.T, z.T, 128, cmap=color_map,
                    vmin=color_min, vmax=color_max)

    ax.set_xticks(180 / np.pi * np.linspace(azimuth[0, 0], azimuth[-1, -1], 9))
    ax.set_label('Azimuth angle $\phi$')
    ax.set_yticks(180 / np.pi * np.linspace(polar[0, 0], polar[-1, -1], 5))
    ax.set_label('Polar angle $\\theta$')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    # ax.invert_xaxis() # To look more like Matthew
    ax.grid()
