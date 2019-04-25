import numpy as np
import logging
log = logging.getLogger(__name__)


# TODO all functions should return the axis objects they create. Functions that
# generate a single axis should take ax as an argument which can be None. Functions
# which produce multiple axes can take an axs argument which must be of the right dimensions.
# There is no need to return the Figure object as it can be retrieved from ax.figure.
# No plot function should generate more than one figure.


def plot_equirectangular(geometry, value, ax, vmin=None, vmax=None, cmap='RdBu_r'):

    centers_polar, centers_azimuth = geometry.centers()
    corners_polar, corners_azimuth = geometry.corners()

    img = ax.pcolormesh(np.rad2deg(corners_azimuth.T),
                      np.rad2deg(corners_polar.T),
                      value.T,
                      cmap=cmap,
                      vmin=vmin, vmax=vmax)

    if np.min(value) < 0 < np.max(value):
        ax.contour(180 / np.pi * centers_azimuth.T, 180 / np.pi * centers_polar.T, value.T, levels=[0], colors=('k',), linewidths=.25)

    ax.set_xticks(180 / np.pi * np.linspace(corners_azimuth[0, 0], corners_azimuth[-1, -1], 9))
    ax.set_xlabel('Azimuth angle $\phi$')
    ax.set_yticks(180 / np.pi * np.linspace(corners_polar[0, 0], corners_polar[-1, -1], 5))
    ax.set_ylabel('Polar angle $\\theta$')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid()

    return img


# TODO move to utils or similar.
def latex_float(value, pattern="{0:+.3g}"):
    float_str = pattern.format(value)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


