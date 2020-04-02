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
    ax.set_ylabel(r'Polar angle $\theta$')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid()

    return img


def plot_magnetic_field(ax,
                        azimuth_centers, polar_centers, field_centers,
                        polar_corners=None, azimuth_corners=None,
                        symmetric=None,
                        cmap=None,
                        abs_max=None,
                        legend_str='X'):
    """
    Used by plot_zdi_field to plot magnetic fields, also used by quicklook.py
    :param ax: matplotlib axis on which to plot
    :param azimuth_centers:
    :param polar_centers:
    :param field_centers:
    :param polar_corners: Used in pcolormesh (if specified)
    :param azimuth_corners: Used in pcolormesh (if specified)
    :param symmetric: Setting this to True forces the colour scale to be symmetric around 0.
    :param cmap: Use the given matplotlib colormap. Otherwise "jet" or "RdBu_r" will be used.
    :return:
    """

    if symmetric is None:
        if np.min(field_centers) < 0 < np.max(field_centers):
            symmetric = True
        else:
            symmetric = False

    if symmetric and cmap is None:
        cmap = "RdBu_r"
    elif cmap is None:
        cmap = "viridis"

    # Prefer to use corners/centers
    if polar_corners is not None and azimuth_corners is not None:
        img = ax.pcolormesh(np.rad2deg(azimuth_corners),
                            np.rad2deg(polar_corners),
                            field_centers,
                            cmap=cmap)
    else:
        img = ax.pcolormesh(np.rad2deg(azimuth_centers),
                            np.rad2deg(polar_centers),
                            field_centers,
                            cmap=cmap)

    if abs_max is not None:
        img.set_clim(vmax=abs_max)

    if symmetric:
        img.set_clim(np.array([-1, 1]) * np.max(np.abs(img.get_clim())))

        zero_contour = ax.contour(np.rad2deg(azimuth_centers),
                                  np.rad2deg(polar_centers),
                                  field_centers,
                                  0,
                                  linewidths=1,
                                  colors='k',
                                  )

        # Even if there is only one line, the collections array has 3 elements.
        # At least it has in matplotlib 3.1.1 but on the HPC (currently matplotlib 2.1.2)
        # this has only 1 element
        try:
            collection = zero_contour.collections[1]
        except IndexError:
            collection = zero_contour.collections[0]
        finally:
            collection.set_label(f'${legend_str}=0$ G')
    else:
        zero_contour = None

    # add_colorbar(img, azimuth_centers, polar_centers, field_centers, ax, zero_contour, legend_str)
    # add_range(azimuth_centers, polar_centers, field_centers, ax, legend_str)

    ax.xaxis.set_ticks(np.arange(0, 361, 45))
    ax.yaxis.set_ticks(np.arange(0, 181, 30))
    ax.grid()
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xlabel("Azimuth angle [deg]")
    ax.set_ylabel("Polar angle [deg]")

    return img, zero_contour


# def add_colorbar(img1, azimuth_centers, polar_centers, field_centers, ax, zero_contour, legend_str, cax=None, markers='12'):
#     cb = ax.figure.colorbar(img1, ax=ax, cax=cax)
#
#     if zero_contour:
#         cb.add_lines(zero_contour)
#     cb.ax.plot(np.mean(cb.ax.get_xlim()),
#                np.min(field_centers), color='k', marker=markers[0], linestyle="none")
#     cb.ax.plot(np.mean(cb.ax.get_xlim()),
#                np.max(field_centers), color='k', marker=markers[0], linestyle="none")
#
#     return cb


def add_contours(azimuth_centers, polar_centers, field_centers, ax, legend_str, cb):
    contours = ax.contour(np.rad2deg(azimuth_centers),
                          np.rad2deg(polar_centers),
                          field_centers,
                          cb.get_ticks(),
                          linewidths=.5,
                          colors='k',
                          linestyles='dashed'
                          )
    contours.collections[0].set_label(fr'$\Delta {legend_str} = %g$ G' % (cb.get_ticks()[1] - cb.get_ticks()[0]))
    cb.add_lines(contours)


def add_extrema(azimuth_centers, polar_centers, field, ax, legend_str='x', markers='12'):
    field_max_indices = np.unravel_index(np.argmax(field, axis=None), field.shape)
    field_max_polar = polar_centers[field_max_indices]
    field_max_azimuth = azimuth_centers[field_max_indices]
    field_min_indices = np.unravel_index(np.argmin(field, axis=None), field.shape)
    field_min_polar = polar_centers[field_min_indices]
    field_min_azimuth = azimuth_centers[field_min_indices]
    field_max = field[field_max_indices]
    field_min = field[field_min_indices]

    ax.plot(np.rad2deg(field_max_azimuth),
             np.rad2deg(field_max_polar), color='k', marker=markers[0], linestyle="none", fillstyle='none',
             label=f'Max ${legend_str}={latex_float(field_max)}$ G')
    ax.plot(np.rad2deg(field_min_azimuth),
             np.rad2deg(field_min_polar), color='k', marker=markers[1], linestyle="none", fillstyle='none',
             label=f'Min ${legend_str}={latex_float(field_min)}$ G')


# TODO move to utils or similar.
def latex_float(value, pattern="{0:+.3g}"):
    float_str = pattern.format(value)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


