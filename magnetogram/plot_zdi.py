import logging
log = logging.getLogger(__name__)

import numpy as np
from matplotlib import pyplot as plt, colors
from matplotlib.ticker import IndexLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from stellarwinds import magnetogram
import stellarwinds.magnetogram.geometry
import stellarwinds.magnetogram.plots


def plot_energy(zc, types="total", negative_orders=False, axs=None):

    if type(types) == str:
        types = types,  # Turn into a tuple
    if axs is None:
        _, axs = plt.subplots(1, len(types), figsize=(5 * len(types), 4))
        axs = np.atleast_1d(axs)

    for _type, ax in zip(types, axs):
        _plot_energy(zc, _type, negative_orders, ax)

    return ax.figure, axs


def _plot_energy(zc, name="total", negative_orders=False, ax=None):

    if ax is None:
        _, ax = plt.subplots()

    erad, epol, etor = zc.energy_matrix()  # Todo change to read perhaps just some types. Also alpha, beta and gamma.
    etot = epol + etor  # Note that epol already contains erad
    _energies = dict(zip(("total", "radial", "poloidal", "toroidal"), (etot, erad, epol, etor)))

    energy = _energies[name]

    if not negative_orders:
        def remove_negative(values):
            split_id = values.shape[1] // 2
            values_neg, values_nonneg = np.split(values, [split_id], axis=1)
            assert np.allclose(values_neg, 0), "Negative orders not accepted."
            return values_nonneg

        energy = remove_negative(energy)

    if np.allclose(energy, 0):
        # If values are all zero use scale colorscale 0-1
        vmin = .1
        vmax = 1
    else:
        vmax = energy.max()
        vmin = (energy[energy > 0]).max()**-1
    img = ax.imshow(energy,
                    # cmap='Greys',
                    norm=colors.LogNorm(vmin=vmin, vmax=vmax,))
    ax.minorticks_on()
    # ax.yaxis.set_major_locator(IndexLocator(base=1, offset=.5))
    ax.xaxis.set_minor_locator(IndexLocator(base=1, offset=0))
    ax.yaxis.set_minor_locator(IndexLocator(base=1, offset=0))
    ax.tick_params(axis='both', color='none')
    ax.set_xlabel("Order $m$")
    ax.set_ylabel(r"Degree $\ell$")
    ax.set_title("Coefficient %s energy (ZDI)" % name)

    if energy.shape[0] != energy.shape[1]:
        # This means that negative orders are included.
        ax.set_xticklabels(np.array(ax.get_xticks() - energy.shape[1]//2, dtype=int))
    ax.grid(which='minor', axis='both')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    ax.figure.colorbar(img, cax=cax)

    return ax


def plot_energy_by_degree(zc, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    erad, epol, etor = zc.energy_matrix()

    ax.semilogy(np.sum(epol + etor, axis=1), 'o:', label='Total', color='k')
    _lp = ax.semilogy(np.sum(epol, axis=1), '^:', label='Poloidal')
    _lt = ax.semilogy(np.sum(etor, axis=1), '>:', label='Toroidal')
    ax.semilogy(np.cumsum(np.sum(epol + etor, axis=1)), 'o-', label='Total cumulative', color='k')
    ax.semilogy(np.cumsum(np.sum(epol, axis=1)), '^-', label='Poloidal cumulative',
                color=_lp[0].get_color())
    ax.semilogy(np.cumsum(np.sum(etor, axis=1)), '>-', label='Toroidal cumulative',
                color=_lt[0].get_color())
    ax.grid()
    ax.legend(ncol=2)
    ax.set_xlabel(r"Degree $\ell$")
    ax.set_ylabel("Summed component energy [B$^2$]")
    ax.set_title(r"Energy as a function of $\ell$")

    return ax.figure, ax


def plot_zdi_components(mgm, radius=1, axs=None, zg=None, symmetric=None, cmap=None):
    """
    Plot the 3 components of the magnetic field.
    """

    if zg is None:
        zg = magnetogram.geometry.ZdiGeometry(61)

    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(24, 6))

    polar_centers, azimuth_centers = zg.centers()
    polar_corners, azimuth_corners = zg.corners()

    axs[0].figure.subplots_adjust(right=0.8)  # Make space for colorbar.

    getter_fns = (mgm.get_radial_field, mgm.get_polar_field, mgm.get_azimuthal_field)
    Brpa = np.array([g(polar_centers, azimuth_centers) for g in getter_fns])
    # import pdb; pdb.set_trace()

    def place_colorbar_axis_right(ax, dx=.22):
        p0 = ax.get_position().p0
        p1 = ax.get_position().p1

        cbar_ax = axs[0].figure.add_axes((p0[0] + dx, p0[1], .01, p1[1] - p0[1]))
        return cbar_ax

    abs_max = np.max(np.abs(Brpa))
    for ax, Bi in zip(axs, Brpa):
        log.debug("Axis " + str(ax))
        img = stellarwinds.magnetogram.plots.plot_equirectangular(zg, Bi, ax, vmin=-abs_max, vmax=abs_max)
        ax.set_ylabel(None)

        # plot_magnetic_field(ax,
        #                     azimuth_centers, polar_centers, Bi,
        #                     polar_corners=polar_corners, azimuth_corners=azimuth_corners)


    cax = place_colorbar_axis_right(ax)
    ax.figure.colorbar(img, cax=cax)

    axs[0].set_title(r"Radial field $B_r$ at $r = %2.1f r_\star$" % radius)
    axs[1].set_title(r"Polar field $B_\theta$ at $r = %2.1f r_\star$" % radius)
    axs[2].set_title(r"Azimuthal field $B_\phi$ at $r = %2.1f r_\star$" % radius)
    axs[0].set_ylabel(r"Polar angle $\theta$ [deg]")

    for ax in axs:
        ax.set_xlabel(r"Azimuth angle $\phi$ [deg]")

    abs_field_mean = np.sum(np.abs(Brpa) * zg.areas()) / (4 * np.pi)  # Scaled by area
    log.debug(f"Mean fiend strength {abs_field_mean:.3G} G")
    axs[1].text(.5, .5, #1, 0,
                f"abs mean: {abs_field_mean:.3G} G ",
            transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='bottom')

    return axs


# Expand to plot other than radial??
# TODO This is a hack to get field strength quickly to match the plots in the kappa Ceti paper.
def plot_zdi_field(getter_fn, ax=None, zg=None, symmetric=None, cmap=None, legend_str='X'):

    if zg is None:
        zg = magnetogram.geometry.ZdiGeometry(61)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    polar_centers, azimuth_centers = zg.centers()
    polar_corners, azimuth_corners = zg.corners()

    field_centers = getter_fn(polar_centers, azimuth_centers)

    plot_magnetic_field(ax, azimuth_centers, polar_centers, field_centers,
                        polar_corners=polar_corners, azimuth_corners=azimuth_corners,
                        symmetric=symmetric,
                        cmap=cmap,
                        legend_str=legend_str)

    abs_field_mean = np.sum(np.abs(field_centers) * zg.areas()) / (4 * np.pi)  # Scaled by area
    ax.text(1, 0, f"abs mean: {abs_field_mean:.3G} G ",
            transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='bottom')

    return fig, ax


def plot_magnetic_field(ax,
                        azimuth_centers, polar_centers, field_centers,
                        polar_corners=None, azimuth_corners=None,
                        symmetric=None,
                        cmap=None,
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
        img1 = ax.pcolormesh(np.rad2deg(azimuth_corners),
                             np.rad2deg(polar_corners),
                             field_centers,
                             cmap=cmap)
    else:
        img1 = ax.pcolormesh(np.rad2deg(azimuth_centers),
                             np.rad2deg(polar_centers),
                             field_centers,
                             cmap=cmap)

    cb = ax.figure.colorbar(img1, ax=ax)

    if symmetric:
        img1.set_clim(np.array([-1, 1]) * np.max(np.abs(img1.get_clim())))

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
        cb.add_lines(zero_contour)

    cb.ax.plot(np.mean(cb.ax.get_xlim()),
               np.min(field_centers), color='k', marker='1')
    cb.ax.plot(np.mean(cb.ax.get_xlim()),
               np.max(field_centers), color='k', marker='2')

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

    ax.xaxis.set_ticks(np.arange(0, 361, 45))
    ax.yaxis.set_ticks(np.arange(0, 181, 30))
    plt.grid()

    ax.invert_yaxis()

    ax.set_aspect('equal')

    add_range(azimuth_centers, polar_centers, field_centers, legend_str)

    plt.legend(ncol=2, loc='lower left')

    ax.set_xlabel("Azimuth angle [deg]")
    ax.set_ylabel("Polar angle [deg]")


def add_range(azimuth_centers, polar_centers, field, legend_str='x'):
    field_max_indices = np.unravel_index(np.argmax(field, axis=None), field.shape)
    field_max_polar = polar_centers[field_max_indices]
    field_max_azimuth = azimuth_centers[field_max_indices]
    field_min_indices = np.unravel_index(np.argmin(field, axis=None), field.shape)
    field_min_polar = polar_centers[field_min_indices]
    field_min_azimuth = azimuth_centers[field_min_indices]
    field_max = field[field_max_indices]
    field_min = field[field_min_indices]

    plt.plot(np.rad2deg(field_max_azimuth),
             np.rad2deg(field_max_polar), 'k2',
             label=f'Max ${legend_str}={stellarwinds.magnetogram.plots.latex_float(field_max)}$ G')
    plt.plot(np.rad2deg(field_min_azimuth),
             np.rad2deg(field_min_polar), 'k1',
             label=f'Min ${legend_str}={stellarwinds.magnetogram.plots.latex_float(field_min)}$ G')
