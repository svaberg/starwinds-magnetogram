import numpy as np
from matplotlib import pyplot as plt, colors
from matplotlib.ticker import IndexLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from stellarwinds import magnetogram
import stellarwinds.magnetogram.geometry
from stellarwinds.magnetogram.plots import log, latex_float


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
    ax.set_ylabel("Degree $\ell$")
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
    ax.set_xlabel("Degree $\ell$")
    ax.set_ylabel("Summed component energy [B$^2$]")
    ax.set_title("Energy as a function of $\ell$")

    return ax.figure, ax


def plot_map(zdi_magnetogram, star_name):
    zg = magnetogram.geometry.ZdiGeometry(61)

    polar_centers, azimuth_centers = zg.centers()

    stellarwinds.magnetogram.geometry.numerical_description(zg, zdi_magnetogram)

    b_radial = zdi_magnetogram.get_radial_field(polar_centers, azimuth_centers)
    b_radial_max_indices = np.unravel_index(np.argmax(b_radial, axis=None), b_radial.shape)
    b_radial_max_polar = polar_centers[b_radial_max_indices]
    b_radial_max_azimuth = azimuth_centers[b_radial_max_indices]
    b_radial_max = b_radial[b_radial_max_indices]

    b_radial_min_indices = np.unravel_index(np.argmin(b_radial, axis=None), b_radial.shape)
    b_radial_min_polar = polar_centers[b_radial_min_indices]
    b_radial_min_azimuth = azimuth_centers[b_radial_min_indices]
    b_radial_min = b_radial[b_radial_min_indices]

    if np.abs(b_radial_max) > np.abs(b_radial_min):
        abs_b_radial_max = np.abs(b_radial_max)
        abs_b_radial_max_polar = b_radial_max_polar
        abs_b_radial_max_azimuth = b_radial_max_azimuth
    else:
        abs_b_radial_max = np.abs(b_radial_min)
        abs_b_radial_max_polar = b_radial_min_polar
        abs_b_radial_max_azimuth = b_radial_min_azimuth

    abs_b_radial_mean = np.sum(np.abs(b_radial) * zg.areas()) / (4 * np.pi)

    log.info("|B_r|_max = %4.4g Gauss" % abs_b_radial_max)
    log.info("|B_r|_max at az=%2.2f deg, pl=%3.2f deg" % (np.rad2deg(abs_b_radial_max_azimuth),
                                                          np.rad2deg(abs_b_radial_max_polar)))
    log.info("|B_r|_mean = %4.4g Gauss" % abs_b_radial_mean)

    fig, ax = plt.subplots(figsize=(12, 6))
    polar_corners, azimuth_corners = zg.corners()
    img1 = ax.pcolormesh(np.rad2deg(azimuth_corners),
                         np.rad2deg(polar_corners),
                         b_radial,
                         vmin=-abs_b_radial_max, vmax=abs_b_radial_max,
                         cmap="RdBu_r")
    cb = fig.colorbar(img1, ax=ax)

    i1 = ax.contour(np.rad2deg(zg.centers()[1]),
                    np.rad2deg(zg.centers()[0]),
                    b_radial,
                    0,
                    linewidths=1,
                    colors='k',
                    )

    i1.collections[1].set_label('$B_r=0$ G')  # Even if there is only one line, the collections array has 3 elements.
    cb.add_lines(i1)

    i2 = ax.contour(np.rad2deg(zg.centers()[1]),
                    np.rad2deg(zg.centers()[0]),
                    b_radial,
                    cb.get_ticks(),
                    linewidths=.5,
                    colors='k',
                    linestyles='dashed'
                    )

    i2.collections[0].set_label('$\\Delta B_r = %g$ G' % (cb.get_ticks()[1] - cb.get_ticks()[0]))
    # import pdb; pdb.set_trace()
    # ax.clabel(i2, fmt='%2.1f', colors='k', fontsize=6)
    # h2, _ =
    # for _id, h2l in enumerate(i2.collections):
    #     h2l.set_label(_id)

    cb.add_lines(i2)

    # _ticks = cb.get_ticks()
    if np.abs(b_radial_max) > np.abs(b_radial_min):
        cb.ax.axhline(y=b_radial_min, color='y')
        _value_to_add = b_radial_min
    else:
        cb.ax.axhline(y=b_radial_max, color='g')
        _value_to_add = b_radial_max

    ax.xaxis.set_ticks(np.arange(0, 361, 45))
    ax.yaxis.set_ticks(np.arange(0, 181, 30))
    plt.grid()

    plt.plot(np.rad2deg(b_radial_max_azimuth),
             np.rad2deg(b_radial_max_polar), 'g^',
             label='Max $B_r=%s$ G' % latex_float(b_radial_max))
    plt.plot(np.rad2deg(b_radial_min_azimuth),
             np.rad2deg(b_radial_min_polar), 'yv',
             label='Min $B_r=%s$ G' % latex_float(b_radial_min))

    plt.title('%s: $|B_r|_\mathrm{mean}=%4.4g$ G' % (star_name, abs_b_radial_mean))
    ax.invert_yaxis()

    ## Dipole max
    b_radial = zdi_magnetogram.as_dipole().get_radial_field(polar_centers, azimuth_centers)
    b_radial_max_indices = np.unravel_index(np.argmax(b_radial, axis=None), b_radial.shape)
    b_radial_max_polar = polar_centers[b_radial_max_indices]
    b_radial_max_azimuth = azimuth_centers[b_radial_max_indices]
    b_radial_max = b_radial[b_radial_max_indices]

    b_radial_min_indices = np.unravel_index(np.argmin(b_radial, axis=None), b_radial.shape)
    b_radial_min_polar = polar_centers[b_radial_min_indices]
    b_radial_min_azimuth = azimuth_centers[b_radial_min_indices]
    b_radial_min = b_radial[b_radial_min_indices]

    log.info("Dipole polar angle %g deg" % np.rad2deg(np.min([b_radial_max_polar, b_radial_min_polar])))

    plt.plot(np.rad2deg(b_radial_max_azimuth, ), np.rad2deg(b_radial_max_polar), 'g^', fillstyle='none', label='Dipole $B_r=%s$ G' % latex_float(b_radial_max))
    plt.plot(np.rad2deg(b_radial_min_azimuth, ), np.rad2deg(b_radial_min_polar), 'yv', fillstyle='none', label='Dipole $B_r=%s$ G' % latex_float(b_radial_min))

    plt.legend(ncol=3, loc='lower left')

    ax.set_aspect('equal')

    plt.title('%s: $|B_r|_\mathrm{mean}=%4.4g$ G. Dipole at %4.4g deg' % (star_name,
                                                                abs_b_radial_mean,
                                                                np.rad2deg(np.min([b_radial_max_polar, b_radial_min_polar]))))

    return fig, ax


def pole_walk(zdi_magnetogram, geometry=None, ax=None):
    """
    Forget this; the pole walks all over the place
    :param zdi_magnetogram:
    :param geometry:
    :return:
    """
    if ax is None:
        ax = plt.gca()
    if geometry is None:
        geometry = magnetogram.geometry.ZdiGeometry(128)

    polar_centers, azimuth_centers = geometry.centers()

    # Dipole max
    b_radial_max_polar = []
    b_radial_max_azimuth = []
    b_radial_min_polar = []
    b_radial_min_azimuth = []

    for max_degree in range(zdi_magnetogram.degree()):
        b_radial = zdi_magnetogram.as_restricted(degree_l_range=(0, max_degree)).get_radial_field(polar_centers, azimuth_centers)
        b_radial_max_indices = np.unravel_index(np.argmax(b_radial, axis=None), b_radial.shape)
        b_radial_max_polar.append(polar_centers[b_radial_max_indices])
        b_radial_max_azimuth.append(azimuth_centers[b_radial_max_indices])
        b_radial_max = b_radial[b_radial_max_indices]

        b_radial_min_indices = np.unravel_index(np.argmin(b_radial, axis=None), b_radial.shape)
        b_radial_min_polar.append(polar_centers[b_radial_min_indices])
        b_radial_min_azimuth.append(azimuth_centers[b_radial_min_indices])
        b_radial_min = b_radial[b_radial_min_indices]

    ax.plot(np.rad2deg(b_radial_max_azimuth, ), np.rad2deg(b_radial_max_polar), 'g:^', label='Max $B_r=%s$ G' % latex_float(b_radial_max))
    ax.plot(np.rad2deg(b_radial_min_azimuth, ), np.rad2deg(b_radial_min_polar), 'y:v', label='Min $B_r=%s$ G' % latex_float(b_radial_min))

    for i in range(len(b_radial_min_azimuth)):
        ax.text(np.rad2deg(b_radial_max_azimuth[i]), np.rad2deg(b_radial_max_polar[i]), str(i))
        ax.text(np.rad2deg(b_radial_min_azimuth[i]), np.rad2deg(b_radial_min_polar[i]), str(i))
    ax.legend()
    return plt.gcf(), ax