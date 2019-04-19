import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import IndexLocator
from matplotlib import colors
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging
log = logging.getLogger(__name__)

from stellarwinds.magnetogram import pfss_stanford
from stellarwinds.magnetogram import zdi_geometry
from stellarwinds import coordinate_transforms


def plot_pfss_equirectangular(magnetogram, geometry=None, radius_source_surface=3):

    if geometry is None:
        geometry = zdi_geometry.ZdiGeometry()

    #
    # Evaluate magnetic field at coordinates.
    #
    degree_l, order_m, alpha_lm = magnetogram.as_arrays(include_unset=False)
    polar, azimuth = zdi_geometry.ZdiGeometry().centers()

    Bs = pfss_stanford.evaluate_on_sphere(
        degree_l,
        order_m,
        np.real(alpha_lm),
        np.imag(alpha_lm),
        polar, azimuth,
        radius=1, radius_source_surface=radius_source_surface)

    Bss = pfss_stanford.evaluate_on_sphere(
        degree_l,
        order_m,
        np.real(alpha_lm),
        np.imag(alpha_lm),
        polar, azimuth,
        radius=1.5, radius_source_surface=radius_source_surface)

    fig, axs_mat = plt.subplots(2, 3, figsize=(12, 5))
    fig.subplots_adjust(right=0.8)  # Make space for colorbar.

    for ax_row, B, radius in zip(axs_mat, (Bs, Bss), (1, radius_source_surface)):
        abs_max = np.max(np.abs(B))
        for component_id in range(len(B)):
            img = plot_equirectangular(geometry, B[component_id], ax_row[component_id],
                                       vmin=-abs_max, vmax=abs_max)
            ax_row[component_id].set_ylabel(None)

        def place_colorbar_axis_right(ax, dx=.22):
            p0 = ax.get_position().p0
            p1 = ax.get_position().p1

            cbar_ax = fig.add_axes((p0[0] + dx, p0[1], .01, p1[1]-p0[1]))
            return cbar_ax

        cax = place_colorbar_axis_right(ax_row[2])

        # divider = make_axes_locatable(ax_row[2])  # This changes the size of ax_row[2]
        # cax = divider.append_axes("right", size="5%", pad=0.15)
        fig.colorbar(img, cax=cax)

        ax_row[0].set_title("Radial field $B_r$ at $r = %2.1f r_\star$" % radius)
        ax_row[1].set_title("Polar field $B_\\theta$ at $r = %2.1f r_\star$" % radius)
        ax_row[2].set_title("Azimuthal field $B_\\phi$ at $r = %2.1f r_\star$" % radius)

        ax_row[0].set_ylabel("Polar angle $\\theta$ [deg]")

        ax_row[0].set_xlabel("Azimuth angle $\\phi$ [deg]")
        ax_row[1].set_xlabel("Azimuth angle $\\phi$ [deg]")
        ax_row[2].set_xlabel("Azimuth angle $\\phi$ [deg]")

    return fig, axs_mat


def plot_pfss_slice(magnetogram, geometry=None, normal="x", rmax=8,
                    radius_source_surface=3, radius_star=1):

    # Points in normal plane.
    _p1 = np.linspace(-1, 1, 302) * rmax
    _p2 = np.linspace(-1, 1, 304) * rmax
    _p3 = np.atleast_1d(0)

    p1, p2 = np.meshgrid(_p1, _p2)

    assert p1.shape == (len(_p2), len(_p1))  # np.meshgrid switches argument 1 and 2 in result.

    _shape = p1.shape

    def rotate(p1, p2, normal):
        p3 = np.zeros_like(p1)
        if normal == "x":
            return p3[..., np.newaxis], p1[..., np.newaxis], p2[..., np.newaxis]
        elif normal == "y":
            return p1[..., np.newaxis], p3[..., np.newaxis], p2[..., np.newaxis]
        if normal == "z":
            return p1[..., np.newaxis], p2[..., np.newaxis], p3[..., np.newaxis]

    px, py, pz = rotate(p1, p2, normal)

    pr, pp, pa = coordinate_transforms.spherical_coordinates_from_rectangular(px, py, pz)

    degree_l, order_m, alpha_lm = magnetogram.as_arrays(include_unset=False)
    field_radial, field_polar, field_azimuthal = pfss_stanford.evaluate_in_space(degree_l, order_m,
                                                                                 np.real(alpha_lm),
                                                                                 np.imag(alpha_lm),
                                                                                 pr, pp, pa,
                                                                                 radius_star=radius_star,
                                                                                 radius_source_surface=radius_source_surface)

    assert field_radial.shape == pr.shape
    assert field_polar.shape == pr.shape
    assert field_azimuthal.shape == pr.shape

    Frpa = np.stack([c.flatten() for c in (field_radial, field_polar, field_azimuthal)], axis=-1)

    transformation_matrix = coordinate_transforms.spherical_to_rectangular_transformation_matrix(pp.flatten(),
                                                                                                 pa.flatten())
    Fxyz = transformation_matrix @ Frpa[:, :, np.newaxis]
    # Get rid of last dimension which is 1
    Fxyz = np.squeeze(Fxyz)

    fx = Fxyz[:, 0].reshape(_shape)
    fy = Fxyz[:, 1].reshape(_shape)
    fz = Fxyz[:, 2].reshape(_shape)

    fig, ax = plt.subplots(figsize=(9, 6))

    fmag = (fx[:, :] ** 2 + fy[:, :] ** 2 + fz[:, :] ** 2) ** .5
    fmin = np.min(fmag[np.where(fmag > 0)])
    fmax = np.max(fmag)

    norm = colors.SymLogNorm(linthresh=100 * fmin,
                             linscale=1,
                             vmin=-fmax,
                             vmax=fmax)

    im = ax.pcolormesh(p1, p2, np.squeeze(field_radial),
                       norm=norm,
                       cmap='RdBu_r')

    fig.colorbar(im).set_label('Radial field strength')

    radial_distance = np.squeeze(pr)
    ax.contour(p1[:, :], p2[:, :], radial_distance, levels=[radius_star, radius_source_surface], colors='k')


    # Streamplots only make sense if the field has a symmetry axis.

    # ax.streamplot(p1[:, :].transpose(), p2[:, :].transpose(),
    #               fx[:, :].transpose(), fz[:, :].transpose(), # TODO FIx
    #               color='gray')

    ax.set_aspect('equal')
    ax.set_xlabel('Distance $x/r_{\\star}$')
    ax.set_ylabel('Distance $z/r_{\\star}$')

    return fig, ax


def plot_pfss_streamtraces(magnetogram, geometry=zdi_geometry.ZdiGeometry()):

    degree_l, order_m, alpha_lm = magnetogram.as_arrays(include_unset=False)
    polar, azimuth = zdi_geometry.ZdiGeometry().centers()

    B_radial, B_polar, B_azimuthal = pfss_stanford.evaluate_on_sphere(
        degree_l,
        order_m,
        np.real(alpha_lm),
        np.imag(alpha_lm),
        polar, azimuth)

    B_tangential_mag = np.sqrt(B_polar ** 2 + B_azimuthal ** 2)
    B_radial_abs_max = np.max(np.abs(B_radial))

    fig, ax = plt.subplots(figsize=(18, 6))

    img = plot_equirectangular(geometry, B_radial, ax,
                               vmin=-B_radial_abs_max,
                               vmax=B_radial_abs_max,
                               cmap='RdBu_r')

    c = plt.colorbar(img, extend='both')
    # c.set_clim(vmin=-180, vmax=180)
    # c.set_ticks((-180, -120, -60, 0, 60, 120, 180), update_ticks=True)

    _s = plt.streamplot(np.rad2deg(azimuth.T),
                        np.rad2deg(polar.T),
                        B_azimuthal.T, B_polar.T, density=[2, 1],
                        # color=B_radial.T,
                        color=(0, 0, 0, .2),
                        linewidth=3 * B_tangential_mag.T / B_tangential_mag.max(),
                        # norm=Normalize(-15, 15),
                        )

    # Draw zero contour
    _br = ax.contour(180 / np.pi * azimuth.T, 180 / np.pi * polar.T, B_radial.T, levels=[0], colors=('b',), linewidths=1)
    _bp = ax.contour(180 / np.pi * azimuth.T, 180 / np.pi * polar.T, B_polar.T, levels=[0], colors=('g',), linewidths=1)
    _ba = ax.contour(180 / np.pi * azimuth.T, 180 / np.pi * polar.T, B_azimuthal.T, levels=[0], colors=('r',), linewidths=1)

    legend_items = [_br.collections[0], _bp.collections[0], _ba.collections[0]]
    legend_strs = ["Radial $B_r = 0$", "Polar $B_\\theta = 0$", "Azimuth $B_\phi = 0$"]

    locator = MaxNLocator(nbins=3, prune="lower")
    line_values = locator.tick_values(np.min(B_tangential_mag), np.max(B_tangential_mag))
    line_thicknesses = 3 * line_values / np.max(line_values)

    custom_lines = [Line2D([0], [0], color='k', lw=l) for l in line_thicknesses]
    custom_strs = ["$B_\perp=%4.1f$" % x for x in line_values]
    plt.legend(legend_items + custom_lines, legend_strs + custom_strs)

    return fig, ax


def plot_zdi_energy(zc, types="total", negative_orders=False):
    if type(types) == str:
        types = types,  # Turn into a tuple

    erad, epol, etor = zc.energy_matrix()

    if not negative_orders:
        def remove_negative(values):
            split_id = values.shape[1] // 2
            values_neg, values_nonneg = np.split(values, [split_id], axis=1)
            assert np.allclose(values_neg, 0), "Negative orders not accepted."
            return values_nonneg

        erad = remove_negative(erad)
        epol = remove_negative(epol)
        etor = remove_negative(etor)

    etot = epol + etor  # Note that epol already contains erad

    _energies = dict(zip(("total", "radial", "poloidal", "toroidal"), (etot, erad, epol, etor)))

    fig, axs = plt.subplots(1, len(types), figsize=(5 * len(types), 4))
    axs = np.atleast_1d(axs)

    for _type, ax in zip(types, axs):
        values = _energies[_type]

        if np.allclose(values, 0):
            # If values are all zero use scale colorscale 0-1
            vmin = .1
            vmax = 1
        else:
            vmax = values.max()
            vmin = (values[values > 0]).max()**-1
        img = ax.imshow(values,
                        # cmap='Greys',
                        norm=colors.LogNorm(vmin=vmin, vmax=vmax,))
        ax.minorticks_on()
        # ax.yaxis.set_major_locator(IndexLocator(base=1, offset=.5))
        ax.xaxis.set_minor_locator(IndexLocator(base=1, offset=0))
        ax.yaxis.set_minor_locator(IndexLocator(base=1, offset=0))
        ax.tick_params(axis='both', color='none')
        ax.set_xlabel("Order $m$")
        ax.set_ylabel("Degree $\ell$")
        ax.set_title("Coefficient %s energy (ZDI)" % _type)

        if values.shape[0] != values.shape[1]:
            # This means that negative orders are included.
            ax.set_xticklabels(np.array(ax.get_xticks() - values.shape[1]//2, dtype=int))
        ax.grid(which='minor', axis='both')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        fig.colorbar(img, cax=cax)

    return fig, axs


def plot_energy_by_degree(zc):
    erad, epol, etor = zc.energy_matrix()

    fig, ax = plt.subplots()
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
    ax.set_ylabel("Component energy [B$^2$]")
    ax.set_title("Energy as a function of $\ell$")

    return fig, ax


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



def plot_map(lz, star_name):

    zg = zdi_geometry.ZdiGeometry(61)

    polar_centers, azimuth_centers = zg.centers()

    zdi_geometry.numerical_description(zg, lz)

    b_radial = lz.get_radial_field(polar_centers, azimuth_centers)
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
    b_radial = lz.as_dipole().get_radial_field(polar_centers, azimuth_centers)
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


def pole_walk(lz, zg=None, ax=None):
    """
    Forget this; the pole walks all over the place
    :param lz:
    :param zg:
    :return:
    """
    if ax is None:
        ax = plt.gca()
    if zg is None:
        zg = zdi_geometry.ZdiGeometry(128)

    polar_centers, azimuth_centers = zg.centers()

    # Dipole max
    b_radial_max_polar = []
    b_radial_max_azimuth = []
    b_radial_min_polar = []
    b_radial_min_azimuth = []

    for max_degree in range(lz.degree()):
        b_radial = lz.as_restricted(degree_l_range=(0, max_degree)).get_radial_field(polar_centers, azimuth_centers)
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


# TODO move to utils or similar.
def latex_float(value, pattern="{0:+.3g}"):
    float_str = pattern.format(value)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


