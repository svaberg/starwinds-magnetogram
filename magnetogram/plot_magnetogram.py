import numpy as np
import scipy.special
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

import logging
log = logging.getLogger(__name__)

from stellarwinds.magnetogram import pfss_stanford
from stellarwinds.magnetogram import zdi_geometry
from stellarwinds import coordinate_transforms


def plot_pfss_spheres(magnetogram, geometry=None, radius_source_surface=3):

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

    fig, axs_mat = plt.subplots(2, 3)
    fig.subplots_adjust(right=0.8)  # Make space for colorbar.

    for ax_row, B, radius in zip(axs_mat, (Bs, Bss), (1, radius_source_surface)):
        abs_max = np.max(np.abs(B))
        for component_id in range(len(B)):
            img = plot_equirectangular(geometry, B[component_id], ax_row[component_id],
                                       vmin=-abs_max, vmax=abs_max)
            ax_row[component_id].set_ylabel(None)

        # TODO check out make_axes_locatable
        def place_colorbar_axis_right(ax, dx=.22):
            p0 = ax.get_position().p0
            p1 = ax.get_position().p1

            cbar_ax = fig.add_axes((p0[0] + dx, p0[1], .01, p1[1]-p0[1]))
            return cbar_ax

        cax = place_colorbar_axis_right(ax_row[2])
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

    norm = matplotlib.colors.SymLogNorm(linthresh=100 * fmin,
                                        linscale=1,
                                        vmin=-fmax,
                                        vmax=fmax)

    im = ax.pcolormesh(p1, p2, np.squeeze(field_radial),
                       norm=norm,
                       cmap='RdBu_r')

    fig.colorbar(im).set_label('Radial field strength')

    radial_distance = np.squeeze(pr)
    ax.contour(p1[:, :], p2[:, :], radial_distance, levels=[radius_star, radius_source_surface], colors='k')


    # To draw streamplot, project f into the p plane.

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