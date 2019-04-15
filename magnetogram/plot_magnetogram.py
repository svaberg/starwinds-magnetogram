import numpy as np
import scipy.special
import matplotlib
import matplotlib.pyplot as plt
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
            img = pretty_plot(geometry, B[component_id], ax_row[component_id],
                              vmin=-abs_max, vmax=abs_max)
            ax_row[component_id].set_ylabel(None)

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


def plot_pfss_slice(magnetogram, geometry=None, normal=np.array([0, 0, 1]), rmax=98,
                    radius_source_surface=3, radius_star=1):
    degree_l, order_m, alpha_lm = magnetogram.as_arrays(include_unset=False)

    # Polar angle (colatitude) and azimuth angle (longitude)
    _x = np.linspace(-1, 1, 302) * rmax
    _y = 0
    _z = np.linspace(-1, 1, 304) * rmax

    px, py, pz = np.meshgrid(_x, _y, _z)

    log.debug(_x.shape)

    assert px.shape == (1, len(_x), len(_z))  # np.meshgrid switches argument 1 and 2 in result.

    _shape = px.shape

    pr, pp, pa = coordinate_transforms.spherical_coordinates_from_rectangular(px, py, pz)

    field_radial, field_polar, field_azimuthal = pfss_stanford.evaluate_in_space(degree_l, order_m,
                                                                                 np.real(alpha_lm),
                                                                                 np.imag(alpha_lm),
                                                                                 pr, pp, pa,
                                                                                 radius_star=radius_star,
                                                                                 radius_source_surface=radius_source_surface)

    assert field_radial.shape == pr.shape
    assert field_polar.shape == pr.shape
    assert field_azimuthal.shape == pr.shape

    # assert np.all(field_polar[np.where(pr > rss)] == 0)
    # assert np.all(field_azimuthal[np.where(pr > rss)] == 0)

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

    fmag = (fx[0, :, :] ** 2 + fy[0, :, :] ** 2 + fz[0, :, :] ** 2) ** .5
    fmin = np.min(fmag[np.where(fmag > 0)])
    fmax = np.max(fmag)

    norm = matplotlib.colors.SymLogNorm(linthresh=100 * fmin,
                                 linscale=1,
                                 vmin=-fmax,
                                 vmax=fmax)

    im = ax.pcolormesh(px[0, :, :], pz[0, :, :], field_radial[0, :, :],
                       norm=norm,
                       cmap='RdBu_r')

    fig.colorbar(im).set_label('Radial field strength')

    ax.contour(px[0, :, :], pz[0, :, :], pr[0, :, :], levels=[radius_star, radius_source_surface], colors='k')

    ax.streamplot(px[0, :, :].transpose(), pz[0, :, :].transpose(),
                  fx[0, :, :].transpose(), fz[0, :, :].transpose(),
                  color='gray')

    ax.set_aspect('equal')
    ax.set_xlabel('Distance $x/r_{\\star}$')
    ax.set_ylabel('Distance $z/r_{\\star}$')

    return fig, ax

def pretty_plot(geometry, value, ax, vmin=None, vmax=None, color_map='RdBu_r'):

    centers_polar, centers_azimuth = geometry.centers()
    corners_polar, corners_azimuth = geometry.corners()

    img = ax.pcolormesh(np.rad2deg(corners_azimuth.T),
                      np.rad2deg(corners_polar.T),
                      value.T,
                      cmap=color_map,
                      vmin=vmin, vmax=vmax)

    if np.min(value) < 0 < np.max(value):
        ax.contour(180 / np.pi * centers_azimuth.T, 180 / np.pi * centers_polar.T, value.T, levels=[0], colors=('g',), linewidths=.25)

    ax.set_xticks(180 / np.pi * np.linspace(corners_azimuth[0, 0], corners_azimuth[-1, -1], 9))
    ax.set_xlabel('Azimuth angle $\phi$')
    ax.set_yticks(180 / np.pi * np.linspace(corners_polar[0, 0], corners_polar[-1, -1], 5))
    ax.set_ylabel('Polar angle $\\theta$')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    # ax.invert_xaxis() # To look more like Matthew
    ax.grid()

    return img