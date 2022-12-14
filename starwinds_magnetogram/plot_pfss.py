import numpy as np
from matplotlib import pyplot as plt, colors
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from starwinds_magnetogram import pfss_magnetogram
from starwinds_magnetogram import plots
from starwinds_magnetogram.geometry import ZdiGeometry


#TODO rename this to plot_vector_field or something.
def plot_components(coefficients, geometry=None, radius_source_surface=None, axs=None):
    r"""
    Plot a 2-by-3 arangement of PFSS field components at the stellar surface and
    at the source surface.
    :param coefficients:
    :param geometry:
    :param radius_source_surface:
    :return:
    """

    if geometry is None:
        geometry = ZdiGeometry()

    if radius_source_surface is None:
        radius_source_surface = pfss_magnetogram.default_radius_source_surface

    if axs is None:
        _, axs = plt.subplots(1, 3, figsize=(12, 5))

    #
    # Evaluate magnetic field at coordinates.
    #
    polar, azimuth = geometry.centers()

    Bs = pfss_magnetogram.evaluate_spherical(
        coefficients,
        1, polar, azimuth,
        radius_star=1,
        radius_source_surface=radius_source_surface)

    plots.plot_components(polar, azimuth, Bs, axs)

    return axs[0].figure, axs


def plot_slice(coefficients, normal="x", rmax=5,
               radius_source_surface=None, radius_star=None):

    if radius_star is None:
        radius_star = pfss_magnetogram.default_radius_star

    if radius_source_surface is None:
        radius_source_surface = pfss_magnetogram.default_radius_source_surface

    assert radius_star < radius_source_surface

    # Points in slice plane xy coordinate system.
    p1 = np.linspace(-1, 1, 302) * rmax
    p2 = np.linspace(-1, 1, 304) * rmax
    p1, p2 = np.meshgrid(p1, p2)

    px, py, pz = pfss_magnetogram.normal_plane(p1, p2, normal)

    fr, fp, fa, fx, fy, fz = pfss_magnetogram.evaluate_cartesian(coefficients, px, py, pz,
                                                                 radius_star, radius_source_surface)

    #
    # Plot stars here...
    #
    fig, ax = plt.subplots(figsize=(9, 6))
    fmag = (fx ** 2 + fy ** 2 + fz ** 2) ** .5
    fmin = np.min(fmag[np.where(fmag > 0)])
    fmax = np.max(fmag)
    norm = colors.SymLogNorm(linthresh=100 * fmin,
                             linscale=1,
                             vmin=-fmax,
                             vmax=fmax,
                             base=10)
    im = ax.pcolormesh(p1, p2, np.squeeze(fr),
                       norm=norm,
                       cmap='RdBu_r')
    fig.colorbar(im).set_label('Radial field strength')

    ax.set_aspect('equal')
    ax.set_xlabel(r'Distance $x/r_{\star}$')
    ax.set_ylabel(r'Distance $z/r_{\star}$')

    # Add star and source surface.
    radial_distance = np.squeeze((px**2 + py**2 + pz**2)**.5)
    ax.contour(p1[:, :], p2[:, :], radial_distance, levels=[radius_star, radius_source_surface], colors='k')

    # Add field lines:
    ax.streamplot(py[..., 0], pz[..., 0], fy[..., 0], fz[..., 0], color='k')

    return fig, ax


def plot_streamtraces(coefficients, geometry=ZdiGeometry()):

    polar, azimuth = geometry.centers()

    B_radial, B_polar, B_azimuthal = pfss_magnetogram.evaluate_spherical(
        coefficients,
        1, polar, azimuth)

    B_tangential_mag = np.sqrt(B_polar ** 2 + B_azimuthal ** 2)
    B_radial_abs_max = np.max(np.abs(B_radial))

    fig, ax = plt.subplots(figsize=(18, 6))

    img = plots.plot_equirectangular(geometry, B_radial, ax,
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
    legend_strs = [r"Radial $B_r = 0$", r"Polar $B_\theta = 0$", r"Azimuth $B_\phi = 0$"]

    locator = MaxNLocator(nbins=3, prune="lower")
    line_values = locator.tick_values(np.min(B_tangential_mag), np.max(B_tangential_mag))
    line_thicknesses = 3 * line_values / np.max(line_values)

    custom_lines = [Line2D([0], [0], color='k', lw=l) for l in line_thicknesses]
    custom_strs = [r"$B_\perp=%4.1f$" % x for x in line_values]
    plt.legend(legend_items + custom_lines, legend_strs + custom_strs)

    return fig, ax
