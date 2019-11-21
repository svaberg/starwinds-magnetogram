import numpy as np
import scipy.special
import logging


log = logging.getLogger(__name__)

# Test "context"
from tests import context  # Test context
import pytest
from tests.magnetogram import magnetograms

# Local
import stellarwinds.magnetogram.pfss_magnetogram as pfss_stanford
from stellarwinds.magnetogram import plots  #TODO: remove this.
from stellarwinds.magnetogram import plot_pfss
from stellarwinds.magnetogram import geometry


@pytest.mark.skip(reason="Never worked...")
def test_reference(request,
                   magnetogram_name="mengel"):

    magnetogram = magnetograms.get_radial(magnetogram_name)
    degree_l, order_m, alpha_lm = magnetogram.as_arrays(include_unset=False)
    polar, azimuth = geometry.ZdiGeometry().centers()

    B1 = evaluate_real_magnetogram_stanford_pfss_reference(
        degree_l,
        order_m,
        np.real(alpha_lm),
        np.imag(alpha_lm),
        polar, azimuth)


    B2 = pfss_stanford.evaluate_spherical(
        magnetogram,
        1, polar, azimuth)

    assert(np.allclose(B2[0], B1[0]))
    log.debug("Implementations match for B_radial.")

    assert(np.allclose(B2[1], B1[1]))
    log.debug("Implementations match for B_polar.")

    log.debug('B1 azimuth component range [%g, %g]' % (np.min(B1[2]), np.max(B1[2])))
    log.debug('B2 azimuth component range [%g, %g]' % (np.min(B2[2]), np.max(B2[2])))

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        for cid, name in enumerate(("radial", "polar", "azimuth")):
            fig, axs = plt.subplots(2, 2)

            img1 = axs[0, 0].pcolormesh(np.rad2deg(azimuth), np.rad2deg(polar), B1[cid], cmap='RdBu_r')
            fig.colorbar(img1, ax=axs[0, 0])

            img2 = axs[0, 1].pcolormesh(np.rad2deg(azimuth), np.rad2deg(polar), B2[cid], cmap='RdBu_r')
            fig.colorbar(img2, ax=axs[0, 1])

            img10 = axs[1, 0].pcolormesh(np.rad2deg(azimuth), np.rad2deg(polar), B2[cid]-B1[cid])
            fig.colorbar(img10, ax=axs[1, 0])

            img11 = axs[1, 1].pcolormesh(np.rad2deg(azimuth), np.rad2deg(polar), B2[cid]/B1[cid], vmin=-2, vmax=2)
            fig.colorbar(img11, ax=axs[1, 1])

            fig.suptitle('Field %s component' % name)

            for ax in axs.ravel():
                ax.invert_xaxis()
                ax.invert_yaxis()

            plt.savefig(pn.get())

    # Why do these not match??
    assert(np.allclose(B2[2], B1[2]))


@pytest.mark.parametrize("points_shape", ((1,), (2,), (2, 3), (2, 3, 5), (5, 8, 2, 2, 1, 3)))
@pytest.mark.parametrize("magnetogram_name", ("mengel",))
def test_evaluate_cartesian(request,
                            points_shape,
                            magnetogram_name):

    coeffs = magnetograms.get_radial(magnetogram_name)

    px = 1 + np.random.rand(*points_shape)
    py = 1 + np.random.rand(*points_shape)
    pz = 1 + np.random.rand(*points_shape)

    fr, fp, fa, fx, fy, fz = pfss_stanford.evaluate_cartesian(
        coeffs,
        px, py, pz)

    assert fr.shape == points_shape


@pytest.mark.parametrize("points_shape", ((1,), (2,), (2, 3), (2, 3, 5), (5, 8, 2, 2, 1, 3)))
@pytest.mark.parametrize("magnetogram_name", ("mengel",))
def test_evaluate_spherical(request,
                            points_shape,
                            magnetogram_name):

    coeffs = magnetograms.get_radial(magnetogram_name)

    pr = 1 + np.random.rand(*points_shape)
    pp = np.pi * np.random.rand(*points_shape)
    pa = 2 * np.pi * np.random.rand(*points_shape)

    fr, fp, fa = pfss_stanford.evaluate_spherical(
        coeffs,
        pr, pp, pa)

    assert fr.shape == points_shape


def test_plot_pfss_equirectangular(request,
                           magnetogram_name="mengel"):

    magnetogram = magnetograms.get_radial(magnetogram_name)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, axs = plot_pfss.plot_equirectangular(magnetogram)
        plt.savefig(pn.get())


def test_plot_pfss_slice(request,
                      magnetogram_name="dipole"):

    magnetogram = magnetograms.get_radial(magnetogram_name)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, axs = plot_pfss.plot_slice(magnetogram)
        fig.savefig(pn.get())


def test_plot_pfss_magnitudes(request,
                magnetogram_name="mengel"):

    _geometry = geometry.ZdiGeometry()
    coefficients = magnetograms.get_radial(magnetogram_name)
    polar, azimuth = geometry.ZdiGeometry().centers()

    radius_source_surface = 3

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        fig, axs = plt.subplots(1, 2)

        for radius, ax in zip((1, 3), axs):
            field_rpa = pfss_stanford.evaluate_spherical(
                coefficients,
                radius, polar, azimuth,
                radius_star=1,
                radius_source_surface=radius_source_surface)
            field_magnitude = np.sqrt(np.sum([f**2 for f in field_rpa], axis=0))
            img = plots.plot_equirectangular(_geometry, field_magnitude, ax, cmap='viridis')
            fig.colorbar(img, ax=ax, orientation='horizontal')
            ax.set_title("Bmag at r=%2.1f" % radius)

        plt.savefig(pn.get())


def test_plot_pfss_quiver(request,
                          magnetogram_name="mengel"):

    _geometry = geometry.ZdiGeometry()
    magnetogram = magnetograms.get_radial(magnetogram_name)
    polar, azimuth = geometry.ZdiGeometry().centers()

    field_rpa = pfss_stanford.evaluate_spherical(
        magnetogram,
        1, polar, azimuth)

    field_pa_mag = np.sqrt(field_rpa[1] ** 2 + field_rpa[2] ** 2)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        fig, ax = plt.subplots()
        hstride = int(2);
        vstride = int(hstride / 2);
        img = plots.plot_equirectangular(_geometry, field_pa_mag, ax, cmap='viridis')
        fig.colorbar(img, ax=ax, orientation='horizontal')

        ax.quiver(np.rad2deg(azimuth[::hstride, ::vstride]),
                  np.rad2deg(polar[::hstride, ::vstride]),
                  field_rpa[2][::hstride, ::vstride],
                  field_rpa[1][::hstride, ::vstride])

        ax.set_title("Surface tangential field strength")

        plt.savefig(pn.get())


def test_plot_pfss_streamtraces(request,
                                magnetogram_name="mengel"):

    magnetogram = magnetograms.get_radial(magnetogram_name)
    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, ax = plot_pfss.plot_streamtraces(magnetogram)
        fig.savefig(pn.get())


# TODO which reference implementation is this? Do we even have a reference implementation?
def evaluate_real_magnetogram_stanford_pfss_reference(degree_l, order_m, cosine_coefficients_g, sine_coefficients_h,
                                                      points_polar, points_azimuth, radius=None, r0=1, rss=3):
    if radius is None:
        radius = r0

    assert np.min(order_m) >= 0, "Stanford PFSS expects only positive orders (TBC)."
    field_radial = np.zeros_like(points_azimuth)
    field_polar = np.zeros_like(field_radial)
    field_azimuthal = np.zeros_like(field_radial)

    for row_id in range(len(degree_l)):

        deg_l = degree_l[row_id]
        ord_m = order_m[row_id]
        g_lm = cosine_coefficients_g[row_id]
        h_lm = sine_coefficients_h[row_id]

        p_lm = scipy.special.lpmv(ord_m, deg_l, np.cos(points_polar))

        if True:  # Apply correction
            # https://en.wikipedia.org/wiki/Spherical_harmonics#Condon%E2%80%93Shortley_phase
            # https://en.wikipedia.org/wiki/Spherical_harmonics#Conventions
            d0 = 0 + (ord_m == 0)
            p_lm *= (-1) ** ord_m * np.sqrt(scipy.special.factorial(deg_l - ord_m) / scipy.special.factorial(deg_l + ord_m)) * np.sqrt(2 - d0)

        # Estimate DPml
        DPml = (p_lm - np.roll(p_lm, 1)) / (np.cos(points_polar) - np.roll(np.cos(points_polar), 1)) * np.sin(points_polar)

        fixed = (r0 / radius) ** (deg_l + 2) / (deg_l + 1 + deg_l * (r0 / rss) ** (2 * deg_l + 1))

        field_radial += p_lm * (g_lm * np.cos(ord_m * points_azimuth) + h_lm * np.sin(ord_m * points_azimuth)) * (
                deg_l + 1 + deg_l * (radius / rss) ** (2 * deg_l + 1)) * fixed
        field_polar -= DPml * (g_lm * np.cos(ord_m * points_azimuth) + h_lm * np.sin(ord_m * points_azimuth)) * (
                1 - (radius / rss) ** (2 * deg_l + 1)) * fixed
        field_azimuthal -= p_lm * (g_lm * np.sin(ord_m * points_azimuth) - h_lm * np.cos(ord_m * points_azimuth)) * (
                1 - (radius / rss) ** (2 * deg_l + 1)) * fixed

    return field_radial, field_polar, field_azimuthal

