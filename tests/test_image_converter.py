import logging
log = logging.getLogger(__name__)
import numpy as np

import pytest
from tests import context  # Test context

from starwinds_magnetogram import geometry
from starwinds_magnetogram import plots
from starwinds_magnetogram import image_converter

from matplotlib.image import imread

_zgs = (
    geometry.ZdiGeometry(np.geomspace(1, np.pi + 1) - 1, np.geomspace(1, 2 * np.pi + 1) - 1),
    geometry.ZdiGeometry(),
)


@pytest.mark.parametrize("zg", _zgs)
def test_image(request, zg):

    polar, azimuth = zg.centers()
    image_in = np.ones_like(polar)
    image_in[polar > azimuth] = -1

    coeffs, _ = image_converter.from_image(image_in, zg)

    image_out = np.real(image_converter.to_image(coeffs, zg))

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        vmax = np.maximum(np.max(image_in), np.max(image_out))
        vmin = np.minimum(np.min(image_in), np.min(image_out))

        fig, axs = plt.subplots(2, 1, figsize=(8, 8))

        for data, ax in zip((image_in, image_out), axs):
            img = plots.plot_equirectangular(zg, np.real(data),
                                             ax,
                                             vmin=vmin,
                                             vmax=vmax,
                                             cmap='viridis')

        fig.colorbar(img, ax=axs.ravel().tolist())
        fig.savefig(pn.get())


def test_scipy_lpmv(request):
    from scipy.special import lpmv
    from scipy.special import sph_harm
    from scipy.special import factorial

    l, m = zip(*image_converter._indices(30))

    zg = geometry.ZdiGeometry(120, 60)

    phi_pl, theta_az = zg.centers()

    l = np.reshape(l, (-1, 1, 1))
    m = np.reshape(m, (-1, 1, 1))

    with image_converter.TimeLogger("split", log) as t:
        clm = np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - m) / factorial(l + m))
        expimtheta = np.exp(1j * m * theta_az)
        plmcosphi = lpmv(m, l, np.cos(phi_pl))
        ylm_a = clm * expimtheta * plmcosphi

    with image_converter.TimeLogger("sph_harm", log):
        ylm_b = sph_harm(m, l, theta_az, phi_pl)
    #
    # vmin, vmax = -.3,.3
    # with context.PlotNamer(__file__, request.node.name) as (pn, plt):
    #
    #     fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    #
    #     for data, ax in zip((ylm_a, ylm_b), axs[0]):
    #         img = plots.plot_equirectangular(zg, np.real(data[1, ...]),
    #                                          ax,
    #                                          vmin=vmin,
    #                                          vmax=vmax,
    #                                          cmap='viridis')
    #
    #     for data, ax in zip((ylm_a, ylm_b), axs[1]):
    #         img = plots.plot_equirectangular(zg, np.imag(data[1, ...]),
    #                                          ax,
    #                                          vmin=vmin,
    #                                          vmax=vmax,
    #                                          cmap='viridis')
    #
    #     fig.colorbar(img, ax=axs.ravel().tolist())
    #     fig.savefig(pn.get())
    #
    #

    assert np.allclose(ylm_a, ylm_b)

