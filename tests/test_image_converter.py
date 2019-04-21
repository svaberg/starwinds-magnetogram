import logging
log = logging.getLogger(__name__)
import numpy as np

import pytest
from tests import context  # Test context

from stellarwinds.magnetogram import geometry
from stellarwinds.magnetogram import plots
from stellarwinds.magnetogram import image_converter

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
