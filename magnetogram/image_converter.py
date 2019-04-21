import numpy as np
from scipy.special import sph_harm

from stellarwinds.magnetogram import geometry
from stellarwinds.magnetogram import coefficients


def from_image(image, zg=None, lmax=15):
    """
    Get spherical harmonics coefficients from an image matrix
    :param image: image matrix
    :param zg: geometry object
    :param lmax: maximum degree
    :return: spherical harmonics coefficients
    """

    if zg is None:
        num_corners = [f+1 for f in image.shape]
        zg = geometry.ZdiGeometry(*num_corners[::-1])

    polar, azimuth = zg.centers()
    delta_polar, delta_azimuth = _deltas(zg)

    coeffs = coefficients.Coefficients()
    for l, m in _indices(lmax):
        ylm = sph_harm(m, l, azimuth, polar)
        flm = np.sum(image * np.conj(ylm) * np.sin(polar) * delta_azimuth * delta_polar)
        coeffs.append(l, m, flm)

    return coeffs, zg


def to_image(coeffs, zg):
    """
    Get image matrix from spherical harmonics coefficients
    :param coeffs: spherical harmonics coefficients
    :param zg: geometry
    :return: image matrix
    """
    polar, azimuth = zg.centers()
    image = np.zeros_like(azimuth, dtype=complex)
    for (deg_l, ord_m), val in coeffs.contents():
        image += val * sph_harm(ord_m, deg_l, azimuth, polar)

    return image


def _deltas(zg):
    """
    Get matrix containing the length of each integration interval.
    TODO how much faster returning 1d arrays would be in from_image.
    :param zg: geometry
    :return: delta polar, delta azimuth
    """
    polar_corners, azimuth_corners = zg.corners()
    delta_polar = np.diff(polar_corners, axis=1)
    delta_azimuth = np.diff(azimuth_corners, axis=0)

    assert np.allclose(delta_polar, delta_polar[0, :])
    assert np.allclose(delta_azimuth, delta_azimuth[:, 0][:, np.newaxis])

    return delta_polar[1:, ...], delta_azimuth[..., 1:]


def _indices(lmax=15):
    """
    Yield full set of indices up to lmax
    :param lmax: maximum degree
    :return: degree, order
    """
    for deg_l in range(0, lmax):
        for ord_m in range(-deg_l, deg_l + 1):
            yield deg_l, ord_m

