import logging
log = logging.getLogger(__name__)
import numpy as np
from scipy.special import sph_harm
import time
import functools

from stellarwinds.magnetogram import geometry
from stellarwinds.magnetogram import coefficients


_degree_l_max = 30  # Highest expected harmonic degree


class TimeLogger:

    def __init__(self, msg, log=log):
        self.msg = msg
        self.log = log

    def __enter__(self):
        self.log.debug("Timing %s..." % self.msg)
        self.start_time = time.time()

    def __exit__(self, *args):
        elapsed_time = time.time() - self.start_time
        self.log.info("%s in %f seconds" % (self.msg, elapsed_time))


def from_image(image, zg=None, lmax=_degree_l_max):
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

    with TimeLogger("Built coefficients", log):
        # These are the same for all $\ell, m$.
        fixed_factors = image * np.sin(polar) * delta_azimuth * delta_polar

        coeffs = coefficients.Coefficients()

        for l, m in _indices(lmax):
            ylm = spherical_harmonic(m, l, azimuth, polar)  # This is the slow call, not the looping.
            flm = np.sum(np.conj(ylm) * fixed_factors)
            coeffs.append(l, m, flm)

    log.debug(_cached_spherical_harmonic.cache_info())
    return coeffs, zg


def to_image(coeffs, zg):
    """
    Get image matrix from spherical harmonics coefficients
    :param coeffs: spherical harmonics coefficients
    :param zg: geometry
    :return: image matrix
    """
    with TimeLogger("Built image", log) as t:
        polar, azimuth = zg.centers()
        image = np.zeros_like(azimuth, dtype=complex)
        for (deg_l, ord_m), val in coeffs.contents():
            image += val * spherical_harmonic(ord_m, deg_l, azimuth, polar)

    log.debug(_cached_spherical_harmonic.cache_info())
    return image


@functools.lru_cache(maxsize=_degree_l_max ** 2)
def _cached_spherical_harmonic(ord_m, deg_l, az_tuple, pl_tuple):
    pl = np.array(pl_tuple)
    az = np.array(az_tuple)
    az = az[..., np.newaxis]
    return sph_harm(ord_m, deg_l, az, pl)


def spherical_harmonic(ord_m, deg_l, azimuth, polar, use_cache=True):

    if use_cache:
        azimuth = np.atleast_2d(azimuth)
        polar = np.atleast_2d(polar)

        azimuth_tuple = tuple(azimuth[..., 0])
        polar_tuple = tuple(polar[0, ...])

        ylm = _cached_spherical_harmonic(ord_m, deg_l, azimuth_tuple, polar_tuple)

        return ylm
    else:
        return sph_harm(ord_m, deg_l, azimuth, polar)


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


def _indices(lmax=_degree_l_max):
    """
    Yield full set of indices up to lmax
    :param lmax: maximum degree
    :return: degree, order
    """
    for deg_l in range(0, lmax):
        for ord_m in range(-deg_l, deg_l + 1):
            yield deg_l, ord_m

