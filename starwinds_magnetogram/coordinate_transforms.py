import numpy as np
import logging
log = logging.getLogger(__name__)


def spherical_coordinates_from_rectangular(px, py, pz):
    """
    Return point polar coordinates.
    Where pr = 0, set polar angle to the arbitrarily chosen value 0.
    The polar values range (0, pi), and the azimuth values range (-pi, pi)

    Note: The azimuth values are (-pi, pi) and the azimuth jump is at -x
    # TODO is this the best azimuth convention?

    :param px: x coordinates
    :param py: y coordinates
    :param pz: z coordinates
    :return: radial, polar and azimuth coordinates
    """
    pr = (px**2 + py**2 + pz**2)**(1/2)
    pp = np.arccos(np.divide(pz, pr, out=np.zeros_like(pr), where=pr != 0))
    pa = np.arctan2(py, px)

    assert not np.any(np.isnan(pp))
    assert not np.any(np.isnan(pa))

    return pr, pp, pa


def rectangular_coordinates_from_spherical(pr, pp, pa):
    """

    :param pr:
    :param pp:
    :param pa:
    :return:
    """
    px = pr * np.sin(pp) * np.cos(pa)
    py = pr * np.sin(pp) * np.sin(pa)
    pz = pr * np.cos(pp)

    assert not np.any(np.isnan(px))
    assert not np.any(np.isnan(py))
    assert not np.any(np.isnan(pz))

    return px, py, pz


def spherical_to_rectangular_transformation_matrix(polar_theta, azimuthal_phi):
    """
    Follow convention where
    the polar angle is $theta$
    the azimuthal angle is $phi$
    :param polar_theta:
    :param azimuthal_phi:
    :return:
    """
    polar_theta = np.atleast_1d(polar_theta)
    azimuthal_phi = np.atleast_1d(azimuthal_phi)

    # Intermediate quantities cos/sin theta/phi
    sin_theta = np.sin(polar_theta)
    cos_theta = np.cos(polar_theta)
    sin_phi = np.sin(azimuthal_phi)
    cos_phi = np.cos(azimuthal_phi)

    # Calculate transformation matrix
    # Note that the inverse is also the transpose
    transformation_matrix = np.zeros((len(polar_theta), 3, 3))

    transformation_matrix[:, 0, 0] = sin_theta * cos_phi
    transformation_matrix[:, 0, 1] = cos_theta * cos_phi
    transformation_matrix[:, 0, 2] =            -sin_phi

    transformation_matrix[:, 1, 0] = sin_theta * sin_phi
    transformation_matrix[:, 1, 1] = cos_theta * sin_phi
    transformation_matrix[:, 1, 2] =             cos_phi

    transformation_matrix[:, 2, 0] = cos_theta
    transformation_matrix[:, 2, 1] = -sin_theta
    transformation_matrix[:, 2, 2] = 0

    return transformation_matrix


def rectangular_to_spherical_transformation_matrix(polar_theta, azimuthal_phi):
    """

    :param polar_theta:
    :param azimuthal_phi:
    :return:
    """
    polar_theta = np.atleast_1d(polar_theta)
    azimuthal_phi = np.atleast_1d(azimuthal_phi)

    # Intermediate quantities cos/sin polar_theta/azimuthal_phi
    sin_theta = np.sin(polar_theta)
    cos_theta = np.cos(polar_theta)
    sin_phi = np.sin(azimuthal_phi)
    cos_phi = np.cos(azimuthal_phi)

    # Calculate transformation matrix
    transformation_matrix = np.zeros((len(polar_theta), 3, 3))
    transformation_matrix[:, 0, 0] = sin_theta * cos_phi
    transformation_matrix[:, 0, 1] = sin_theta * sin_phi
    transformation_matrix[:, 0, 2] = cos_theta

    transformation_matrix[:, 1, 0] = cos_theta * cos_phi
    transformation_matrix[:, 1, 1] = cos_theta * sin_phi
    transformation_matrix[:, 1, 2] = -sin_theta

    transformation_matrix[:, 2, 0] = -sin_phi
    transformation_matrix[:, 2, 1] = cos_phi
    transformation_matrix[:, 2, 2] = 0

    return transformation_matrix
