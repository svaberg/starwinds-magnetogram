import numpy as np
import logging
log = logging.getLogger(__name__)
# The idea is to only import these two in this file.
# Loads slowly, so import only if needed.
import quaternion
import spherical_functions

import stellarwinds.magnetogram.spherical_harmonics_coefficients as shc


def rotate_magnetogram(magnetogram, rotation):
    """
    Rotate magnetogram.
    :param magnetogram:
    :param rotation:
    :return: Rotated magnetogram
    """

    rotated_magnetogram = shc.empty_like(magnetogram)

    for deg_l_in in range(magnetogram.degree_max + 1):
        order_m_max = deg_l_in

        for order_m_out in range(-order_m_max, order_m_max + 1):

            coeff_out = magnetogram.default_coefficients

            for order_m_in in range(-order_m_max, order_m_max + 1):
                wde = spherical_functions.Wigner_D_element(rotation, deg_l_in, order_m_in, order_m_out)
                coeff_out += wde * magnetogram.get(deg_l_in, order_m_in)

            rotated_magnetogram.append(deg_l_in, order_m_out, coeff_out)

    return rotated_magnetogram


def rotate_magnetogram_euler_zyz_deg(magnetogram, euler_zyz_deg):
    """

    :param magnetogram:
    :param euler_zyz_deg:
    :return:
    """
    rotation_quaternion = quaternion.from_euler_angles(np.deg2rad(euler_zyz_deg))
    return rotate_magnetogram(magnetogram, rotation_quaternion)
