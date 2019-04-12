import numpy as np
import logging
log = logging.getLogger(__name__)
import spherical_functions  # Loads slowly, so import only if needed.

import stellarwinds.magnetogram.spherical_harmonics_coefficients as shc


def rotate_magnetogram(magnetogram,
                       alpha=np.pi, beta=0, gamma=0):
    """
    Rotate magnetogram in a 3-1-3 Euler angle
    :param magnetogram:
    :param alpha:
    :param beta:
    :param gamma:
    :return: Rotated magnetogram
    """

    rotated_magnetogram = shc.empty_like(magnetogram)

    for deg_l_in in range(magnetogram.degree_max + 1):
        order_m_max = deg_l_in

        for order_m_out in range(-order_m_max, order_m_max + 1):

            coeff_out = magnetogram.default_coefficients

            for order_m_in in range(-order_m_max, order_m_max + 1):
                wde = spherical_functions.Wigner_D_element(alpha, beta, gamma, deg_l_in, order_m_in, order_m_out)
                coeff_out += wde * magnetogram.get(deg_l_in, order_m_in)

            rotated_magnetogram.append(deg_l_in, order_m_out, coeff_out)

    return rotated_magnetogram
