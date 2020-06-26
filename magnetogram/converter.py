import numpy as np
import cmath
import logging

log = logging.getLogger(__name__)

import stellarwinds.magnetogram.coefficients as shc
from stellarwinds.magnetogram import pfss_magnetogram

def collect_cosines(r, alpha, s, beta):
    """

    :param r:
    :param alpha:
    :param s:
    :param beta:
    :return:
    """
    cos_terms = r * np.cos(alpha) + s * np.cos(beta)
    sin_terms = r * np.sin(alpha) - s * np.sin(beta)

    t = np.sqrt(cos_terms**2 + sin_terms**2)
    gamma = np.arctan2(sin_terms, cos_terms)

    return t, gamma


def collect_sines(r, alpha, s, beta):
    """

    :param r:
    :param alpha:
    :param s:
    :param beta:
    :return:
    """
    cos_terms = r * np.cos(alpha) - s * np.cos(beta)
    sin_terms = r * np.sin(alpha) + s * np.sin(beta)

    t = np.sqrt(cos_terms**2 + sin_terms**2)
    gamma = np.arctan2(sin_terms, cos_terms)

    return t, gamma


def map_to_positive_orders(magnetogram):
    """
    After rotation, a set of coefficients may have nonzero negative orders; this is normally
    not part of the ZDI magnetogram definition. Use this to "fold" the negative orders onto the
    positive orders without changing the magnetogram.
    :param magnetogram:
    :return:
    """
    output = shc.empty_like(magnetogram)
    for degree_l in range(magnetogram.degree_max + 1):
        output.append(degree_l, 0, magnetogram.get(degree_l, 0))
        for order_m in range(1, degree_l + 1):  # No need to map m=0 as it has no negative partner.
            c_pos = magnetogram.get(degree_l, + order_m)
            c_neg = magnetogram.get(degree_l, - order_m)

            r, alpha = cmath.polar(c_pos)
            s, beta  = cmath.polar(c_neg)

            t, gamma = collect_cosines(r, alpha, (-1) ** order_m * s, beta)
            # t, gamma = collect_cosines(r, alpha, s, beta)
            # log.debug('collect_c', (r, alpha), (s, beta), (t, gamma))

            c = cmath.rect(t, gamma)
            output.append(degree_l, order_m, c)
    return output


def forward_conversion_factor(degree_l, order_m):
    """
    Conversion function from zdipy format to wso format
    :param degree_l:
    :param order_m:
    :return:
    """
    # Calculate the complex-to-real rescaling factor $\sqrt{2-\delta_{m,0}}$
    #
    #  The Dirac delta function $\delta_{m0}$ has
    # $\delta_{m,0} = 1$ for $m = 0$ and
    # $\delta_{m,0} = 0$ for $m \neq 0$
    delta_m0 = np.where(order_m == 0, 1, 0)
    complex_to_real_rescaling = np.sqrt(-delta_m0+2)

    # Calculate the value of the Corton-Shortley phase, $(-1)^m$
    corton_shortley_phase = (-1)**(order_m % 2)

    # Calculate the unit sphere area compensation $\sqrt{4\pi}$
    unit_sphere_factor = np.sqrt(4.0 * np.pi)

    # Calculate the Schmidt scaling factor $\sqrt{2\ell+1}$
    schmidt_scaling = np.sqrt(2 * degree_l + 1)

    # The full conversion factor.
    conversion_factor = schmidt_scaling / (corton_shortley_phase *
                                           complex_to_real_rescaling *
                                           unit_sphere_factor)

    return conversion_factor


def convert_zdi_to_pfss(zdi_radial_coeffs):
    """
    Convert from ZDI to WSO PFSS format.
    :param zdi_radial_coeffs: Radial ZDI coefficents.
    :return: Coefficients in WSO PFSS format.
    """
    pfss_coeffs = zdi_radial_coeffs.scale(forward_conversion_factor, 1)
    return pfss_coeffs


def _beta_inverse_conversion_factor(degree_l, order_m):
    R, _ = pfss_magnetogram.r_l(degree_l, 1, 1, 3)
    return R * (degree_l + 1)


def convert_pfss_to_zdi(pfss_coeffs):
    """
    Convert from WSO PFSS format to ZDI.
    :param pfss_coeffs:
    :return:
    """
    # The alpha values are easy, just invert the forward conversion.
    alpha = pfss_coeffs.scale(forward_conversion_factor, -1)

    # This is very close or exact.
    beta = shc.scale(alpha, _beta_inverse_conversion_factor, 1)

    # The potential PFSS field has no toroidal component.
    gamma = shc.zeros_like(alpha)

    return shc.hstack((alpha, beta, gamma))