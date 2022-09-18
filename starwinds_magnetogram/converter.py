import numpy as np
import cmath
import logging


log = logging.getLogger(__name__)

import starwinds_magnetogram.coefficients as shc
from starwinds_magnetogram import pfss_magnetogram
from starwinds_magnetogram import zdi_magnetogram
from starwinds_magnetogram import associated_legendre


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
    positive orders without changing the magnetogram. This folding is possible because only the
    real components of the complex magnetograms are used.
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
    if len(zdi_radial_coeffs.default_coefficients) != 1:
        log.warning("Expected single complex coefficient values")
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
    # alpha = shc.scale(pfss_coeffs, forward_conversion_factor, -1)

    # This is very close or exact.
    beta = shc.scale(alpha, _beta_inverse_conversion_factor, 1)

    # The potential PFSS field has no toroidal component.
    gamma = shc.zeros_like(alpha)

    return shc.hstack((alpha, beta, gamma))


def convert_latlon_to_zdi(pl, az, field_r, field_polar, field_azimuthal, area=None, max_degree=5):

    coeffs = convert_latlon_to_zdi_coeffs(pl, az,
                            field_r, field_polar, field_azimuthal,
                            area, max_degree)

    return zdi_magnetogram.from_coefficients(coeffs)


def convert_latlon_to_zdi_coeffs(pl, az, field_r, field_polar, field_azimuthal, area=None, max_degree=5):

    degrees_l, orders_m = list(zip(*positive_lm(max_degree)))

    coeffs = get_zdi_coeffs(pl, az,
                            degrees_l, orders_m,
                            field_r, field_polar, field_azimuthal,
                            area)

    return coeffs


def get_zdi_coeffs(pl, az, degrees_l, orders_m, field_r, field_pl, field_az, area=None):

    degrees_l = np.atleast_1d(degrees_l)
    orders_m = np.atleast_1d(orders_m)
    orders_m_shape = orders_m.shape

    if pl.shape[0] <= pl.shape[1]:
        log.warning("Expected first coordinate to be azimuth, but second coordinate has more elements.")

    if area is None:
        # More like pseudo-area
        dpl = np.mean(np.diff(pl, axis=1))
        daz = np.mean(np.diff(az, axis=0))
        area = dpl * daz

    plmct, dplmct = associated_legendre.calculate_lpmn(degrees_l, orders_m, pl)  # Uses n for degree
    c_lm = zdi_magnetogram.get_c_lm(degrees_l, orders_m)

    Winv = 2.0 - (np.array(orders_m) == 0)
    assert Winv.shape == c_lm.shape
    Winv *= c_lm
    W = Winv**-1
    del Winv

    #
    # Set dimensions of all the arrays to [az, pl, number of lm pairs]
    #
    orders_m = np.asarray(orders_m)[np.newaxis, np.newaxis, ...]
    degrees_l = np.asarray(degrees_l)[np.newaxis, np.newaxis, ...]
    W = W[np.newaxis, np.newaxis, ...]

    az = az[..., np.newaxis]
    pl = pl[..., np.newaxis]
    field_r = field_r[..., np.newaxis]
    field_pl = field_pl[..., np.newaxis]
    field_az = field_az[..., np.newaxis]

    total_area = np.sum(np.sin(pl), axis=(0, 1)) * area
    sphere_coverage = total_area / (4 * np.pi)
    if sphere_coverage < .8:
        log.error("Sphere is not covered by area; check inputs.")

    #
    # Calculate alpha
    #
    for v in (orders_m, degrees_l, W, az, pl, field_r, field_pl, field_az):
        assert len(v.shape) == 3

    alpha_re = +np.sum(field_r * np.cos(orders_m * az) * plmct * np.sin(pl) / W, axis=(0, 1))
    alpha_im = -np.sum(field_r * np.sin(orders_m * az) * plmct * np.sin(pl) / W, axis=(0, 1))
    alpha = (alpha_re + 1.0j * alpha_im) * area

    #
    # Calculate beta
    #
    degrees_l0 = degrees_l.copy()
    degrees_l0[degrees_l0 == 0] = 1e12
    mplmct = orders_m * plmct

    # Vidotto et al. (2016) has different sign convention from Folsom et al. (2018).
    # B_polar sign is opposite, while B_azimuth sign is the same.
    # Therefore the sign of field_pl is switched here compared to Vidotto et al. (2016).
    beta_re = +np.sum((-field_pl * np.cos(orders_m * az) * dplmct * np.sin(pl) +
                       field_az * np.sin(orders_m * az) * mplmct) /
                      W / degrees_l0, axis=(0, 1))
    beta_im = -np.sum((-field_pl * np.sin(orders_m * az) * dplmct * np.sin(pl) -
                       field_az * np.cos(orders_m * az) * mplmct) /
                      W / degrees_l0, axis=(0, 1))
    beta = (beta_re + 1.0j * beta_im) * area


    #
    # Calculate gamma
    #
    gamma_re = -np.sum((-field_pl * np.sin(orders_m * az) * mplmct -
                        field_az * np.cos(orders_m * az) * dplmct * np.sin(pl)) /
                       W / degrees_l0, axis=(0, 1))
    gamma_im = -np.sum((-field_pl * np.cos(orders_m * az) * mplmct +
                        field_az * np.sin(orders_m * az) * dplmct * np.sin(pl)) /
                       W / degrees_l0, axis=(0, 1))
    gamma = (gamma_re + 1.0j * gamma_im) * area
    gamma *= -1  # TODO why is this required?

    # Return as coefficients object.
    degrees_l = degrees_l.reshape(orders_m_shape)
    orders_m = orders_m.reshape(orders_m_shape)

    return shc.from_arrays(degrees_l, orders_m,
                           np.stack((alpha, beta, gamma), axis=-1))


def positive_lm(max_degree):
    for l in range(0, max_degree + 1):
        for m in range(0, l + 1):
            yield l, m