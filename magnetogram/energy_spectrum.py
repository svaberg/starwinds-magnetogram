import numpy as np
import logging

from . import convert_magnetogram
import matplotlib.pyplot as plt
from scipy import stats

log = logging.getLogger(__name__)

# TODO this is a bit obsolete as there is a new program convert_magnetogram.
def normalisation_none(degree_l, order_m):
    r"""
    Placeholder normalisation that does nothing
    :param degree_l:
    :param order_m:
    :return:
    """
    return 0 * degree_l + 0*order_m + 1


def normalisation_schmidt(degree_l, order_m):
    r"""
    Schmidt normalisation $\frac{1}{2\ell +1}$
    :param degree_l: Degree $\ell$ of harmonic coefficient
    :param order_m: Order $m$ of harmonic coefficient
    :return: Schmidt scaling value for degree_l
    """
    return (2.0 * degree_l + 1.0) ** -1.0


def normalisation_zdipy(degree_l, order_m):
    r"""
    As we understand zdipy does not use Schmidt scaling but
    :param degree_l:
    :param order_m:
    :return:
    """
    return ((4 * np.pi) * np.where(order_m == 0, 1, 2))**-1.0


def energy_spectrum(degree_l, order_m, g_lm, h_lm, normalisation=normalisation_zdipy):
    """

    :param degree_l: Degree of harmonic coefficients
    :param order_m: Order of harmonic coefficient
    :param g_lm: Cosine factor of real harmonic coefficient
    :param h_lm: Sine factor of real harmonic coefficient
    :param normalisation: Normalisation function of harmonic functions
    :return: Normalised squared field spectrum (in units of $g_{\ell m}^2$).
    """
    degree_l, order_m, g_lm, h_lm = map(np.atleast_1d, (degree_l, order_m, g_lm, h_lm))

    assert(len(degree_l) == len(order_m))
    assert(len(degree_l) == len(g_lm))
    assert(len(degree_l) == len(h_lm))

    normalised_squared_field_spectrum = np.zeros(np.max(degree_l)+1)

    for line_id, degree in enumerate(degree_l):
        order = order_m[line_id]
        scaling = normalisation(degree, order)
        log.debug("%d: %d %d\t%f\t%f" % (line_id, degree, order, g_lm[line_id], h_lm[line_id]))
        normalised_squared_field_spectrum[degree] += scaling * (g_lm[line_id]**2 + h_lm[line_id]**2)

    return np.arange(0, np.max(degree_l)+1), normalised_squared_field_spectrum


def rlm(degree_l, r, rss):
    """

    :param degree_l: degree of spherical harmonic function
    :param r: r coordinate in units of r_star (distance from body centre)
    :param r_ss: radius of source surface in units of r_star
    :return: scaling factor for _radial_ values.
    """
    degree_l = np.array(degree_l)

    numerator   = degree_l + 1.0 + degree_l * (  r/rss) ** (2 * degree_l + 1)
    denominator = degree_l + 1.0 + degree_l * (1.0/rss) ** (2 * degree_l + 1)
    return (1.0 / r)**(degree_l + 2) * (numerator / denominator)


def fit_powerlaw(degree_l, spectrum):
    """very basic fit for now"""
    log_spectrum = np.log(spectrum)
    fit_data = stats.linregress(degree_l, log_spectrum)

    log.info("Slope %f, intercept %f, r value %f, p value %f, std_err %f" % fit_data)

    fitted_curve = np.exp(degree_l * fit_data[0] + fit_data[1])
    return fitted_curve, fit_data


def spectrum_plot(coefficient_file,
                  normalisation,
                  lowest_degree_l=1,
                  ax=None,
                  rvals=(1,)):

    source_surface_radius = 2
    if ax is None:
        fig, ax = plt.subplots()

    # TODO this code is dumb
    coeffs = convert_magnetogram.read_magnetogram_file(coefficient_file)
    degree_l = []
    order_m = []
    g_lm = []
    h_lm = []
    for degree in range(0, coeffs.degree_max() + 1):
        for order in range(0, degree + 1):
            _g, _h = coeffs.get(degree, order)
            degree_l.append(degree)
            order_m.append(order)
            g_lm.append(_g)
            h_lm.append(_h)
    # End of dumb code

    for radial_distance in rvals:
        g_lm_r = rlm(degree_l, radial_distance, source_surface_radius) * g_lm
        h_lm_r = rlm(degree_l, radial_distance, source_surface_radius) * h_lm
        spectrum_degree_l, scaled_squared_field_spectrum = energy_spectrum(degree_l, order_m, g_lm_r, h_lm_r,
                                                                           normalisation=normalisation)

        linear_fit, linear_fit_data = fit_powerlaw(
            spectrum_degree_l[lowest_degree_l:],
            scaled_squared_field_spectrum[lowest_degree_l:])

        points, = ax.plot(spectrum_degree_l, scaled_squared_field_spectrum, 'o', markersize=4,
                          label="File: %s.\nNormalisation: %s." % (coefficient_file,
                                                                   normalisation.__name__.split("_")[1]))

        ax.plot(spectrum_degree_l[lowest_degree_l:], linear_fit, 'k-', color=points.get_color(),
                label="$f(\ell)=%3.3g\ell + %3.3g$\n$r=%3.3g$" % linear_fit_data[0:3])

    ax.set_yscale('log')
    ax.grid(True, which='both')
    ax.legend()
    plt.xlabel("Degree $\ell$")
    plt.ylabel("Scaled squared field")
    return ax

