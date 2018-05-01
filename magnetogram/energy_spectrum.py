import numpy as np
import logging

from . import convert_magnetogram
import matplotlib.pyplot as plt
from scipy import stats

log = logging.getLogger(__name__)


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

    degree_l, order_m, g_lm, h_lm = map(np.array, (degree_l, order_m, g_lm, h_lm))

    assert(len(degree_l) == len(order_m))
    assert(len(degree_l) == len(g_lm))
    assert(len(degree_l) == len(h_lm))

    scaled_squared_field_spectrum = np.zeros(np.max(degree_l)+1)

    for line_id, degree in enumerate(degree_l):
        order = order_m[line_id]
        scaling = normalisation(degree, order)
        log.debug("%d: %d %d\t%f\t%f" % (line_id, degree, order, g_lm[line_id], h_lm[line_id]))
        scaled_squared_field_spectrum[degree] += scaling * (g_lm[line_id]**2 + h_lm[line_id]**2)

    return np.arange(0, np.max(degree_l)+1), scaled_squared_field_spectrum


def fit_powerlaw(degree_l, spectrum):
    """very basic fit for now"""
    log_spectrum = np.log(spectrum)
    fit_data = stats.linregress(degree_l, log_spectrum)

    log.info("Slope %f, intercept %f, r value %f, p value %f, std_err %f" % fit_data)

    fitted_curve = np.exp(degree_l * fit_data[0] + fit_data[1])
    return fitted_curve, fit_data


def spectrum_plot(coefficient_file,
                  normalisation,
                  lowest_degree_l = 1,
                  ax=None):

    data = convert_magnetogram.read_magnetogram_file(coefficient_file)

    spectrum_degree_l, scaled_squared_field_spectrum = energy_spectrum(*data, normalisation=normalisation)

    linear_fit, linear_fit_data = fit_powerlaw(
        spectrum_degree_l[lowest_degree_l:],
        scaled_squared_field_spectrum[lowest_degree_l:])

    if ax is None:
        fig, ax = plt.subplots()
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

