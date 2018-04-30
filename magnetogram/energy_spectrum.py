import numpy as np
import logging


log = logging.getLogger(__name__)


def schmidt_normalisation(degree_l, order_m):
    r"""
    Schmidt normalisation $\frac{1}{2\ell +1}$
    :param degree_l: Degree $\ell$ of harmonic coefficient
    :param order_m: Order $m$ of harmonic coefficient
    :return: Schmidt scaling value for degree_l
    """
    return (2.0 * degree_l + 1.0) ** -1.0

def zdipy_normalisation(degree_l, order_m):
    r"""
    As we understand zdipy does not use Schmidt scaling but
    :param degree_l:
    :param order_m:
    :return:
    """
    return (4 * np.pi) * np.where(order_m == 0, 1, 2)

def energy_spectrum(degree_l, order_m, g_lm, h_lm, normalisation=zdipy_normalisation):

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
    from scipy import stats
    log_spectrum = np.log(spectrum)
    fit_data = stats.linregress(degree_l, log_spectrum)
    log.debug("Slope %f, intercept %f, r value %f, p value %f, std_err %f" % fit_data)

    fitted_curve = np.exp(degree_l * fit_data[0] + fit_data[1])
    return fitted_curve, fit_data

def setup_test():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    root_logger.addHandler(ch)


def test_1():
    import matplotlib.pyplot as plt
    import convert_magnetogram
    degree_l, order_m, g_lm, h_lm = convert_magnetogram.read_magnetogram_file("outMagCoeff.dat")
    degree_l, scaled_squared_field_spectrum = energy_spectrum(degree_l, order_m, g_lm, h_lm)
    for line_id, value in enumerate(scaled_squared_field_spectrum):
        log.debug("%d\t%f" % (line_id, value))

    plt.plot(degree_l, scaled_squared_field_spectrum, 'o')
    plt.xlabel("Degree $\ell$")
    plt.ylabel("Scaled squared field")

    linear_fit, linear_fit_data = fit_powerlaw(degree_l[1:], scaled_squared_field_spectrum[1:])
    plt.plot(degree_l[1:], linear_fit, 'k-')
    plt.yscale('log')
    plt.grid(True, which='both')
    plt.show()


if __name__ == "__main__":
    setup_test()
    test_1()

