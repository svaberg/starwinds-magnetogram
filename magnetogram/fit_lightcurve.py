import numpy as np
from scipy import optimize
import scipy.special
from scipy import integrate
import matplotlib.pyplot as plt
import logging

from . import profile_shapes
from . import plot_fit_lightcurve

log = logging.getLogger(__name__)


def fit(x, y, profile, absorbtion=True, guess=None):

    def logme(names, values, errors=None):
        if errors is None:
            errors = [0.0]*len(values)

        return_value = ""
        for key, value, error in zip(names, values, errors):
            # print(key)
            # print(value)
            # print(error)
            return_value += ' %s=%f±%f,' % (key, value, error)

        return_value = return_value[:-1] + '.'
        return return_value

    if guess is None:
        param_names = ['center', 'depth']
        param_guesses = [np.mean(x), .1]
        import inspect
        curve_params = inspect.signature(profile).parameters
        param_names.extend(curve_params.keys())
        param_guesses.extend([1] * len(curve_params.keys()))

    log.debug('Initial guesses:' + logme(param_names, param_guesses))


    if type(profile) is

    def fitting_function(x, center, depth, *args, absorbtion=absorbtion):
        log.debug(['%s=%f' %(a,b) for (a,b) in zip(param_names, (center, depth, *args))])
        fitted_values = depth * profile(*args)(x - center)
        if absorbtion:
            fitted_values = 1 - fitted_values

        return fitted_values

    popt, pcov = scipy.optimize.curve_fit(fitting_function, x, y, p0=param_guesses)


    center = popt[0]
    depth = popt[1]
    fitted_profile = profile(*popt[2:])
    fitted_data = depth * fitted_profile(x - center)
    if absorbtion:
        fitted_data = 1 - fitted_data

    errors = y - fitted_data

    log.info('Final fit of %s: %s' % (fitted_profile, logme(param_names, popt, np.sqrt(np.diag(pcov)))))
    log.info('%s height=%f, area=%f, equivalent width=%f' % (fitted_profile,
                                                             fitted_profile.strength,
                                                             fitted_profile.area,
                                                             fitted_profile.equivalent_width))

    if isinstance(fitted_profile, profile_shapes.Voigt):
        # wl0    str     GaussWidth(km/s)  LorantzWidth(1/gausswidth)  lande_g  limb_darkening
        log.info(
            'Folsom parameters (guesses) wl=?, str=%f widthLorent(1/widthGauss)=%f' %
            (0, fitted_profile.gamma/fitted_profile.sigma))
    return center, depth, fitted_profile, errors


def fit_curve(curve_file, profile=profile_shapes.Gaussian, skip_header=2, absorbtion=True):
    data = np.genfromtxt(curve_file, skip_header=skip_header)

    data_x = data[:, 0]
    data_y = data[:, 1]
    data_y_std = data[:, 2]

    fitted_center, fitted_depth, fitted_profile, errors = fit(data_x, data_y, profile, absorbtion)

    fitted_x = np.linspace(np.min(data_x), np.max(data_x), 200)
    fitted_y = fitted_depth * fitted_profile(fitted_x - fitted_center)
    if absorbtion:
        fitted_y = 1 - fitted_y

    plot_fit_lightcurve.plot_fit_graphics(data_x, data_y, data_y_std, fitted_x, fitted_y, errors, fit_name=fitted_profile, true_x=None, true_y=None, true_name='True')

    plt.show()
