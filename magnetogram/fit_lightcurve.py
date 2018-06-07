import numpy as np
from scipy import optimize
import scipy.special
from scipy import integrate



class EmissionCurve(object):
    """
    A light emission curve centered at center
    """
    def __init__(self, center, shape):
        self._center = center
        self._shape = shape

    def __call__(self, x):
        x_centered = x - self.center
        return self._shape(x_centered)

    @property
    def center(self):
        """
        Return center of light curve.
        :return: Center of light curve.
        """
        return self._center

    @property
    def equivalent_width(self):
        print(self._shape)
        return self._shape.equivalent_width


class Curve(object):
    """
    A curve suitable for use as  a light curve
    """
    pass
    @property
    def height(self):
        """
        Return height of curve
        :return: Highest value of light curve, assumed to be at the centre of the curve.
        """
        return self(0)

    @property
    def equivalent_width(self):
        """
        Return the equivalent width of the light curve, i.e. the width of a unit height rectangle whose area corresponds
        to the area under the light curve. TODO is this right? Or is it equal to height * width?
        :return: light curve equivalent width
        """
        equivalent_width = self.area / self.height
        return equivalent_width

    @property
    def area(self):
        """
        Return area under light curve integrated over the whole number line.
        :return: area under light curve
        """
        area, _ = scipy.integrate.quad(self, -np.inf, np.inf)
        return area

    # @property
    # def fwhm(self, x0=1.5):
    #     """
    #     Return the full width half maximum of the light curve, i.e. the distance along the x axis from the light
    #     curve's maximum to the point where it's value is half of the maximum value.
    #     :param x0: initial guess
    #     :return: full width half maximum of light curve
    #     """
    #     height = self.height
    #     root = optimize.newton(lambda x: self(x) - height / 2, x0)
    #     return root * 2

class Gaussian(Curve):
    """
    Gaussian profile
    """
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        num = np.exp(-x ** 2 / (2 * self.sigma ** 2))
        den = self.sigma * np.sqrt(2 * np.pi)
        return num / den

    def __str__(self):
        return "Gaussian"

    @property
    def fwhm(self):
        return 2 * np.sqrt(2 * np.log(2)) * self.sigma


class Lorentzian(Curve):
    """
    Lorentzian profile
    """
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, x):
        den = np.pi * (x ** 2 + self.gamma ** 2)
        return self.gamma / den

    def __str__(self):
        return "Lorentzian"

    @property
    def fwhm(self):
        return 2 * self.gamma


class Voigt(Curve):
    """
    A Voigt type light curve
    """
    pass


class PseudoVoigt(Voigt):
    """
    Pseudo-Voigt profile from
    https://en.wikipedia.org/wiki/Voigt_profile#Pseudo-Voigt_approximation
    """
    def __init__(self, sigma, gamma):
        self.gaussian = Gaussian(sigma)
        self.lorentzian = Lorentzian(gamma)
        self.eta = self.calculate_eta()

    def calculate_eta(self):
        f_L = self.lorentzian.fwhm # fwhm lorentzian
        f_G = self.gaussian.fwhm  # fwhm gaussian

        f5 = f_G ** 5
        f5 += 2.69269 * f_G ** 4 * f_L ** 1
        f5 += 2.42843 * f_G ** 3 * f_L ** 2
        f5 += 4.47163 * f_G ** 2 * f_L ** 3
        f5 += 0.07842 * f_G ** 1 * f_L ** 4
        f5 += f_L ** 5
        f = f5 ** (1 / 5)

        fLf = f_L / f
        return 1.36603 * fLf - 0.47719 * fLf ** 2 + 0.11116 * fLf ** 3

    def __call__(self, x):
        values = 0
        values += self.eta * self.lorentzian(x)
        values += (1 - self.eta) * self.gaussian(x)
        return values

    def __str__(self):
        return "Pseudo-Voigt"


class FaddeevaVoigt(Voigt):
    """
    Voigt profile using the Faddeeva function
    """
    def __init__(self, sigma, gamma):
        self.sigma = sigma
        self.gamma = gamma

    def __call__(self, x):
        z = (x + 1j * self.gamma) / (self.sigma * np.sqrt(2))
        wofz_real = np.real(scipy.special.wofz(z))
        return wofz_real / (self.sigma * np.sqrt(2 * np.pi))

    def __str__(self):
        return "Faddeeva-Voigt"


class ConvolutionVoigt(Voigt):
    """
    Voigt profile with explicit convolution
    """
    def __init__(self, sigma, gamma):
        self.gaussian = Gaussian(sigma)
        self.lorentzian = Lorentzian(gamma)

    def __call__(self, x):
        values = np.convolve(
            self.gaussian(x),
            self.lorentzian(x),
            'same')
        delta_x = x[1] - x[0]
        return values * delta_x

    def __str__(self):
        return "Convolution-Voigt"

    @property
    def height(self):
        raise NotImplementedError('Oops.')

    @property
    def area(self):
        raise NotImplementedError('Oops.')

    @property
    def fwhm(self, x0=1.5):
        raise NotImplementedError('Oops.')

    @property
    def equivalent_width(self):
        raise NotImplementedError('Oops.')


import matplotlib.pyplot as plt


def demo_plot(light_curve, ax=plt.gca(), xrange=(-10,10)):
    x = np.linspace(*xrange, 1000)
    line, = ax.plot(x, light_curve(x), label=light_curve)

    # Plot equivalent width rectangle
    try:
        half_ew = light_curve.equivalent_width / 2
        height = light_curve.height
        print("%s equivalent width: %f, height: %f." % (light_curve, 2 * half_ew, height))

        plt.plot([xrange[0], -half_ew, -half_ew, half_ew, half_ew, xrange[1]],
                 [0, 0, height, height, 0, 0],
                 linestyle=':',
                 color=line.get_color())
    except NotImplementedError as e:
        print("Equivalent width not supported by %s: %s" % (light_curve, e))


def fit(x, y, profile, absorbtion=True, guess=None):

    def logme(names, values, errors=None):
        if errors is None:
            errors = [0.0]*len(values)

        return_value = ""
        for key, value, error in zip(names, values, errors):
            # print(key)
            # print(value)
            # print(error)
            return_value += ' %s=%f+-%f,' % (key, value, error)

        return_value = return_value[:-1] + '.'
        return return_value

    if guess is None:
        param_names = ['center']
        param_guesses = [np.mean(x)]
        import inspect
        curve_params = inspect.signature(profile).parameters
        param_names.extend(curve_params.keys())
        param_guesses.extend([1] * len(curve_params.keys()))


    print('Initial guesses:' + logme(param_names, param_guesses))

    def fitting_function(x, center, *args, absorbtion=absorbtion):
        fitted_values = profile(*args)(x - center)
        if absorbtion:
            fitted_values = 1 - fitted_values

        return fitted_values


    popt, pcov = scipy.optimize.curve_fit(fitting_function, x, y, p0=param_guesses)

    center = popt[0]
    fitted_values = profile(*popt[1:])
    print('Final fit of %s: %s' % (fitted_values, logme(param_names, popt, np.sqrt(np.diag(pcov)))))
    print('%s height=%f, equivalent width=%f' % (fitted_values, fitted_values.height, fitted_values.equivalent_width))

    return center, fitted_values


def fit_test(true_profile, center=0, absorbtion=False, profile=Gaussian, xrange=None):
    if xrange is None:
        xrange = np.array([-1.0, 1.0])
        xrange *= 3.0 * true_profile.equivalent_width
        xrange += center
    x = np.linspace(*xrange, 200)
    true_data = true_profile(x - center)
    if absorbtion:
        true_data = 1 - true_data
    noise = np.random.normal(size=x.shape, scale=0.1)

    true_line, = plt.plot(x, true_data, label='True data')
    plt.plot(x, true_data + noise, 'x', color=true_line.get_color(), label='Noisy data')

    fitted_center, light_curve = fit(x, true_data + noise, profile, absorbtion)
    fitted_values = light_curve(x - fitted_center)
    if absorbtion:
        fitted_values = 1 - fitted_values
    plt.plot(x, fitted_values, '--', label=light_curve)

    plt.grid(True)
    plt.legend()
    plt.show()


def basic_test():

    demo_plot(Gaussian(1))
    demo_plot(Lorentzian(1))
    demo_plot(PseudoVoigt(1, 1))
    demo_plot(FaddeevaVoigt(1, 1))
    demo_plot(ConvolutionVoigt(1, 1))
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # basic_test()
    fit_test(Gaussian(1), center=1, absorbtion=False, profile=Gaussian)
    fit_test(Gaussian(1), center=1, absorbtion=False, profile=Gaussian)
    fit_test(Gaussian(1), center=1, absorbtion=False, profile=Lorentzian)
    fit_test(Gaussian(1), center=1, absorbtion=False, profile=Lorentzian)
    fit_test(Gaussian(1), center=1, absorbtion=False, profile=FaddeevaVoigt)
    fit_test(Gaussian(1), center=1, absorbtion=False, profile=FaddeevaVoigt)
    fit_test(Gaussian(1), center=1, absorbtion=False, profile=PseudoVoigt)

    fit_test(Gaussian(1), center=10, absorbtion=True, profile=Gaussian)
    fit_test(Gaussian(1), center=10, absorbtion=True, profile=Gaussian)
    fit_test(Gaussian(1), center=10, absorbtion=True, profile=Lorentzian)
    fit_test(Gaussian(1), center=10, absorbtion=True, profile=Lorentzian)
    fit_test(Gaussian(1), center=-10, absorbtion=True, profile=FaddeevaVoigt)
    fit_test(Gaussian(1), center=0, absorbtion=True, profile=FaddeevaVoigt)
    fit_test(Gaussian(1), center=13, absorbtion=True, profile=PseudoVoigt)
