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


def demo_plot(light_curve, ax=None, xrange=(-10,10)):
    if ax is None:
        ax=plt.gca()

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
        param_names = ['center', 'depth']
        param_guesses = [np.mean(x), .1]
        import inspect
        curve_params = inspect.signature(profile).parameters
        param_names.extend(curve_params.keys())
        param_guesses.extend([1] * len(curve_params.keys()))

    print('Initial guesses:' + logme(param_names, param_guesses))

    def fitting_function(x, center, depth, *args, absorbtion=absorbtion):
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

    print('Final fit of %s: %s' % (fitted_profile, logme(param_names, popt, np.sqrt(np.diag(pcov)))))
    print('%s height=%f, equivalent width=%f' % (fitted_profile, fitted_profile.height, fitted_profile.equivalent_width))

    return center, depth, fitted_profile, errors


def fit_test(true_profile, center=0, depth=1, std=0.1, num=50, absorbtion=False, profile=Gaussian, xrange=None):
    if xrange is None:
        xrange = np.array([-1.0, 1.0])
        xrange *= 3.0 * true_profile.equivalent_width
        xrange += center

    def evaluate(profile, center, depth, numel=200, std=0.0):
        x = np.linspace(*xrange, numel)
        y = depth * profile(x - center)
        if absorbtion:
            y = 1 - y
        y += np.random.normal(size=x.shape, scale=std)

        return x, y

    true_x, true_y = evaluate(true_profile, center, depth, 200)
    data_x, data_y = evaluate(true_profile, center, depth, num, std)

    try:
        fitted_center, fitted_depth, fitted_profile, errors = fit(data_x, data_y, profile, absorbtion)
        fitted_x, fitted_y = evaluate(fitted_profile, fitted_center, depth, 200, 0.0)
        # raise RuntimeError('a test')
    except RuntimeError as e:
        # log that fit failed
        fitted_profile = errors = None
        fitted_x = fitted_y = None

    plot_fit_graphics(data_x, data_y, std, fitted_x, fitted_y, errors, fitted_profile, true_x, true_y, true_profile)
    plt.show()


def plot_fit_graphics(data_x, data_y, data_y_std, fitted_x, fitted_y, errors, fit_name='Fit',
                      true_x=None, true_y=None, true_name='True'):

    fig, (ax1, ax2) = plt.subplots(2,1)
    print(fig)
    ax3 = ax2.twiny()
    fig.subplots_adjust(hspace=0)

    ax1.errorbar(data_x, data_y, yerr=data_y_std, label='Noisy data',
                 fmt='o',
                 markersize=1,
                 # color=line.get_color(),
                 elinewidth=0.5)

    if fitted_x is not None:
        ax1.plot(fitted_x, fitted_y, label='Fitted %s' % fit_name)

    if true_x is not None:
        ax1.plot(true_x, true_y, 'k', label='True %s' % true_name)

    ax1.grid(True)
    ax1.set_ylabel('Signal')
    ax1.set_xlabel('Velocity')
    ax1.xaxis.set_ticks_position("top")
    ax1.xaxis.set_label_position("top")
    ax1.yaxis.set_label_position("right")
    ax1.set_zorder(1000)
    ax1.autoscale(enable=True, axis='x', tight=True)
    # ax1.spines['bottom'].set_visible(False)
    ax1.patch.set_alpha(0.0)
    ax1.legend()

    # range = np.max(np.abs(errors))
    # bins = np.linspace(-range, range, len(errors)//50)
    # bins = np.round(50 * bins) / 50
    # n, bins,_ = ax2.hist(errors, bins=bins, orientation='horizontal')
    # centers = 0.5 * (bins[1:] + bins[:-1])
    # centers = bins
    # ax2.set_yticks(centers)

    if errors is not None:
        ax2.hist(errors, orientation='horizontal')

    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.yaxis.set_ticks_position("right")
    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("Residual")
    #ax2.grid(True, axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.tick_params(axis='x', top=False)

    ax3.plot(data_x, np.zeros_like(data_x), 'k-')

    if errors is not None:
        ax3.plot(data_x, errors, 'ko', label='Residuals', markersize=1.0)

    # ax3.xaxis.set_ticks_position("top")
    # ax3.xaxis.set_label_position("top")
    # ax3.axes.get_xaxis().set_visible(False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.grid(True, axis='x')
    ax3.autoscale(enable=True, axis='x', tight=True)
    ax3.spines['top'].set_visible(False)
    ax3.tick_params(axis='x', top=False)

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

def fitting_test():
    fit_test(Gaussian(1), center=1, absorbtion=False, profile=Gaussian)
    fit_test(Gaussian(1), center=1, absorbtion=False, profile=Gaussian)
    fit_test(Gaussian(1), center=1, absorbtion=False, profile=Lorentzian)
    fit_test(Gaussian(1), center=1, absorbtion=False, profile=Lorentzian)
    fit_test(Gaussian(1), center=1, absorbtion=False, profile=FaddeevaVoigt)
    fit_test(Gaussian(1), center=1, absorbtion=False, profile=FaddeevaVoigt)
    fit_test(Gaussian(1), center=1, absorbtion=False, profile=PseudoVoigt)

    fit_test(Gaussian(1), center=10, absorbtion=True, num=500, profile=Gaussian)
    fit_test(Gaussian(1), center=10, absorbtion=True, profile=Gaussian)
    fit_test(Gaussian(1), center=10, absorbtion=True, profile=Lorentzian)
    fit_test(Gaussian(1), center=10, absorbtion=True, profile=Lorentzian)
    fit_test(Gaussian(1), center=-10, absorbtion=True, profile=FaddeevaVoigt)
    fit_test(Gaussian(1), center=0, absorbtion=True, profile=FaddeevaVoigt)
    fit_test(Gaussian(1), center=13, absorbtion=True, profile=PseudoVoigt)

def all_the_combinations():
    pass


def fit_an_actual_curve(curve_file='/Users/u1092841/Documents/PHD/toupies-pipeline/lopeg_31aug14_v_09.s.out.lsd',
                        profile=Gaussian, skip_header=2, absorbtion=True):
    data = np.genfromtxt(curve_file, skip_header=skip_header)

    data_x = data[:, 0]
    data_y = data[:, 1]
    data_y_std = data[:, 2]

    fitted_center, fitted_depth, fitted_profile, errors = fit(data_x, data_y, profile, absorbtion)

    fitted_x = np.linspace(np.min(data_x), np.max(data_x), 200)
    fitted_y = fitted_depth * fitted_profile(fitted_x - fitted_center)
    if absorbtion:
        fitted_y = 1 - fitted_y

    plot_fit_graphics(data_x, data_y, data_y_std, fitted_x, fitted_y, errors,
                      fit_name=fitted_profile, true_x=None, true_y=None, true_name='True')


if __name__ == "__main__":
    # basic_test()
    # fitting_test()
    fit_an_actual_curve()
    fit_an_actual_curve(profile=FaddeevaVoigt)

