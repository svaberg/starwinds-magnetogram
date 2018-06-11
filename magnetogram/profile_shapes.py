import numpy as np
import scipy
#from scipy import optimize
#import scipy.special
from scipy import integrate

import logging
log = logging.getLogger(__name__)


class ProfileShape(object):
    """
    A profile shape suitable for use as a light curve such as a Gaussian profile or a Voigt profile. The profile
    shape is assumed to have its extremal value at zero.
    """

    @property
    def strength(self) -> float:
        """
        Return height of curve
        :return: Highest value of light curve, assumed to be at the centre of the curve.
        """
        return self(0)

    @property
    def equivalent_width(self) -> float:
        """
        Return the equivalent width of the light curve, i.e. the width of a unit height rectangle whose area corresponds
        to the area under the light curve. TODO is this right? Or is it equal to height * width?
        :return: light curve equivalent width
        """
        equivalent_width = self.area / self.strength
        return equivalent_width

    @property
    def area(self) -> float:
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


class Gaussian(ProfileShape):
    """
    Gaussian profile, normalised and centered at zero.
    """
    def __init__(self, sigma):
        super().__init__()
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


class Lorentzian(ProfileShape):
    """
    Lorentzian profile, normalised and centered at zero.
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


class Voigt(ProfileShape):
    """
    A Voigt type light curve
    """
    pass


class PseudoVoigt(Voigt):
    """
    Pseudo-Voigt profile from
    https://en.wikipedia.org/wiki/Voigt_profile#Pseudo-Voigt_approximation
    not normalised
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
    Voigt profile using the Faddeeva function. Normalised and centered at zero
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
    def strength(self):
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
