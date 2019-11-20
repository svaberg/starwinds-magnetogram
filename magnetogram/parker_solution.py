import numpy as np
import scipy.constants
from scipy.special import lambertw


class AnalyticBase:

    @staticmethod
    def parker_lhs(speed_dimensionless):
        """
        Left hand side of dimensionless Parker equation.
        :param speed_dimensionless:
        :return:
        """
        return speed_dimensionless ** 2 - 1 - 2 * np.log(speed_dimensionless)

    @staticmethod
    def parker_rhs(radius_dimensionless):
        """
        Right hand side of dimensionless Parker equation.
        :param radius_dimensionless:
        :return:
        """
        return 4 * np.log(radius_dimensionless) + 4 * (1 / radius_dimensionless - 1)

    @classmethod
    @np.vectorize
    def find_parker_analytic(cls, radius_dimensionless, all_sols=False):
        """
        Dimensionless Parker equation solved with Lambert W function.
        :param radius_dimensionless:
        :param all_sols:
        :return:
        """

        _parker_rhs = cls.parker_rhs(radius_dimensionless)

        w = np.nan

        branches = (0, -1)

        def _branch(_radius_dimensionless):
            if _radius_dimensionless < 1:
                return branches[0]
            else:
                return branches[1]

        if not all_sols:
            w = np.sqrt(-lambertw(-np.exp(-_parker_rhs - 1), _branch(radius_dimensionless)))
            return np.real_if_close(w)
        else:
            w = np.sqrt(-lambertw(-np.exp(-_parker_rhs - 1), branches))
            w = np.real_if_close(w)
            return w[0], w[1]


class ParkerSolution(AnalyticBase):

    def __init__(self,
                 temperature=1.5e6,
                 base_density=1.5e14 * scipy.constants.proton_mass,
                 stellar_radius=6.95510e8,
                 stellar_mass=1.989e30):
        """
        Create a Parker solution instance. The Parker solution is isothermal.
        :param temperature:
        :param base_density:
        :param stellar_radius:
        :param stellar_mass:
        """

        self.temperature = temperature
        self.base_density = base_density
        self.stellar_radius = stellar_radius
        self.stellar_mass = stellar_mass

        G = scipy.constants.gravitational_constant
        proton_mass = scipy.constants.proton_mass
        k_B = scipy.constants.Boltzmann

        # Sonic point characteristics.
        self.radius_sonic = G * self.stellar_mass * proton_mass / (4 * self.temperature * k_B)
        self.speed_sonic = np.sqrt(2 * k_B * self.temperature / proton_mass)

        # The density at the sonic point depends on the surface density.
        self.speed_surface = self.speed(self.stellar_radius)
        self.density_sonic = self.base_density * \
                             (self.radius_sonic / self.stellar_radius) ** -2 * \
                             (self.speed_sonic / self.speed_surface) ** -1

    def __str__(self):
        s0 = "Parker solution T_c=%2.2G K, Rho_c=%2.2G kg/m3, " % (self.temperature, self.base_density)
        s1 = "R=%2.2G, M=%2.2G," % (self.stellar_radius, self.stellar_mass)
        return s0 + s1

    def speed(self, radii):
        """

        :param radii:
        :return:
        """
        radii = np.atleast_1d(radii)
        radii[np.where(radii <= self.stellar_radius)] = self.stellar_radius

        speeds_dimensionless = self.find_parker_analytic(radii / self.radius_sonic)
        speeds = speeds_dimensionless * self.speed_sonic
        return speeds

    def density(self, radii):
        """

        :param radii:
        :return:
        """
        radii = np.atleast_1d(radii)
        radii[np.where(radii <= self.stellar_radius)] = self.stellar_radius

        speeds = self.speed(radii)
        densities = self.base_density * \
                    (radii / self.stellar_radius) ** -2 * \
                    (speeds / self.speed_surface) ** -1

        return densities

    @property
    def total_mass_flux(self):
        """
        Total mass flux through a closed surface enclosing the star. The mass flux is the same through any such
        surface.
        :return:
        """
        return 4 * np.pi * self.radius_sonic**2 * self.speed_sonic * self.density_sonic
