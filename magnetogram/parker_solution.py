import scipy.constants
from scipy.special import lambertw
import numpy as np


def parker_lhs(speed_dimensionless):
    """
    Left hand side of dimensionless Parker equation.
    :param speed_dimensionless:
    :return:
    """
    return speed_dimensionless ** 2 - 1 - 2 * np.log(speed_dimensionless)


def parker_rhs(radius_dimensionless):
    """
    Right hand side of dimensionless Parker equation.
    :param radius_dimensionless:
    :return:
    """
    return 4 * np.log(radius_dimensionless) + 4 * (1 / radius_dimensionless - 1)


@np.vectorize
def find_parker_analytic(radius_dimensionless, all_sols=False):
    """
    Dimensionless Parker equation solved with Lambert W function.
    :param radius_dimensionless:
    :param all_sols:
    :return:
    """

    _parker_rhs = parker_rhs(radius_dimensionless)

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


def parker_speed(radii,
                 temperature,
                 stellar_mass,
                 ):
    """
    Calculate speeds in Parker solution
    :param radii:
    :param temperature: Plasma temperature in Kelvin
    :param stellar_mass: Stellar mass in kg
    :return:
    """

    G = scipy.constants.gravitational_constant  # Nm2/kg2 (gravitational constant)
    proton_mass = scipy.constants.proton_mass  # 1.67e-27 kg (proton mass)
    k_B = scipy.constants.Boltzmann  # J/K (Boltzmann's constant)

    radius_sonic = G * stellar_mass * proton_mass / (4 * temperature * k_B)
    speed_sonic = np.sqrt(2 * k_B * temperature / proton_mass)

    speeds_dimensionless = find_parker_analytic(radii / radius_sonic)

    speeds = speeds_dimensionless * speed_sonic

    return speeds, radius_sonic, speed_sonic


def parker_solution(radii,
                    coronal_temperature=1.5e6,
                    coronal_base_density=1.5e14*scipy.constants.proton_mass,
                    stellar_radius=6.95510e8,
                    stellar_mass=1.989e30):
    """
    Caclulate full Parker solution. The Parker solution is isothermal.
    :param radii:
    :param coronal_temperature: Plasma temperature in Kelvin
    :param coronal_base_density: Coronal base density in kg/m3
    :param stellar_radius: Stellar radius in meters
    :param stellar_mass: Stellar mass in kg
    :return: velocity profile, density profile, sonic radius, speed and density.
    """

    radii[np.where(radii <= stellar_radius)] = stellar_radius

    speeds, radius_sonic, speed_sonic = parker_speed(radii, coronal_temperature, stellar_mass)
    speed_surface, _, _ = parker_speed(stellar_radius, coronal_temperature, stellar_mass)

    densities = coronal_base_density * (radii / stellar_radius) ** -2 * (speeds / speed_surface) ** -1
    density_sonic = coronal_base_density * (radius_sonic / stellar_radius) ** -2 * (speed_sonic / speed_surface) ** -1

    return speeds, densities, radius_sonic, speed_sonic, density_sonic
