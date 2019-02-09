import numpy as np
import scipy as sp
import scipy.special

import logging
log = logging.getLogger(__name__)


class LehmannZdi:
    """
    Reference implementation based on equation 5 in Lehmann et al. (2018).
    Produces same plots as in Folsom (2016, 2018) except that the sign of
    the polar and azimuthal components is always (?) reversed.
    """
    def __init__(self, degrees_l, orders_m, alpha_lm, beta_lm, gamma_lm):
        self.degrees_l = np.atleast_1d(degrees_l)
        self.orders_m = np.atleast_1d(orders_m)
        self.alpha = np.atleast_1d(alpha_lm)
        self.beta = np.atleast_1d(beta_lm)
        self.gamma = np.atleast_1d(gamma_lm)

        # Calculate c_lm as described in article.
        c_lm2 = (
                (2 * self.degrees_l + 1)
                / (4 * np.pi)
                * sp.special.factorial(self.degrees_l - self.orders_m)
                / sp.special.factorial(self.degrees_l + self.orders_m)
        )

        self.c_lm = np.sqrt(c_lm2)

    def get_radial_poloidal_field(self, points_polar, points_azimuth):
        """
        Get the radial component of the poloidal field $B_{r, pol}$.
        :param points_polar:
        :param points_azimuth:
        :return: Radial field
        """
        field_radial_poloidal = np.zeros_like(points_azimuth, dtype=complex)

        for deg_l, ord_m, a_lm, c_lm in zip(self.degrees_l, self.orders_m, self.alpha, self.c_lm):
            p_lm = sp.special.lpmv(ord_m, deg_l, np.cos(points_polar))
            field_radial_poloidal += (a_lm
                                      * c_lm
                                      * p_lm
                                      * np.exp(1.0j * ord_m * points_azimuth)
                                      )

        return np.real(field_radial_poloidal)

    def get_radial_toroidal_field(self, points_polar, points_azimuth):
        """
        Get the radial component of the toroidal field $B_{r, pol}$.
        This is always zero.
        :param points_polar:
        :param points_azimuth:
        :return: Radial field
        """
        field_radial_toroidal = np.zeros_like(points_azimuth, dtype=complex)

        return np.real(field_radial_toroidal)

    def get_radial_field(self, points_polar, points_azimuth):
        """
        Get the total radial component of the field $B_{r}$.
        :param points_polar:
        :param points_azimuth:
        :return: polar field component
        """
        return (
                self.get_radial_poloidal_field(points_polar, points_azimuth)
                + self.get_radial_toroidal_field(points_polar, points_azimuth)
                )

    def get_polar_poloidal_field(self, points_polar, points_azimuth):
        """
        Get the polar component of the poloidal field $B_{\phi, pol}$.
        :param points_polar:
        :param points_azimuth:
        :return: polar component of the poloidal field
        """
        field_polar_poloidal = np.zeros_like(points_azimuth, dtype=complex)

        for deg_l, ord_m, b_lm, c_lm in zip(self.degrees_l, self.orders_m, self.beta, self.c_lm):
            p_lm = sp.special.lpmv(ord_m, deg_l, np.cos(points_polar))
            field_polar_poloidal -= (b_lm
                                     * c_lm
                                     * p_lm
                                     * 1.0j * ord_m * np.exp(1.0j * ord_m * points_azimuth)
                                     / (deg_l + 1)
                                     / np.sin(points_polar)
                                     )

        return np.real(field_polar_poloidal)

    def get_polar_toroidal_field(self, points_polar, points_azimuth):
        """
        Get the polar component of the toroidal field $B_{\phi, tor}$.
        :param points_polar:
        :param points_azimuth:
        :return: polar component of the toroidal field
        """
        field_polar_toroidal = np.zeros_like(points_azimuth, dtype=complex)

        for deg_l, ord_m, g_lm, c_lm in zip(self.degrees_l, self.orders_m, self.gamma, self.c_lm):
            p_lm = sp.special.lpmv(ord_m, deg_l, np.cos(points_polar))
            DPml = (p_lm - np.roll(p_lm, 1)) / (np.cos(points_polar) - np.roll(np.cos(points_polar), 1)) * np.sin(
                points_polar)

            field_polar_toroidal += (g_lm
                                     * c_lm
                                     / (deg_l + 1)
                                     * DPml
                                     * np.exp(1.0j * ord_m * points_azimuth)
                                     )

        return np.real(field_polar_toroidal)

    def get_polar_field(self, points_polar, points_azimuth):
        """
        Get the total polar component of the field $B_{\phi}$.
        :param points_polar:
        :param points_azimuth:
        :return: polar field component
        """
        return (
                self.get_polar_poloidal_field(points_polar, points_azimuth)
                + self.get_polar_toroidal_field(points_polar, points_azimuth)
                )

    def get_azimuthal_poloidal_field(self, points_polar, points_azimuth):
        """
        Get the azimuthal component of the poloidal field $B_{\theta, pol}$.
        :param points_polar:
        :param points_azimuth:
        :return: azimuthal component of the poloidal field
        """
        field_azimuthal_poloidal = np.zeros_like(points_azimuth, dtype=complex)

        for deg_l, ord_m, b_lm, c_lm in zip(self.degrees_l, self.orders_m, self.beta, self.c_lm):
            p_lm = sp.special.lpmv(ord_m, deg_l, np.cos(points_polar))
            DPml = (p_lm - np.roll(p_lm, 1)) / (np.cos(points_polar) - np.roll(np.cos(points_polar), 1)) * np.sin(
                points_polar)

            field_azimuthal_poloidal += (b_lm
                                         * c_lm
                                         / (deg_l + 1)
                                         * DPml
                                         * np.exp(1.0j * ord_m * points_azimuth)
                                         )

        return np.real(field_azimuthal_poloidal)

    def get_azimuthal_toroidal_field(self, points_polar, points_azimuth):
        """
        Get the azimuthal component of the toroidal field $B_{\theta, tor}$.
        :param points_polar:
        :param points_azimuth:
        :return: azimuthal component of the toroidal field
        """
        field_azimuthal_toroidal = np.zeros_like(points_azimuth, dtype=complex)

        for deg_l, ord_m, g_lm, c_lm in zip(self.degrees_l, self.orders_m, self.gamma, self.c_lm):
            p_lm = sp.special.lpmv(ord_m, deg_l, np.cos(points_polar))
            field_azimuthal_toroidal += (g_lm
                                         * c_lm
                                         * p_lm
                                         * 1.0j * ord_m * np.exp(1.0j * ord_m * points_azimuth)
                                         / (deg_l + 1)
                                         / np.sin(points_polar)
                                         )

        return np.real(field_azimuthal_toroidal)

    def get_azimuthal_field(self, points_polar, points_azimuth):
        """
        Get the total azimuthal component of the field $B_{\theta}$.
        :param points_polar:
        :param points_azimuth:
        :return: azimuthal field component
        """
        return (
                self.get_azimuthal_poloidal_field(points_polar, points_azimuth)
                + self.get_azimuthal_toroidal_field(points_polar, points_azimuth)
                )

    # This is not part of Lehmann but from zdipy.
    # When should the real values be taken?
    def get_field_strength(self, points_polar, points_azimuth):
        """
        Get field strength $B$.
        :param points_polar:
        :param points_azimuth:
        :return: field strength
        """
        field_radial = self.get_radial_field(points_polar, points_azimuth)
        field_polar = self.get_polar_field(points_polar, points_azimuth)
        field_azimuthal = self.get_azimuthal_field(points_polar, points_azimuth)

        # Take real values?
        return (field_radial**2 + field_polar**2 + field_azimuthal**2)**.5
