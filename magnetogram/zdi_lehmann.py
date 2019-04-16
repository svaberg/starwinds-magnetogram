import numpy as np
import scipy as sp
import scipy.special

import logging
log = logging.getLogger(__name__)

class LehmannZdi:
    """
    Reference implementation based on equation 5 in Lehmann et al. (2018).
    Produces same plots as Folsom (2016, 2018) except that the sign of
    the polar and azimuthal components is always (?) reversed.
    """
    def __init__(self,
                 degrees_l, orders_m,
                 alpha_lm,
                 beta_lm=None,
                 gamma_lm=None,
                 dpml_method="gradient",  # For testing
                 ):
        """
        Construct from coefficients.
        :param degrees_l: Array of degree indices
        :param orders_m:  Array of order indices
        :param alpha_lm: Array of complex $\alpha$ coefficients
        :param beta_lm:  Array of complex $\beta$ coefficients
        :param gamma_lm: Array of complex $\gamma$ coefficients
        """
        if beta_lm is None:
            beta_lm = 0 * alpha_lm
        if gamma_lm is None:
            gamma_lm = 0 * alpha_lm

        # One dimension or more please.
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

        self._dpml_method = dpml_method

    def degree(self):
        """
        Degree of spherical harmonics
        :return:
        """
        return np.max(self.degrees_l)

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
            DPml = self._dpml(p_lm, points_polar)

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
            DPml = self._dpml(p_lm, points_polar)

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

    def get_all(self):
        """
        Get all the things in a dictionary.
        :param points_polar:
        :param points_azimuth:
        :return:
        """
        _dict = {}
        _dict["radial"] = {"poloidal": self.get_radial_poloidal_field,
                           "toroidal": self.get_radial_toroidal_field}  # This component is always zero, see definition.
        _dict["polar"] = {"poloidal": self.get_polar_poloidal_field,
                          "toroidal": self.get_polar_toroidal_field}
        _dict["azimuthal"] = {"poloidal": self.get_azimuthal_poloidal_field,
                              "toroidal": self.get_azimuthal_toroidal_field}

        return _dict

    def as_dipole(self):

        return self.as_restricted(degree_l_range=(0, 1))

    def as_restricted(self, degree_l_range=(0, 1000), order_m_range=(0, 1000)):

        degree_l_range = tuple(np.atleast_1d(degree_l_range))
        order_m_range = tuple(np.atleast_1d(order_m_range))

        if len(degree_l_range) == 1:
            degree_l_range = degree_l_range*2

        if len(order_m_range) == 1:
            order_m_range = order_m_range*2

        good_indices = np.asarray((degree_l_range[0] <= self.degrees_l) &
                                (degree_l_range[1] >= self.degrees_l) &
                                (order_m_range[0] <= self.orders_m) &
                                (order_m_range[1] >= self.orders_m)).nonzero()[0]  # Returns a tuple; take element 0.

        new_args = [arg[good_indices] for arg in (self.degrees_l,
                                                  self.orders_m,
                                                  self.alpha,
                                                  self.beta,
                                                  self.gamma)]

        log.info("Retaining %d coefficients" % len(good_indices))

        return LehmannZdi(*new_args, self._dpml_method)

    def energy(self):
        """
        Assume units are Gauss
        :return:
        """

        # First calculate energy associated with each coefficient
        energy_alpha = self._energy_helper(self.alpha)
        energy_beta = self._energy_helper(self.beta)
        energy_gamma = self._energy_helper(self.gamma)

        total_energy = np.sum(energy_alpha) + np.sum(energy_beta) + np.sum(energy_gamma)
        total_energy_poloidal = np.sum(energy_alpha) + np.sum(energy_beta)
        total_energy_toroidal = np.sum(energy_gamma)
        log.info('Total energy: %g (B^2)' % total_energy)

        log.info("radial  %g (%% tot)" % (np.sum(energy_alpha) / total_energy))
        log.info('poloidal %g (%% tot)' % (total_energy_poloidal / total_energy))
        log.info('toroidal %g (%% tot)' % (total_energy_toroidal / total_energy))

        log.info ('Fraction of magnetic energy in each component')
        log.info ('This can be summed, and should sum to 1.')
        log.info ('l   m    E(alpha)   E(beta)    E(gamma)')
        for i in range(len(self.alpha)):
            log.info('%2i %2i %10.5f %10.5f %10.5f' % (self.degrees_l[i],
                                                       self.orders_m[i],
                                                       energy_alpha[i] / total_energy,
                                                       energy_beta[i] / total_energy,
                                                       energy_gamma[i] / total_energy))

        return energy_alpha, energy_beta, energy_gamma

    def energy_matrix(self):
        energy_alpha = self._energy_helper(self.alpha)
        energy_beta = self._energy_helper(self.beta)
        energy_gamma = self._energy_helper(self.gamma)

        energy_radial = energy_alpha
        energy_poloidal = energy_alpha + energy_beta
        energy_toroidal = energy_gamma

        er = np.zeros((self.degree() + 1, 2 * self.degree() + 1))
        ep = np.zeros_like(er)
        et = np.zeros_like(ep)

        for i in range(len(energy_poloidal)):
            er[self.degrees_l[i], self.orders_m[i]] = energy_radial[i]
            ep[self.degrees_l[i], self.orders_m[i]] = energy_poloidal[i]
            et[self.degrees_l[i], self.orders_m[i]] = energy_toroidal[i]

        return (np.roll(er, self.degree(), axis=1),
                np.roll(ep, self.degree(), axis=1),
                np.roll(et, self.degree(), axis=1))

    def _energy_helper(self, complex_coeff):
        lTerm = self.degrees_l / (self.degrees_l + 1)  # TODO what is this for.
        m0mask = np.zeros_like(self.alpha)
        for i in range(len(self.alpha)):
            if self.orders_m[i] == 0:
                m0mask[i] = 1

        Es = 0.5 * complex_coeff * np.conj(complex_coeff)
        M0s = m0mask * 0.25 * (complex_coeff ** 2 + np.conj(complex_coeff) ** 2)

        return np.real(Es + M0s)

    def _dpml(self, p_lm, points_polar):
        """
        Derivative delta p_lm / delta points_polar
        :param p_lm:
        :param points_polar:
        :return:
        """
        if self._dpml_method == "roll":
            # Dumb implementation
            # Use chain rule
            # d P(cos(theta)) / d theta = d P / du * du / d theta
            DPml = (p_lm - np.roll(p_lm, 1)) / (np.cos(points_polar) - np.roll(np.cos(points_polar), 1)) * np.sin(
                points_polar)

            return DPml

        elif self._dpml_method == "roll2":
            # More descriptive implementation
            dp = p_lm - np.roll(p_lm, 1)
            du = np.cos(points_polar) - np.roll(np.cos(points_polar), 1)
            du_dtheta = np.sin(points_polar)

            return (dp / du) * du_dtheta

        else:
            dp_du = np.gradient(p_lm, np.cos(points_polar[0, :]), axis=1)
            du_dtheta = np.sin(points_polar)

            return dp_du * du_dtheta


def from_coefficients(shc,
                      dpml_method="gradient",  # For testing
                      ):
    degree_l, order_m, coeffs_lm = shc.as_arrays(include_unset=False)

    assert coeffs_lm.shape[1] <= 3, "Three complex coefficients or less are OK."

    split_coeffs = [coeffs_lm[:, _id] for _id in range(coeffs_lm.shape[1])]

    return LehmannZdi(degree_l, order_m, *split_coeffs, dpml_method=dpml_method)

