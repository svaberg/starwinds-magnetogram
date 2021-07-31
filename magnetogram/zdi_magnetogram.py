import numpy as np
import scipy as sp
import scipy.special

import logging

from stellarwinds.magnetogram import associated_legendre

log = logging.getLogger(__name__)

from stellarwinds import coordinate_transforms


class ZdiMagnetogram:
    r"""
    Reference implementation based on Vidotto et al. (2016) and Lehmann et al. (2018). The coordinates are
    radial (away from the center), polar (from the north pole towards the south pole) and azimuth (in the direction
    of rotation).

    The equations are based on equation 5 in Lehmann et al. (2018).

    This produces same plots as Folsom (2018, 2016) and presumably Donati (2006) except that the sign of the polar and
    azimuthal components is reversed. The reversal happens because Folsom uses phase in place of azimuth and latitude
    values running from 90 (north pole) to -90 (south pole).

    The terms toroidal and poloidal refer to the Mie representation (Backus et al. 1986). In ZDI, the toroidal field is
    governed by the $\gamma_{\ell m}$ coefficients (only), while the poloidal field is governed by two sets of
    coefficients $\alpha_{\ell m}$ and $\beta_{\ell m}$. The $\alpha$ coefficients are called radial and they are a part
    of the poloidal field. Two sets of coefficients should be enough for the Mie representation, but it is customary in
    the field to use 3 sets of coefficients as in Donati et al. (2006).

    In the ZDI world the B field is represented by radial, azimuthal, and meridional components in that
    order. Note: The  azimuthal B component is not last, but rather in the middle.

    TODO Split into a reference magnetogram and a fast magnetogram. Use lpmn in the fast magnetogram.
    """
    def __init__(self,
                 degrees_l, orders_m,
                 alpha_lm,
                 beta_lm=None,
                 gamma_lm=None,
                 dplm_method="gradient",  # For testing
                 ):
        r"""
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

        self.c_lm = get_c_lm(self.degrees_l, self.orders_m)

        self._dplm_method = dplm_method

    def __str__(self):
        s = "ZDI magnetogram\n"
        for l, m, a, b, g in zip(self.degrees_l, self.orders_m, self.alpha, self.beta, self.gamma):
            s += f"{l}, {m}: {a}, {b}, {g}\n"

        return s

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

    def get_azimuthal_poloidal_field(self, points_polar, points_azimuth):
        r"""
        Get the polar component of the poloidal field $B_{\phi, pol}$.
        :param points_polar:
        :param points_azimuth:
        :return: polar component of the poloidal field

        According to eq. (2-4, 5) and the text in Lehmann et al. (2018), this is the
        azimuthal component of the poloidal field.
        $B_{\phi ,\mathrm{pol}} =- \sum _{\ell m} \beta _{\ell m}
        \frac{im P_{\ell m} {\rm e}^{im\phi }}{(\ell + 1) \sin \theta }$
        """
        field_azimuthal_poloidal = np.zeros_like(points_azimuth, dtype=complex)

        for deg_l, ord_m, b_lm, c_lm in zip(self.degrees_l, self.orders_m, self.beta, self.c_lm):
            p_lm = sp.special.lpmv(ord_m, deg_l, np.cos(points_polar))
            field_azimuthal_poloidal -= (b_lm
                                     * c_lm
                                     * p_lm
                                     * 1.0j * ord_m * np.exp(1.0j * ord_m * points_azimuth)
                                     / (deg_l + 1)
                                     / np.sin(points_polar)
                                     )

        return np.real(field_azimuthal_poloidal)

    def get_azimuthal_toroidal_field(self, points_polar, points_azimuth):
        r"""
        Get the polar component of the toroidal field $B_{\phi, tor}$.
        :param points_polar:
        :param points_azimuth:
        :return: polar component of the toroidal field
        """
        log.warning("Please use the new method")
        assert False
        field_azimuthal_toroidal = np.zeros_like(points_azimuth, dtype=complex)

        for deg_l, ord_m, g_lm, c_lm in zip(self.degrees_l, self.orders_m, self.gamma, self.c_lm):
            p_lm = sp.special.lpmv(ord_m, deg_l, np.cos(points_polar))
            DPlm = self._dplm(p_lm, points_polar)

            field_azimuthal_toroidal += (g_lm
                                     * c_lm
                                     / (deg_l + 1)
                                     * DPlm
                                     * np.exp(1.0j * ord_m * points_azimuth)
                                     )

        return np.real(field_azimuthal_toroidal)

    def get_azimuthal_toroidal_field_new(self, points_polar, points_azimuth):
        r"""
        Get the polar component of the toroidal field $B_{\phi, tor}$.
        :param points_polar:
        :param points_azimuth:
        :return: polar component of the toroidal field
        """
        _, Pmn_d_cos_theta_result = self._calculate_lpmn(points_polar)

        fat_1_1_n = (self.gamma * self.c_lm / (self.degrees_l + 1))  # Shape is now (N,)
        fat_1_1_n = fat_1_1_n * np.exp(1.0j * self.orders_m * points_azimuth[..., np.newaxis])
        fat_1_1_n *= -Pmn_d_cos_theta_result  # TODO why is this minus sign required?

        fpp_i_j = np.sum(fat_1_1_n, axis=-1)

        return np.real(fpp_i_j)

    def get_azimuthal_field(self, points_polar, points_azimuth):
        r"""
        Get the total polar component of the field $B_{\phi}$.
        :param points_polar:
        :param points_azimuth:
        :return: polar field component
        """
        return (
                self.get_azimuthal_poloidal_field(points_polar, points_azimuth)
                + self.get_azimuthal_toroidal_field_new(points_polar, points_azimuth)
                )

    def get_polar_poloidal_field(self, points_polar, points_azimuth):
        r"""
        Get the azimuthal component of the poloidal field $B_{\theta, pol}$.
        :param points_polar:
        :param points_azimuth:
        :return: azimuthal component of the poloidal field
        """
        log.warning("Please use the new method")
        field_polar_poloidal = np.zeros_like(points_azimuth, dtype=complex)

        for deg_l, ord_m, b_lm, c_lm in zip(self.degrees_l, self.orders_m, self.beta, self.c_lm):
            p_lm = sp.special.lpmv(ord_m, deg_l, np.cos(points_polar))
            DPlm = self._dplm(p_lm, points_polar)

            field_polar_poloidal += (b_lm
                                         * c_lm
                                         / (deg_l + 1)
                                         * DPlm
                                         * np.exp(1.0j * ord_m * points_azimuth)
                                         )

        return np.real(field_polar_poloidal)

    def get_polar_poloidal_field_new(self, points_polar, points_azimuth):
        r"""
        Get the azimuthal component of the poloidal field $B_{\theta, pol}$.
        :param points_polar:
        :param points_azimuth:
        :return: azimuthal component of the poloidal field
        """

        _, Pmn_d_cos_theta_result = self._calculate_lpmn(points_polar)

        fpp_1_1_n = (self.beta * self.c_lm / (self.degrees_l + 1))  # Shape is now (N,)
        fpp_i_j_n = fpp_1_1_n * np.exp(1.0j * self.orders_m * points_azimuth[..., np.newaxis])
        fpp_i_j_n *= -Pmn_d_cos_theta_result  #TODO why is this minus sign required?

        fpp_i_j = np.sum(fpp_i_j_n, axis=-1)

        return np.real(fpp_i_j)

    def _calculate_lpmn(self, points_polar):
        return associated_legendre.calculate_lpmn(self.degrees_l, self.orders_m, points_polar)

    def get_polar_toroidal_field(self, points_polar, points_azimuth):
        r"""
        Get the azimuthal component of the toroidal field $B_{\theta, tor}$.
        :param points_polar:
        :param points_azimuth:
        :return: azimuthal component of the toroidal field
        """
        field_polar_toroidal = np.zeros_like(points_azimuth, dtype=complex)

        for deg_l, ord_m, g_lm, c_lm in zip(self.degrees_l, self.orders_m, self.gamma, self.c_lm):
            p_lm = sp.special.lpmv(ord_m, deg_l, np.cos(points_polar))
            field_polar_toroidal += (g_lm
                                         * c_lm
                                         * p_lm
                                         * 1.0j * ord_m * np.exp(1.0j * ord_m * points_azimuth)
                                         / (deg_l + 1)
                                         / np.sin(points_polar)
                                         )

        return np.real(field_polar_toroidal)

    def get_polar_field(self, points_polar, points_azimuth):
        r"""
        Get the total azimuthal component of the field $B_{\theta}$.
        :param points_polar:
        :param points_azimuth:
        :return: azimuthal field component
        """
        return (
                self.get_polar_poloidal_field_new(points_polar, points_azimuth)
                + self.get_polar_toroidal_field(points_polar, points_azimuth)
                )

    def get_field_strength(self, points_polar, points_azimuth):
        r"""
        Get field strength $B$.
        :param points_polar:
        :param points_azimuth:
        :return: field strength
        """
        field_radial = self.get_radial_field(points_polar, points_azimuth)
        field_polar = self.get_polar_field(points_polar, points_azimuth)
        field_azimuthal = self.get_azimuthal_field(points_polar, points_azimuth)

        return (field_radial**2 + field_polar**2 + field_azimuthal**2)**.5

    def get_all(self):
        r"""
        Get all the things in a dictionary.
        :return:
        """
        _dict = {}
        _dict["radial"] = {"poloidal": self.get_radial_poloidal_field,
                           "toroidal": self.get_radial_toroidal_field}  # This component is always zero; see definition.
        _dict["polar"] = {"poloidal": self.get_polar_poloidal_field_new,
                          "toroidal": self.get_polar_toroidal_field}
        _dict["azimuthal"] = {"poloidal": self.get_azimuthal_poloidal_field,
                              "toroidal": self.get_azimuthal_toroidal_field_new}

        return _dict

    def as_dipole(self):
        return self.as_restricted(degree_l_range=1)

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

        return ZdiMagnetogram(*new_args, self._dplm_method)

    def copy(self):
        """Create a deep copy"""
        return ZdiMagnetogram(np.copy(self.degrees_l),
                              np.copy(self.orders_m),
                              np.copy(self.alpha),
                              np.copy(self.beta),
                              np.copy(self.gamma),
                              self._dplm_method)

    def as_poloidal(self):
        result = self.copy()
        result.gamma = np.zeros_like(result.gamma)
        return result

    def as_toroidal(self):
        result = self.copy()
        result.alpha = np.zeros_like(result.alpha)
        result.beta = np.zeros_like(result.beta)
        return result

    def energy(self, *, show_fractions=False, dest=None):
        """
        Calculate energy as it is done in ZDIPy
        If dest is a dictionary this writes the values into dest. If dest is None the values are
        printed.
        :param show_fractions: if true print every degree/order combination
        :param dest: optional dictionary in which to save the values
        :return: energy in the alpha, beta and gamma parameters.
        """
        if dest is None:
            dict_ = dict()
        else:
            dict_ = dest

        fmt = "5.1"

        energy_alpha = self._energy_helper(self.alpha)
        energy_beta = self._energy_helper(self.beta, require_l_term=True)
        energy_gamma = self._energy_helper(self.gamma, require_l_term=True)

        total_energy = np.sum(energy_alpha) + np.sum(energy_beta) + np.sum(energy_gamma)
        total_energy_poloidal = np.sum(energy_alpha) + np.sum(energy_beta)
        total_energy_toroidal = np.sum(energy_gamma)

        dict_['magnetogram.total.energy.B2'] = total_energy

        # Ignore warnings when total energy or total poloidal energy is zero.
        with np.errstate(divide="ignore", invalid="ignore"):

            dict_['magnetogram.radial.energy.fraction'] = np.sum(energy_alpha) / total_energy
            dict_['magnetogram.poloidal.energy.fraction'] = total_energy_poloidal / total_energy
            dict_['magnetogram.toroidal.energy.fraction'] = total_energy_toroidal / total_energy

            if dest is None and show_fractions:
                print("Fraction of magnetic energy in each component. Should sum to 1.")
                print("  l    m       E_alpha        E_beta       E_gamma")
                for l, m, a, b, g in zip(self.degrees_l, self.orders_m,
                                         energy_alpha, energy_beta, energy_gamma):
                    print(f"{l:3}  {m:3}  {a/total_energy:e}  {b/total_energy:e}  {g/total_energy:e}")

            if dest is None:
                print(f"Total energy: {dict_['magnetogram.total.energy.B2']:.3g} (B^2)")
                print(f"Radial   energy* {dict_['magnetogram.radial.energy.fraction']:{fmt}%}")
                print(f"Poloidal energy  {dict_['magnetogram.poloidal.energy.fraction']:{fmt}%}")
                print(f"Toroidal energy  {dict_['magnetogram.toroidal.energy.fraction']:{fmt}%}")
                print("* The radial energy is part of the poloidal energy.")

            total_energy_poloidal = np.sum(energy_alpha) + np.sum(energy_beta)
            total_energy_toroidal = np.sum(energy_gamma)

            Epol_l1 = np.sum((energy_alpha + energy_beta)[self.degrees_l == 1])
            Epol_l2 = np.sum((energy_alpha + energy_beta)[self.degrees_l == 2])
            Epol_l3 = np.sum((energy_alpha + energy_beta)[self.degrees_l == 3])
            Etor_l1 = np.sum(energy_gamma[self.degrees_l == 1])
            Etor_l2 = np.sum(energy_gamma[self.degrees_l == 2])
            Etor_l3 = np.sum(energy_gamma[self.degrees_l == 3])

            dict_['magnetogram.total.energy.dipole.fraction'] = Epol_l1 / total_energy_poloidal
            dict_['magnetogram.total.energy.quadrupole.fraction'] = Epol_l2 / total_energy_poloidal
            dict_['magnetogram.total.energy.octopole.fraction'] = Epol_l3 / total_energy_poloidal

            dict_['magnetogram.total.energy.toroidal.l1.fraction'] = Etor_l1 / total_energy_toroidal
            dict_['magnetogram.total.energy.toroidal.l2.fraction'] = Etor_l2 / total_energy_toroidal
            dict_['magnetogram.total.energy.toroidal.l3.fraction'] = Etor_l3 / total_energy_toroidal

            # Note this overlaps with total energy dipole fraction, etc. above but contains all the l values.
            for deg_l in range(1, 16):  # From 1 to 15 inclusive
                pol_energy_at_deg_l = np.sum((energy_alpha + energy_beta)[self.degrees_l == deg_l])
                dict_[f'magnetogram.pol.energy.l{deg_l}.fraction'] = pol_energy_at_deg_l/total_energy_poloidal
            for deg_l in range(1, 16):  # From 1 to 15 inclusive
                tor_energy_at_deg_l = np.sum(energy_gamma[self.degrees_l == deg_l])
                dict_[f'magnetogram.tor.energy.l{deg_l}.fraction'] = tor_energy_at_deg_l/total_energy_toroidal

            if dest is None:
                print(f"poloidal l1 (dipole):     {Epol_l1 / total_energy_poloidal:{fmt}%} (% poloidal)")
                print(f"poloidal l2 (quadrupole): {Epol_l2 / total_energy_poloidal:{fmt}%} (% poloidal)")
                print(f"poloidal l3 (octopole):   {Epol_l3 / total_energy_poloidal:{fmt}%} (% poloidal)")
                print(f"toroidal l1: {Etor_l1 / total_energy_toroidal:{fmt}%} (% toroidal)")
                print(f"toroidal l2: {Etor_l2 / total_energy_toroidal:{fmt}%} (% toroidal)")
                print(f"toroidal l3: {Etor_l3 / total_energy_toroidal:{fmt}%} (% toroidal)")

            radial_axisym = np.sum(energy_alpha[self.orders_m == 0])
            poloidal_axisym = np.sum((energy_alpha + energy_beta)[self.orders_m == 0])
            toroidal_axisym = np.sum(energy_gamma[self.orders_m == 0])
            total_axisym = poloidal_axisym + toroidal_axisym

            dict_['magnetogram.total.energy.axisymmetric.fraction'] = total_axisym / total_energy

            assert np.allclose(total_energy, total_energy_poloidal + total_energy_toroidal)
            dict_['magnetogram.total.energy.radial.axisymmetric.fraction'] = radial_axisym / total_energy_poloidal
            dict_['magnetogram.total.energy.poloidal.axisymmetric.fraction'] = poloidal_axisym / total_energy_poloidal
            dict_['magnetogram.total.energy.toroidal.axisymmetric.fraction'] = toroidal_axisym / total_energy_toroidal

            symdipfrac = float((energy_alpha + energy_beta)[np.logical_and(self.degrees_l == 1, self.orders_m == 0)] /
                               np.sum((energy_alpha + energy_beta)[self.degrees_l == 1]))
            dict_['magnetogram.total.energy.dipole.axisymmetric.fraction'] = symdipfrac

            if dest is None:
                print(f"axisymmetric:          {total_axisym / total_energy:{fmt}%} (% total)")
                print(f"poloidal axisymmetric: {poloidal_axisym / total_energy_poloidal:{fmt}%} (% poloidal)")
                print(f"toroidal axisymmetric: {toroidal_axisym / total_energy_toroidal:{fmt}%} (% toroidal)")
                print(f"dipole axisymmetric:   {symdipfrac:{fmt}%} (% dipole)")

        return energy_alpha, energy_beta, energy_gamma


    def energy_matrix(self):
        """
        Calculate matrix of magnetogram energy indexed by degree and order.
        Returns radial energy from alpha coefficients,
        poloidal energy from alpha and beta coefficients (INCLUDES RADIAL)
        toroidal component from gamma coefficients.
        :return: radial, poloidal, toroidal energy matrices
        """
        energy_alpha = self._energy_helper(self.alpha)
        energy_beta = self._energy_helper(self.beta, require_l_term=True)
        energy_gamma = self._energy_helper(self.gamma, require_l_term=True)

        energy_radial = energy_alpha
        energy_poloidal = energy_alpha + energy_beta
        energy_toroidal = energy_gamma

        # Note that order runs from -m to m here.
        er = np.zeros((self.degree() + 1, 2 * self.degree() + 1))
        ep = np.zeros_like(er)
        et = np.zeros_like(ep)

        er[self.degrees_l, self.orders_m] = energy_radial
        ep[self.degrees_l, self.orders_m] = energy_poloidal
        et[self.degrees_l, self.orders_m] = energy_toroidal

        # Roll on the degree axis so that the elements in the
        # matrix are l = 0, 1, 2, ..., n, 0, ..., 0, -n, ..., -2, -1
        return (np.roll(er, self.degree(), axis=1),
                np.roll(ep, self.degree(), axis=1),
                np.roll(et, self.degree(), axis=1))

    def _energy_helper(self, complex_coeff, require_l_term=False):
        """The require_l_term must be true when calculating with beta and gamma."""

        m0mask = np.zeros_like(self.alpha)
        m0mask[self.orders_m == 0] = 1

        Es = 0.5 * complex_coeff * np.conj(complex_coeff)
        M0s = m0mask * 0.25 * (complex_coeff ** 2 + np.conj(complex_coeff) ** 2)

        result = Es + M0s

        if require_l_term:
            result *= self.degrees_l / (self.degrees_l + 1)

        return np.real_if_close(result)

    def _dplm(self, p_lm, points_polar):
        """
        Derivative delta p_lm / delta points_polar
        :param p_lm:
        :param points_polar:
        :return:
        """
        if self._dplm_method == "roll":
            # Dumb implementation
            # Use chain rule
            # d P(cos(theta)) / d theta = d P / du * du / d theta
            DPlm = (p_lm - np.roll(p_lm, 1)) / (np.cos(points_polar) - np.roll(np.cos(points_polar), 1)) * np.sin(
                points_polar)

            return DPlm

        elif self._dplm_method == "roll2":
            # More descriptive implementation
            dp = p_lm - np.roll(p_lm, 1)
            du = np.cos(points_polar) - np.roll(np.cos(points_polar), 1)
            du_dtheta = np.sin(points_polar)

            return (dp / du) * du_dtheta

        else:
            dp_du = np.gradient(p_lm, np.cos(points_polar[0, :]), axis=1)
            du_dtheta = np.sin(points_polar)

            return dp_du * du_dtheta

    def get_cartesian_field(self, points_polar, points_azimuth):
        """
        Get full field in cartesian coordinates.
        """
        # Field components in spherical coordinates
        frpa = [self.get_radial_field(points_polar, points_azimuth),
                self.get_polar_field(points_polar, points_azimuth),
                self.get_azimuthal_field(points_polar, points_azimuth)]

        return _cartesian_from_spherical_helper(*frpa, points_polar, points_azimuth)

    def get_cartesian_poloidal_field(self, points_polar, points_azimuth):
        """
        Get poloidal field in cartesian coordinates.
        """
        # Field components in spherical coordinates
        frpa = [self.get_radial_poloidal_field(points_polar, points_azimuth),
                self.get_polar_poloidal_field(points_polar, points_azimuth),
                self.get_azimuthal_poloidal_field(points_polar, points_azimuth)]

        return _cartesian_from_spherical_helper(*frpa, points_polar, points_azimuth)

    def get_cartesian_toroidal_field(self, points_polar, points_azimuth):
        """
        Get toroidal field in cartesian coordinates.
        """
        # Field components in spherical coordinates
        frpa = [self.get_radial_toroidal_field(points_polar, points_azimuth),
                self.get_polar_toroidal_field(points_polar, points_azimuth),
                self.get_azimuthal_toroidal_field(points_polar, points_azimuth)]

        return _cartesian_from_spherical_helper(*frpa, points_polar, points_azimuth)


def from_coefficients(shc, *,
                      dplm_method="gradient",  # For testing
                      ):
    degree_l, order_m, coeffs_lm = shc.as_arrays(include_unset=False)

    if np.min(order_m) < 0:
        log.warning("Creating ZDI magnetogram with negative orders;")
        log.warning("use map_to_positive_orders first to avoid this warning.")

    assert coeffs_lm.shape[1] <= 3, "Three complex coefficients or less are OK."

    split_coeffs = [coeffs_lm[:, _id] for _id in range(coeffs_lm.shape[1])]

    return ZdiMagnetogram(degree_l, order_m, *split_coeffs, dplm_method=dplm_method)


def _cartesian_from_spherical_helper(fr, fp, fa, pp, pa):
    assert fr.shape == pp.shape, "Expected matching shapes"
    assert fp.shape == pp.shape, "Expected matching shapes"
    assert fa.shape == pp.shape, "Expected matching shapes"
    # To carry out the transformation, flatten the polar coordinate arrays and stack them
    # calculate the transformation matrix, and apply it to the stack field_rpa.
    field_rpa = np.stack([c.flatten() for c in (fr, fp, fa)], axis=-1)
    transformation_matrix = coordinate_transforms.spherical_to_rectangular_transformation_matrix(pp.flatten(),
                                                                                                 pa.flatten())
    field_xyz = transformation_matrix @ field_rpa[:, :, np.newaxis]
    # Get rid of last dimension which is has length 1
    assert field_xyz.shape[-1] == 1, "Expected last dimension size to be 1"
    assert field_xyz.shape[-2] == 3, "Expected second last dimension size to be 3"
    field_xyz = field_xyz[..., 0].reshape(pp.shape + (3,))
    return field_xyz


def get_c_lm(degrees_l, orders_m, method='low_order'):
    # Calculate c_lm as described in article.
    degrees_l = np.asarray(degrees_l)
    orders_m = np.asarray(orders_m)
    
    if method == "low_order":
        c_lm2 = (
                (2 * degrees_l + 1)
                / (4 * np.pi)
                * sp.special.factorial(degrees_l - orders_m)
                / sp.special.factorial(degrees_l + orders_m)
        )
    else:
        # This takes better care of the cancellations in (l-m)!/(l+m)!
        @np.vectorize
        def factorial_expression(l, m):
            return np.exp(-np.sum(np.log(np.arange(l - m + 1, l + m + 1))))

        c_lm2 = (2 * degrees_l + 1) / (4 * np.pi) * factorial_expression(degrees_l, orders_m)

    return np.sqrt(c_lm2)


