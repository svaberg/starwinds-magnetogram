import numpy as np
import scipy as sp

import logging

from stellarwinds import coordinate_transforms

log = logging.getLogger(__name__)


def theta_lm(deg_l, ord_m, points_polar):
    """
    Calculate $\Theta_{\ell m}(\theta)$
    :param deg_l: Degree $\ell$ of coefficient
    :param ord_m: Order $m$ of coefficient
    :param points_polar: 2d array of polar coordinate values
    :return: 2d array of $\Theta_{\ell m}(\theta)$ values
    """
    def _inner(deg_l, ord_m, points_polar):
        #  Associated Legendre polynomial $P_{\ell}^{m}(\cos\theta)$
        plm = sp.special.lpmv(ord_m, deg_l, np.cos(points_polar))
        # Correction (scaling)
        d0 = 0 + (ord_m == 0)
        clm = (-1) ** ord_m * np.sqrt(sp.special.factorial(deg_l - ord_m)
                                      / sp.special.factorial(deg_l + ord_m)) * np.sqrt(2 - d0)
        value = clm * plm
        return value

    value = _inner(deg_l, ord_m, points_polar)

    # Estimate derivative of associated Legendre polynomial (TODO different approach??)
    # This is a hack. TODO this does not work for zero
    # This derivative goes all over the place at the endpoints.
    # http://www.autodiff.org/ad16/Oral/Buecker_Legendre.pdf
    if len(points_polar.shape) > 2:
        _dp = 1e-2
        dv = _inner(deg_l, ord_m, points_polar + _dp) - value
        du =.5 * (np.cos(points_polar + _dp) - np.cos(points_polar - _dp))
        dudx = np.sin(points_polar)
        deriv = -dv/du * dudx
    else:
        deriv = (value - np.roll(value, 1)) / (np.cos(points_polar) - np.roll(np.cos(points_polar), 1)) * np.sin(points_polar)

    # Hack continues; if polar is 0 or pi, set the derivative to 0
    # At least it is better than infinity
    if deg_l == 1:
        deriv[points_polar == 0] = 0

    return value, deriv


def r_l(deg_l, r, r0, rss):
    """
    Calculate $R_\ell(r;r_\star, r_\text{ss})$
    :param deg_l: Degree $\ell$ of coefficient
    :param r: Evaluation radius
    :param r0: Stellar radius
    :param rss: Source surface radius
    :return: value of $R_\ell(r;r_\star, r_\text{ss})$
    """
    #
    # Calculate r_l
    #
    # Calculate $\frac{r_\star^{\ell+2}}{r^{\ell+1}}$
    a = r * (r0 / r) ** (deg_l + 2)
    # Calculate $\frac{r^{2\ell+1}}{r_\text{ss}^{2\ell+1}}$
    b = 1 - (r/rss)**(2*deg_l + 1)
    # Calculate $\ell + 1 + \ell \left(\frac{r_\star}{r_\text{ss}}\right)^{2\ell+1}$
    e = deg_l + 1 + deg_l * (r0/rss)**(2*deg_l + 1)
    value = a * b / e

    #
    # Calculate r_l'
    #
    c = (r0/r)**(deg_l + 2)
    d = deg_l + 1 + deg_l * (r/rss)**(2*deg_l + 1)
    deriv = -c * d / e

    return value, deriv


def phi_lm(ord_m, g_lm, h_lm, phi):
    """
    Calculate $Phi_{\ell m}(\phi)$
    :param ord_m: Order $m$ of coefficient
    :param g_lm: Cosine-like real $g_{\ell m}$ coefficient
    :param h_lm: Sine-like real  $h_{\ell m}$  coefficient
    :param phi: 2d array of azimuth coordinate values
    :return: 2d array of $Phi_{\ell m}(\phi)$ values
    """
    value = (+g_lm * np.cos(ord_m * phi) + h_lm * np.sin(ord_m * phi))
    deriv = (-g_lm * np.sin(ord_m * phi) + h_lm * np.cos(ord_m * phi)) * ord_m

    return value, deriv


def evaluate_on_sphere(
        degree_l, order_m, cosine_coefficients_g, sine_coefficients_h,
        points_polar, points_azimuth,
        radius=None, radius_star=1, radius_source_surface=3):
    """
    Evaluate the radial, polar and azimuthal field components of a magnetogram represented
    as a set of real spherical harmonics on Stanford PFSS form. The unit is the same as the
    unit of the magnetogram, normally Gauss.
    :param degree_l: Degree $\ell$ of coefficients
    :param order_m:  Order $m$ of coefficients
    :param cosine_coefficients_g: Cosine-like real $g_{\ell m}$ coefficients
    :param sine_coefficients_h: Sine-like real  $h_{\ell m}$  coefficients
    :param points_polar: 2d array of polar coordinate values
    :param points_azimuth: 2d array of azimuth coordinate values
    :param radius: Evaluation radius
    :param radius_star: Stellar radius
    :param radius_source_surface: Source surface radius
    :return:
    """
    assert(points_polar.shape == points_azimuth.shape)
    assert np.min(order_m) >= 0, "Stanford PFSS expects only positive orders (TBC)."

    if radius is None:
        radius = radius_star

    # Initialize field variables
    field_radial = np.zeros_like(points_polar)
    field_polar = np.zeros_like(field_radial)
    field_azimuthal = np.zeros_like(field_radial)

    # Loop over magnetogram coefficient lines
    for (deg_l, ord_m, g_lm, h_lm) in zip(degree_l, order_m, cosine_coefficients_g, sine_coefficients_h):

        _r_l = r_l(deg_l, radius, radius_star, radius_source_surface)
        _theta_lm = theta_lm(deg_l, ord_m, points_polar)
        _phi_lm = phi_lm(ord_m, g_lm, h_lm, points_azimuth)

        unscaled_radial    = _r_l[1] * _theta_lm[0] * _phi_lm[0]
        unscaled_polar     = _r_l[0] * _theta_lm[1] * _phi_lm[0]
        unscaled_azimuthal = _r_l[0] * _theta_lm[0] * _phi_lm[1]

        field_radial    -= unscaled_radial
        field_polar     -= unscaled_polar / radius
        field_azimuthal -= np.divide(unscaled_azimuthal, radius * np.sin(points_polar),
                                     out=np.zeros_like(field_azimuthal),
                                     where=unscaled_azimuthal != 0)

    return field_radial, field_polar, field_azimuthal



def evaluate_in_space(
        # degree_l, order_m, cosine_coefficients_g, sine_coefficients_h,
        coefficients,
        points_radial, points_polar, points_azimuth,
        radius_star=1, radius_source_surface=3):
    """
    Evaluate the radial, polar and azimuthal field components of a magnetogram represented
    as a set of real spherical harmonics on Stanford PFSS form. The unit is the same as the
    unit of the magnetogram, normally Gauss.
    TODO this is not dry with evaluate on sphere
    :param coefficients: Coefficients object
    :param points_radial: 3d array of radial coordinate values
    :param points_polar: 3d array of polar coordinate values
    :param points_azimuth: 3d array of azimuth coordinate values
    :param radius_star: Stellar radius
    :param radius_source_surface: Source surface radius
    :return:
    """
    assert points_polar.shape == points_azimuth.shape, "Shape mismatch."

    # The PFSS method is not valid inside the star
    _invalid_inner_ids = np.where(points_radial < radius_star)
    # The PFSS method is not valid outside the source surface
    _invalid_outer_ids = np.where(points_radial > radius_source_surface)
    original_radius = points_radial
    points_radial = np.minimum(points_radial, radius_source_surface)

    # Initialize field variables
    field_radial = np.zeros_like(points_polar, dtype=float)
    field_polar = np.zeros_like(field_radial, dtype=float)
    field_azimuthal = np.zeros_like(field_radial, dtype=float)

    degree_l, order_m, alpha_lm = coefficients.as_arrays(include_unset=False)
    assert np.min(order_m) >= 0, "Stanford PFSS expects only positive orders (TBC)."

    # Loop over magnetogram coefficient lines
    for (deg_l, ord_m, g_lm, h_lm) in zip(degree_l,
                                          order_m,
                                          np.real(alpha_lm),
                                          np.imag(alpha_lm)):

        _r_l = r_l(deg_l, points_radial, radius_star, radius_source_surface)
        _theta_lm = theta_lm(deg_l, ord_m, points_polar)
        _phi_lm = phi_lm(ord_m, g_lm, h_lm, points_azimuth)

        unscaled_radial    = _r_l[1] * _theta_lm[0] * _phi_lm[0]
        unscaled_polar     = _r_l[0] * _theta_lm[1] * _phi_lm[0]
        unscaled_azimuthal = _r_l[0] * _theta_lm[0] * _phi_lm[1]

        field_radial    -= unscaled_radial
        field_polar     -= unscaled_polar / points_radial
        field_azimuthal -= np.divide(unscaled_azimuthal, points_radial * np.sin(points_polar),
                                     out=np.zeros_like(field_azimuthal),
                                     where=unscaled_azimuthal != 0)

    # Set field inside star to zero
    field_radial[_invalid_inner_ids] = 0.0
    field_polar[_invalid_inner_ids] = 0.0
    field_azimuthal[_invalid_inner_ids] = 0.0

    # Extrapolate field outside source surface from source surface values.
    field_radial[_invalid_outer_ids]    *= (points_radial[_invalid_outer_ids]/original_radius[_invalid_outer_ids])**2
    field_polar[_invalid_outer_ids]     *= (points_radial[_invalid_outer_ids]/original_radius[_invalid_outer_ids])**2
    field_azimuthal[_invalid_outer_ids] *= (points_radial[_invalid_outer_ids]/original_radius[_invalid_outer_ids])**2

    return field_radial, field_polar, field_azimuthal


def evaluate_on_slice(coefficients, px, py, pz, radius_source_surface, radius_star):
    """
    Is this really on a slice, or just in cartesian coordinates ?? The unit is the same as the
    unit of the magnetogram, normally Gauss.
    :param coefficients:
    :param px:
    :param py:
    :param pz:
    :param radius_source_surface:
    :param radius_star:
    :return:
    """

    pr, pp, pa = coordinate_transforms.spherical_coordinates_from_rectangular(px, py, pz)
    fr, fp, fa = evaluate_in_space(coefficients,
                                   pr, pp, pa,
                                   radius_star=radius_star,
                                   radius_source_surface=radius_source_surface)
    assert fr.shape == pr.shape
    assert fp.shape == pr.shape
    assert fa.shape == pr.shape
    Frpa = np.stack([c.flatten() for c in (fr, fp, fa)], axis=-1)
    transformation_matrix = coordinate_transforms.spherical_to_rectangular_transformation_matrix(pp.flatten(),
                                                                                                 pa.flatten())
    Fxyz = transformation_matrix @ Frpa[:, :, np.newaxis]
    # Get rid of last dimension which is 1
    Fxyz = np.squeeze(Fxyz)
    fx = Fxyz[:, 0].reshape(px.shape)
    fy = Fxyz[:, 1].reshape(px.shape)
    fz = Fxyz[:, 2].reshape(px.shape)
    return fr, fp, fa, fx, fy, fz


def normal_plane(p1, p2, normal):
    """
    Return normal plane (does not alter magnetogram)
    :param p1:
    :param p2:
    :param normal:
    :return:
    """
    p3 = np.zeros_like(p1)
    if normal == "x":
        return p3[..., np.newaxis], p1[..., np.newaxis], p2[..., np.newaxis]
    elif normal == "y":
        return p1[..., np.newaxis], p3[..., np.newaxis], p2[..., np.newaxis]
    if normal == "z":
        return p1[..., np.newaxis], p2[..., np.newaxis], p3[..., np.newaxis]
