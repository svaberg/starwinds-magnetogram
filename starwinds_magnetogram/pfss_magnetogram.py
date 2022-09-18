import logging
log = logging.getLogger(__name__)
import numpy as np
import scipy as sp
from starwinds_magnetogram import coordinate_transforms


# This is the default in the SWMF so follow them.
default_radius_star = 1
default_radius_source_surface = 2.5


def theta_lm(deg_l, ord_m, points_polar):
    r"""
    Calculate $\Theta_{\ell m}(\theta)$
    :param deg_l: Degree $\ell$ of coefficient
    :param ord_m: Order $m$ of coefficient
    :param points_polar: 2d array of polar coordinate values
    :return: 2d array of $\Theta_{\ell m}(\theta)$ values
    TODO redo this with lpmn like in zdi magnetogram.
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

    #
    # Estimate derivative of associated Legendre polynomial.
    #
    # This is a hack, find a better method.
    # The analytic derivative goes all over the place at the endpoints.
    # http://www.autodiff.org/ad16/Oral/Buecker_Legendre.pdf
    _dp = 1e-2
    dv = _inner(deg_l, ord_m, points_polar + _dp) - value
    du = .5 * (np.cos(points_polar + _dp) - np.cos(points_polar - _dp))
    dudx = np.sin(points_polar)

    # This is just a fancy way of writing `deriv = -dv/du * dudx` to get better poles
    with np.errstate(divide='ignore'):
        dvdu = np.divide(dv, du, out=np.zeros_like(dv), where=du != 0)
        deriv = -dvdu * dudx

    # Hack continues; if `polar` is 0 or pi, set the derivative to 0
    # At least it is better than infinity
    # Only if $\ell=1$.
    if deg_l == 1:
        deriv[points_polar == 0] = 0

    return value, deriv


def calculate_all_theta(degree_l, order_m, points_polar, scipy=True):
    
    degree_l = np.asarray(degree_l)
    order_m = np.asarray(order_m)

    # Calculate all the theta_lm and theta_lm' values in one go using scipy
    from starwinds_magnetogram.associated_legendre import calculate_lpmn
    Pmn_cos_theta_result, Pmn_d_cos_theta_result = calculate_lpmn(degree_l, order_m, points_polar)

    d0 = 0 + (np.asarray(order_m) == 0)
    clm = (-1) ** order_m * np.sqrt(sp.special.factorial(degree_l - order_m)
                                  / sp.special.factorial(degree_l + order_m)) * np.sqrt(2 - d0)

    buffered_theta_scipy = Pmn_cos_theta_result * clm
    buffered_dtheta_scipy = Pmn_d_cos_theta_result * clm

    # The old method.
    buffered_theta = []
    buffered_dtheta = []

    for (deg_l, ord_m) in zip(degree_l, order_m):
        result = theta_lm(deg_l, ord_m, points_polar)
        buffered_theta.append(result[0])
        buffered_dtheta.append(result[1])

    buffered_theta = np.stack(buffered_theta, axis=-1)
    buffered_dtheta = np.stack(buffered_dtheta, axis=-1)

    assert Pmn_cos_theta_result.shape == buffered_theta.shape
    assert Pmn_d_cos_theta_result.shape == buffered_theta.shape
    assert buffered_theta_scipy.shape == buffered_theta.shape
    assert buffered_dtheta_scipy.shape == buffered_theta.shape

    assert np.allclose(buffered_theta_scipy, buffered_theta)
    # assert np.allclose(buffered_dtheta_scipy, buffered_dtheta)

    # Return here.
    if scipy:
        return buffered_theta_scipy, buffered_dtheta_scipy
    else:
        return buffered_theta, buffered_dtheta


def r_l(deg_l, r, r0, rss):
    r"""
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
    r"""
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


def evaluate_spherical(
        coefficients,
        points_radial, points_polar, points_azimuth,
        radius_star=None,
        radius_source_surface=None):
    r"""
    Evaluate the radial, polar and azimuthal field components of a magnetogram represented
    as a set of real spherical harmonics on Stanford PFSS form. The unit is the same as the
    unit of the magnetogram, normally Gauss.
    :param coefficients: Coefficients object
    :param points_radial: 3d array of radial coordinate values
    :param points_polar: 3d array of polar coordinate values
    :param points_azimuth: 3d array of azimuth coordinate values
    :param radius_star: Stellar radius
    :param radius_source_surface: Source surface radius
    :return:
    """
    if radius_star is None:
        radius_star = default_radius_star

    if radius_source_surface is None:
        radius_source_surface = default_radius_source_surface

    assert radius_star < radius_source_surface

    points_radial = np.atleast_1d(points_radial)
    points_polar = np.atleast_1d(points_polar)
    points_azimuth = np.atleast_1d(points_azimuth)
    assert points_polar.shape == points_azimuth.shape, "Shape mismatch."

    # This is to accomodate points_radial having just 1 element
    points_radial = points_radial * np.ones_like(points_polar)
    assert points_radial.shape == points_polar.shape, "Shape mismatch."

    assert radius_star < radius_source_surface

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

    buffered_theta, buffered_dtheta = calculate_all_theta(degree_l, order_m, points_polar)

    # Loop over magnetogram coefficient lines
    for coeff_id, (deg_l, ord_m, g_lm, h_lm) in enumerate(zip(degree_l,
                                                       order_m,
                                                       np.real(alpha_lm),
                                                       np.imag(alpha_lm))):

        _r_l = r_l(deg_l, points_radial, radius_star, radius_source_surface)
        _theta_lm = theta_lm(deg_l, ord_m, points_polar)
        _phi_lm = phi_lm(ord_m, g_lm, h_lm, points_azimuth)

        # assert np.allclose(buffered_theta[..., coeff_id], _theta_lm[0])
        # assert np.allclose(buffered_dtheta[..., coeff_id], _theta_lm[1])

        del _theta_lm

        unscaled_radial    = _r_l[1] * buffered_theta[..., coeff_id]  * _phi_lm[0]
        unscaled_polar     = _r_l[0] * buffered_dtheta[..., coeff_id] * _phi_lm[0]
        unscaled_azimuthal = _r_l[0] * buffered_theta[..., coeff_id]  * _phi_lm[1]

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


def evaluate_cartesian(
        coefficients,
        px, py, pz,
        radius_star=None,
        radius_source_surface=None):
    r"""
    Evaluate the radial, polar and azimuthal field components of a magnetogram represented
    as a set of real spherical harmonics on Stanford PFSS form. The unit is the same as the
    unit of the magnetogram, normally Gauss.

    This function calls evaluate_spherical internally.
    :param coefficients: Coefficients object
    :param px: Array of x coordinates
    :param py: Array of y coordinates
    :param pz: Array of z coordinates
    :param radius_star: Stellar radius
    :param radius_source_surface: Source surface radius
    :return:
    """
    if radius_star is None:
        radius_star = default_radius_star

    if radius_source_surface is None:
        radius_source_surface = default_radius_source_surface

    assert radius_star < radius_source_surface

    px = np.atleast_1d(px)
    py = np.atleast_1d(py)
    pz = np.atleast_1d(pz)
    assert px.shape == py.shape
    assert px.shape == pz.shape

    pr, pp, pa = coordinate_transforms.spherical_coordinates_from_rectangular(px, py, pz)
    assert pr.shape == px.shape
    assert pp.shape == px.shape
    assert pa.shape == px.shape

    fr, fp, fa = evaluate_spherical(coefficients,
                                    pr, pp, pa,
                                    radius_star=radius_star,
                                    radius_source_surface=radius_source_surface)

    assert fr.shape == pr.shape
    assert fp.shape == pr.shape
    assert fa.shape == pr.shape

    # To carry out the transformation, flatten the polar coordinate arrays and stack them
    # calculate the transformation matrix, and apply it to the stack Frpa.
    Frpa = np.stack([c.flatten() for c in (fr, fp, fa)], axis=-1)

    transformation_matrix = coordinate_transforms.spherical_to_rectangular_transformation_matrix(pp.flatten(),
                                                                                                 pa.flatten())
    Fxyz = transformation_matrix @ Frpa[:, :, np.newaxis]

    # Get rid of last dimension which is has length 1
    assert Fxyz.shape[-1] == 1, "Last dimension of Fxyz must have size 1"
    Fxyz = Fxyz[..., 0]

    # Reshape fx-fz to the original shape of fr
    fx = Fxyz[:, 0].reshape(px.shape)
    fy = Fxyz[:, 1].reshape(px.shape)
    fz = Fxyz[:, 2].reshape(px.shape)
    return fr, fp, fa, fx, fy, fz


def normal_plane(p1, p2, normal):
    r"""
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
