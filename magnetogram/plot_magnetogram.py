import numpy as np
import scipy as sp

import logging
log = logging.getLogger(__name__)


# Evaluate spherical harmonics for on a polar, azimuthal grid)
def evaluate_real_magnetogram_stanford_pfss_reference(degree_l, order_m, cosine_coefficients_g, sine_coefficients_h,
                                                      points_polar, points_azimuth, radius=None, r0=1, rss=3):
    if radius is None:
        radius = r0

    field_radial = np.zeros_like(points_azimuth)
    field_polar = np.zeros_like(field_radial)
    field_azimuthal = np.zeros_like(field_radial)

    for row_id in range(len(degree_l)):

        deg_l = degree_l[row_id]
        ord_m = order_m[row_id]
        g_lm = cosine_coefficients_g[row_id]
        h_lm = sine_coefficients_h[row_id]

        p_lm = sp.special.lpmv(ord_m, deg_l, np.cos(points_polar))

        if True:  # Apply correction
            # https://en.wikipedia.org/wiki/Spherical_harmonics#Condon%E2%80%93Shortley_phase
            # https://en.wikipedia.org/wiki/Spherical_harmonics#Conventions
            d0 = 0 + (ord_m == 0)
            p_lm *= (-1) ** ord_m * np.sqrt(sp.special.factorial(deg_l - ord_m) / sp.special.factorial(deg_l + ord_m)) * np.sqrt(2 - d0)

        # Estimate DPml
        DPml = (p_lm - np.roll(p_lm, 1)) / (np.cos(points_polar) - np.roll(np.cos(points_polar), 1)) * np.sin(points_polar)

        fixed = (r0 / radius) ** (deg_l + 2) / (deg_l + 1 + deg_l * (r0 / rss) ** (2 * deg_l + 1))

        field_radial += p_lm * (g_lm * np.cos(ord_m * points_azimuth) + h_lm * np.sin(ord_m * points_azimuth)) * (
                deg_l + 1 + deg_l * (radius / rss) ** (2 * deg_l + 1)) * fixed
        field_polar -= DPml * (g_lm * np.cos(ord_m * points_azimuth) + h_lm * np.sin(ord_m * points_azimuth)) * (
                1 - (radius / rss) ** (2 * deg_l + 1)) * fixed
        field_azimuthal -= p_lm * (g_lm * np.sin(ord_m * points_azimuth) - h_lm * np.cos(ord_m * points_azimuth)) * (
                1 - (radius / rss) ** (2 * deg_l + 1)) * fixed

    return field_radial, field_polar, field_azimuthal

def theta_lm(deg_l, ord_m, points_polar):
    """
    Calculate $\Theta_{\ell m}(\theta)$
    :param deg_l: Degree $\ell$ of coefficient
    :param ord_m: Order $m$ of coefficient
    :param points_polar: 2d array of polar coordinate values
    :return: 2d array of $\Theta_{\ell m}(\theta)$ values
    """
    #  Associated Legendre polynomial $P_{\ell}^{m}(\cos\theta)$
    plm = sp.special.lpmv(ord_m, deg_l, np.cos(points_polar))
    # Correction (scaling)
    d0 = 0 + (ord_m == 0)
    clm = (-1) ** ord_m * np.sqrt(sp.special.factorial(deg_l - ord_m)
                                  / sp.special.factorial(deg_l + ord_m)) * np.sqrt(2 - d0)
    value = clm * plm

    # Estimate derivative of Legendre polynomial (TODO different approach??)
    deriv = (value - np.roll(value, 1)) / (np.cos(points_polar) - np.roll(np.cos(points_polar), 1)) * np.sin(points_polar)

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


def evaluate_real_magnetogram_stanford_pfss(
        degree_l, order_m, cosine_coefficients_g, sine_coefficients_h,
        points_polar, points_azimuth,
        radius=None, radius_star=1, radius_source_surface=3):
    """
    Evaluate the radial, polar and azimuthal field components of a magnetogram represented
    as a set of real spherical harmonics on Stanford PFSS form.
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


# TODO move stuff from test_plot_magnetogram.plot_test to here.
def pretty_plot(polar, azimuth, z, ax, color_range=(), color_map='RdBu_r'):
    """

    :param polar:
    :param azimuth:
    :param z:
    :param ax:
    :param color_range:
    :param color_map:
    """
    if len(color_range) == 1:
        color_min = -color_range[0]
        color_max = color_range[0]
    elif len(color_range) == 2:
        color_min = color_range[0]
        color_max = color_range[1]
    else:
        color_max = np.absolute(z[~np.isnan(z)]).max()
        color_min = np.absolute(z[~np.isnan(z)]).min()

    log.info('Color map %s, range [%f, %f].' % (color_map, color_min, color_max))

    # Print zero contour
    ax.contour(180 / np.pi * azimuth.T, 180 / np.pi * polar.T, z.T, levels=[0], colors=('g',), linewidths=.25)

    q = ax.contourf(180 / np.pi * azimuth.T, 180 / np.pi * polar.T, z.T, 128, cmap=color_map,
                    vmin=color_min, vmax=color_max)

    ax.set_xticks(180 / np.pi * np.linspace(azimuth[0, 0], azimuth[-1, -1], 9))
    ax.set_label('Azimuth angle $\phi$')
    ax.set_yticks(180 / np.pi * np.linspace(polar[0, 0], polar[-1, -1], 5))
    ax.set_label('Polar angle $\\theta$')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    # ax.invert_xaxis() # To look more like Matthew
    ax.grid()
