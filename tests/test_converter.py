import numpy as np
import numpy.random
import logging
import pytest
import cmath
import os.path
from tests import context  # Test context
from tests.magnetogram import magnetograms

from stellarwinds.magnetogram import converter
from stellarwinds.magnetogram import coefficients as shc
from stellarwinds.magnetogram import zdi_magnetogram
from stellarwinds.magnetogram import pfss_magnetogram
from stellarwinds.magnetogram import geometry
from stellarwinds.magnetogram import plots


def test_conversion(request):
    """Test that the radial field matches"""
    zg = geometry.ZdiGeometry()
    polar_centers, azimuth_centers = zg.centers()

    coeffs_zdi = shc.Coefficients()
    # coeffs_zdi.append(1, 0, 1.0+1j)
    # coeffs_zdi.append(11, 9, 4)
    # coeffs_zdi.append(1, 1, 1.0)
    coeffs_zdi.append(1, 1, 4 - 2j)

    Br_zdi = zdi_magnetogram.from_coefficients(coeffs_zdi).get_radial_field(polar_centers, azimuth_centers)
    Br_zdi = np.flipud(Br_zdi)  # In the plot, this flips left and right. It might have to do with how solar longitude is defined.
    coeffs_pfss = coeffs_zdi.scale(converter.forward_conversion_factor)
    Br_pfss, *_ = pfss_magnetogram.evaluate_spherical(coeffs_pfss, 1, polar_centers, azimuth_centers)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        _, axs = plt.subplots(3, 1, figsize=(8, 14))
        for f, ax in zip([Br_zdi, Br_pfss, np.abs(Br_zdi-Br_pfss)], axs):
            plots.plot_magnetic_field(ax, polar_centers, azimuth_centers, f, legend_str='B_r', )
            plots.add_extrema(polar_centers, azimuth_centers, f, ax, legend_str='B_r', markers='12')
            ax.legend()
        axs[0].set_title("ZDI")
        axs[1].set_title("PFSS")
        axs[2].set_title("Error")
        plt.savefig(pn.get())

    assert np.allclose(Br_zdi, Br_pfss)


def test_beta_values(request):
    """Test the $beta$ coefficients of the ZDI magnetogram."""
    zg = geometry.ZdiGeometry(256)
    polar_centers, azimuth_centers = zg.centers()

    coeffs_pfss = shc.Coefficients()
    coeffs_pfss.append(3, 2, 1.0 + 0.3j)

    coeffs_zdi = converter.convert_pfss_to_zdi(coeffs_pfss)

    Br_pfss, Bp_pfss, Ba_pfss = pfss_magnetogram.evaluate_spherical(coeffs_pfss, 1, polar_centers, azimuth_centers)

    Br_zdi = zdi_magnetogram.from_coefficients(coeffs_zdi).get_radial_field(polar_centers, azimuth_centers)
    Bp_zdi = zdi_magnetogram.from_coefficients(coeffs_zdi).get_polar_poloidal_field_new(polar_centers, azimuth_centers)
    Ba_zdi = zdi_magnetogram.from_coefficients(coeffs_zdi).get_azimuthal_field(polar_centers, azimuth_centers)

    # In the plot, this flips left and right. It might have to do with how solar longitude is defined.
    Br_zdi = np.flipud(Br_zdi)
    Bp_zdi = np.flipud(Bp_zdi)
    Ba_zdi = np.flipud(Ba_zdi)
    Ba_zdi *= -1  # This is also because of the left-right flip

    b = 1  # Bad border pixels

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        _, axs = plt.subplots(3, 2, figsize=(12, 12))
        for f, ax in zip([Br_zdi, Br_pfss, np.abs(Br_zdi-Br_pfss)], axs[:, 0]):
            plots.plot_magnetic_field(ax, polar_centers, azimuth_centers, f, legend_str='B_r', )
            plots.add_extrema(polar_centers, azimuth_centers, f, ax, legend_str='B_r', markers='12')
            ax.legend()
        axs[0, 0].set_title("ZDI radial")
        axs[1, 0].set_title("PFSS radial")
        axs[2, 0].set_title("Error radial")

        for f, ax in zip([Bp_zdi, Bp_pfss, np.abs(Bp_zdi-Bp_pfss)], axs[:, 1]):
            plots.plot_magnetic_field(ax,
                                      polar_centers[b:-b, b:-b],
                                      azimuth_centers[b:-b, b:-b],
                                      f[b:-b, b:-b],
                                      legend_str='B_p', )
            plots.add_extrema(polar_centers[b:-b, b:-b],
                              azimuth_centers[b:-b, b:-b],
                              f[b:-b, b:-b],
                              ax, legend_str='B_p', markers='12')
            ax.legend()
        axs[0, 1].set_title("ZDI polar")
        axs[1, 1].set_title("PFSS polar")
        axs[2, 1].set_title("Error polar")
        plt.savefig(pn.get())

        # Scatter curve of errors
        _, axs = plt.subplots(1, 3, figsize=(12, 4))

        for (fz, fp), ax in zip([(Br_zdi, Br_pfss), (Bp_zdi, Bp_pfss), (Ba_zdi, Ba_pfss)], axs):
            ax.plot([np.min(fz), np.max(fz)], [np.min(fz), np.max(fz)])
            ax.plot(fz.ravel(), fp.ravel(), ',')

        plt.savefig(pn.get())

        # Residual curve of errors
        _, axs = plt.subplots(1, 3, figsize=(12, 4))

        for (fz, fp), ax in zip([(Br_zdi, Br_pfss), (Bp_zdi, Bp_pfss), (Ba_zdi, Ba_pfss)], axs):
            # ax.plot([np.min(fz), np.max(fz)], [np.min(fz), np.max(fz)])
            ax.plot(fz.ravel(), fz.ravel() - fp.ravel(), ',')

        plt.savefig(pn.get())

        # Residual curve of errors
        _, axs = plt.subplots(1, 3, figsize=(12, 4))

        for (fz, fp), ax in zip([(Br_zdi, Br_pfss), (Bp_zdi, Bp_pfss), (Ba_zdi, Ba_pfss)], axs):
            # ax.plot([np.min(fz), np.max(fz)], [np.min(fz), np.max(fz)])
            ax.plot(fz.ravel(), np.abs(1 - fz.ravel() / fp.ravel()), ',')
            ax.set_yscale("log")

        plt.savefig(pn.get())

    assert np.allclose(Br_zdi, Br_pfss)

    assert np.allclose(Bp_zdi[b:-b, b:-b], Bp_pfss[b:-b, b:-b])


def test_beta_loop(request, magnetogram_name="mengel"):

    zg = geometry.ZdiGeometry(128)

    centers = zg.centers()

    coeffs_zdi0 = magnetograms.get_all(magnetogram_name)
    zm0 = zdi_magnetogram.from_coefficients(coeffs_zdi0)
    Br_zdi0 = zm0.get_radial_field(*centers)
    Bp_zdi0 = zm0.get_polar_field(*centers)
    Ba_zdi0 = zm0.get_azimuthal_field(*centers)

    # In the plot, this flips left and right. It might have to do with how solar longitude is defined.
    Br_zdi0= np.flipud(Br_zdi0)
    Bp_zdi0 = np.flipud(Bp_zdi0)
    Ba_zdi0 = np.flipud(Ba_zdi0)
    Ba_zdi0 *= -1  # This is also because of the left-right flip


    Br_zdi0p = zm0.get_radial_poloidal_field(*centers)
    Bp_zdi0p = zm0.get_polar_poloidal_field(*centers)
    Ba_zdi0p = zm0.get_azimuthal_poloidal_field(*centers)

    # In the plot, this flips left and right. It might have to do with how solar longitude is defined.
    Br_zdi0p= np.flipud(Br_zdi0p)
    Bp_zdi0p = np.flipud(Bp_zdi0p)
    Ba_zdi0p = np.flipud(Ba_zdi0p)
    Ba_zdi0p *= -1  # This is also because of the left-right flip


    alpha0, *_ = shc.hsplit(coeffs_zdi0)

    coeffs_pfss = converter.convert_zdi_to_pfss(alpha0)
    Br_pfss, Bp_pfss, Ba_pfss = pfss_magnetogram.evaluate_spherical(coeffs_pfss, 1, *centers)

    coeffs_zdi = converter.convert_pfss_to_zdi(coeffs_pfss)
    zm = zdi_magnetogram.from_coefficients(coeffs_zdi)
    Br_zdi = zm.get_radial_field(*centers)
    # Bp_zdi = zm.get_polar_poloidal_field_new(*centers)
    Bp_zdi = zm.get_polar_field(*centers)
    Ba_zdi = zm.get_azimuthal_field(*centers)

    # In the plot, this flips left and right. It might have to do with how solar longitude is defined.
    Br_zdi = np.flipud(Br_zdi)
    Bp_zdi = np.flipud(Bp_zdi)
    Ba_zdi = np.flipud(Ba_zdi)
    Ba_zdi *= -1  # This is also because of the left-right flip

    def plot_row(fields, axs, direction):

        for f, ax in zip(fields, axs):
            plots.plot_magnetic_field(ax, *centers, f, legend_str='B_r', )
            plots.add_extrema(*centers, f, ax, legend_str='B_r', markers='12')
            ax.legend()

        axs[0].set_title(f"ZDI full {direction}")
        axs[1].set_title(f"ZDI poloidal {direction}")
        axs[2].set_title(f"PFSS {direction}")
        axs[3].set_title(f"ZDI full (from PFSS) {direction}")
        axs[4].set_title(f"Error {direction}")


    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        _, axs = plt.subplots(5, 3, figsize=(18, 18))
        plot_row([Br_zdi0, Br_zdi0p, Br_pfss, Br_zdi, np.abs(Br_zdi-Br_pfss)], axs[:, 0], direction="radial")
        plot_row([Bp_zdi0, Bp_zdi0p, Bp_pfss, Bp_zdi, np.abs(Bp_zdi-Bp_pfss)], axs[:, 1], direction="polar")
        plot_row([Ba_zdi0, Ba_zdi0p, Ba_pfss, Ba_zdi, np.abs(Ba_zdi-Ba_pfss)], axs[:, 2], direction="azimuthal")

        plt.savefig(pn.get())



def test_forward_conversion_factor():
    assert(np.isclose(converter.forward_conversion_factor(0, 0),  (4.0 * np.pi)**(-0.5)))
    assert(np.isclose(converter.forward_conversion_factor(0, 1), -(8.0 * np.pi)**(-0.5)))
    assert(np.isclose(converter.forward_conversion_factor(0, 2),  (8.0 * np.pi)**(-0.5)))
    assert(np.isclose(converter.forward_conversion_factor(0, 3), -(8.0 * np.pi)**(-0.5)))

    assert(np.isclose(converter.forward_conversion_factor(0, 98),  (8.0 * np.pi)**(-0.5)))
    assert(np.isclose(converter.forward_conversion_factor(0, 99), -(8.0 * np.pi)**(-0.5)))

    order_m = np.arange(0, 4)
    degree_l = np.zeros_like(order_m)
    expected = np.array([(4.0 * np.pi)**(-0.5),
                        -(8.0 * np.pi)**(-0.5),
                        (8.0 * np.pi)**(-0.5),
                        -(8.0 * np.pi)**(-0.5)])
    assert(np.allclose(converter.forward_conversion_factor(degree_l, order_m), expected))


@pytest.mark.parametrize("l", (0, 1, 2, 3, 10, 20))
@pytest.mark.parametrize("m", (0, 1, 2, 3, 10, 21))
def test_independent_implementation(l, m):
    """ Test if I still understand the implementation a few months later."""
    expected_result = ((2*l+1)**.5) * (1/np.sqrt(4*np.pi)) * ((-1)**m / np.sqrt(2-int(m==0)))

    cm_result = converter.forward_conversion_factor(l, m)

    assert(np.isclose(cm_result, expected_result))


def test_map_to_positive_orders(request):
    """
    Test that the mapping to positive orders for rotated ZDI magnetograms work
    by comparing the computed values at a set of points.
    """
    degree_l_max = 5

    for degree_l in range(degree_l_max+1):
        for order_m in range(-degree_l, degree_l+1):
            original = shc.Coefficients()
            original.append(degree_l, order_m, 1)
            compare_values(request, original, converter.map_to_positive_orders(original))

    for degree_l in range(degree_l_max+1):
        for order_m in range(-degree_l, degree_l+1):
            original = shc.Coefficients()
            original.append(degree_l, order_m, 1+1j)
            compare_values(request, original, converter.map_to_positive_orders(original))

    for degree_l in range(degree_l_max + 1):
        for order_m in range(-degree_l, degree_l + 1):
            original = shc.Coefficients()
            original.append(degree_l, order_m, 3-2j)
            compare_values(request, original, converter.map_to_positive_orders(original))

    original = shc.Coefficients()
    for degree_l in range(degree_l_max + 1):
        for order_m in range(-degree_l, degree_l + 1):
            original.append(degree_l, order_m, 3-2j)
            compare_values(request, original, converter.map_to_positive_orders(original))

    # If this passes it should be correct.
    original = shc.Coefficients()
    for degree_l in range(degree_l_max + 1):
        for order_m in range(-degree_l, degree_l + 1):
            coeff = np.random.uniform(low=-10, high=10, size=2)
            coeff = coeff[0] + 1j * coeff[1]
            original.append(degree_l, order_m, coeff)
            compare_values(request, original, converter.map_to_positive_orders(original))

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        fig, axs = plt.subplots(1, 2)
        zg = geometry.ZdiGeometry()

        for magnetogram, ax in zip((original, converter.map_to_positive_orders(original)), axs):
            degrees, orders, coefficients = magnetogram.as_arrays()
            lz = zdi_magnetogram.ZdiMagnetogram(degrees, orders, coefficients)
            fv = lz.get_radial_field(*zg.centers())
            img1 = ax.pcolormesh(*zg.corners()[::-1], np.real(fv))
            fig.colorbar(img1, ax=ax, orientation='horizontal')

        plt.savefig(pn.get())


def compare_values(request, m0, m1):
    zg = geometry.ZdiGeometry()

    field_values = get_field_values(zg, m0, m1)

    try:
        assert np.allclose(np.imag(field_values[0]), 0), "No imaginary component expected."
        assert np.allclose(np.imag(field_values[1]), 0), "No imaginary component expected."
        assert np.allclose(np.abs(field_values[0]), np.abs(field_values[1])), "Field absolute values must match."
        assert np.allclose(field_values[0], field_values[1]), "Field values must match; i.e. sign must be correct."
    except AssertionError as e:
        with context.PlotNamer(__file__, request.node.name) as (pn, plt):

            fig, axs = plt.subplots(1, 2)

            for fv, ax in zip(field_values, axs):

                img1 = ax.pcolormesh(*zg.corners()[::-1], np.real(fv))
                fig.colorbar(img1, ax=ax, orientation='horizontal')

            plt.savefig(pn.get())
            raise


def get_field_values(zg, *magnetograms):
    """
    Get ZDI field values on the geometry zg for the supplied magnetograms.
    """
    field_values = []
    for magnetogram in magnetograms:
        degrees, orders, coefficients = magnetogram.as_arrays()
        lz = zdi_magnetogram.ZdiMagnetogram(degrees, orders, coefficients)
        field_values.append(lz.get_radial_field(*zg.centers()))

    return field_values


def test_collect_cosines():
    r, alpha = np.sqrt(2.0),  np.pi/4
    s, beta =  np.sqrt(2.0), -np.pi/4
    t, gamma = converter.collect_cosines(r, alpha, s, beta)

    m = 3
    theta = np.linspace(0, 2*np.pi, 100)
    fc = np.real(cmath.rect(r, alpha) * np.exp(1j * m * theta) + cmath.rect(s, beta) * np.exp(-1j * m * theta))
    frs = r * np.cos(alpha + m * theta) + s * np.cos(beta - m * theta)
    ft  = t * np.cos(m * theta + gamma)
#     log.debug('plot_comp', (r, alpha), (s, beta), (t, gamma))
#     plt.plot(theta, fc)
#     plt.plot(theta, frs,'-', linewidth=3)
#     plt.plot(theta, ft, '--', linewidth=3)
    assert np.allclose(fc, frs)
    assert np.allclose(fc, ft)


def test_latlon_conversion(request):
    """Test that the discretized field can be reconstructed as zdi components."""
    zg = geometry.ZdiGeometry()
    polar_centers, azimuth_centers = zg.centers()

    field_r = np.ones_like(polar_centers)
    field_polar = np.zeros_like(field_r)
    field_azimuthal = np.zeros_like(field_r)

    zm = converter.convert_latlon_to_zdi(polar_centers, azimuth_centers,
                                          field_r, field_polar, field_azimuthal)

    assert np.allclose(zm.alpha[1:], 0)
    assert np.allclose(zm.beta, 0)
    assert np.allclose(zm.gamma, 0)

@pytest.mark.parametrize("coeff_name", ("low", "m0", "m01", "m10", "random"))
def test_latlon_alpha_conversion(coeff_name, request):
    """Test that the discretized field can be reconstructed as zdi components."""
    zg = geometry.ZdiGeometry(64)
    polar_centers, azimuth_centers = zg.centers()

    if coeff_name == "low":
        coeffs0_alpha = shc.Coefficients()
        # coeffs0_alpha.append(0, 0, 1.1)  # Does not work with 0, 0 yet.
        coeffs0_alpha.append(1, 1, 0.1 + 0.1j)
        coeffs0_alpha.append(3, 2, 0.1)
        coeffs0_alpha.append(3, 3, 0.1)
        coeffs0_alpha.append(2, 2, 0.1)
    elif coeff_name == "m0":
        coeffs0_alpha = shc.Coefficients()
        # coeffs0_alpha.append(0, 0, 1.1)  # Does not work with 0, 0 yet.
        coeffs0_alpha.append(1, 0, 0.1)
        coeffs0_alpha.append(2, 0, 0.13)
        coeffs0_alpha.append(3, 3, 0.0)

    elif coeff_name == "m01":
        coeffs0_alpha = shc.Coefficients()
        # coeffs0_alpha.append(0, 0, 1.1)  # Does not work with 0, 0 yet.
        coeffs0_alpha.append(1, 0, 0.1 + .3j)
        coeffs0_alpha.append(1, 1, -0.1 - .1j)
        # coeffs0_alpha.append(2, 0, 0.13)
        coeffs0_alpha.append(3, 3, 0.0)
    elif coeff_name == "m10":
        coeffs0_alpha = shc.Coefficients()
        coeffs0_alpha.append(1, 0, -0.3 - 0.2j)
        coeffs0_alpha.append(3, 3, 0.0)
    else:
        coeffs0_alpha = shc.noise(degree_max=5, beta=1) * 1
        coeffs0_alpha.set(0, 0, 0.0)
        # print(coeffs0_alpha)
        # coeffs0_alpha = shc.Coefficients()
        # coeffs0_alpha.append(1, 0, -0.26914045-0.18614253j)
        # coeffs0_alpha.append(1, 1, 0.45666501+0.61277559j)
        # coeffs0_alpha.append(1, 0, -0.3 - 0.2j)  # Sign of imaginary part is reconstructed wrongly.
        # coeffs0_alpha.append(1, 0, 0.5 + 0.6j)

    # The imaginary component is undetermined when order_m is zero. This means that it is impossible to fully
    # recover the input coefficients, even when the magnetogram is fully reconstructed. Work around this here
    # by setting it to zero
    for (degree, order), val in coeffs0_alpha.contents():
        if order == 0:
            coeffs0_alpha.set(degree, order, np.real(val))

    zm0 = zdi_magnetogram.from_coefficients(coeffs0_alpha)
    field_r = zm0.get_radial_field(polar_centers, azimuth_centers)
    field_polar = zm0.get_polar_field(polar_centers, azimuth_centers)
    field_azimuthal = zm0.get_azimuthal_field(polar_centers, azimuth_centers)

    zm1 = converter.convert_latlon_to_zdi(polar_centers, azimuth_centers,
                                          field_r, field_polar, field_azimuthal,
                                          max_degree=5)

    coeffs1_alpha = shc.from_arrays(zm1.degrees_l, zm1.orders_m, zm1.alpha)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        from stellarwinds.magnetogram import plot_zdi

        fig, axs = plt.subplots(1, 2)
        ax = axs[0]
        plot_zdi.plot_energy_by_degree(zm0, ax)
        plt.draw()
        ylim = np.asarray(ax.get_ylim()) * np.array([.1, 10])
        ax.set_ylim(ylim)

        ax = axs[1]
        plot_zdi.plot_energy_by_degree(zm1, ax)
        ax.set_ylim(ylim)
        plt.savefig(pn.get())
        plt.close()

        fig, axs = plt.subplots(2, 3, figsize=(6*3, 2*3))
        plot_zdi.plot_zdi_components(zm0, zg=zg, axs=axs[0])
        plot_zdi.plot_zdi_components(zm1, zg=zg, axs=axs[1])
        plt.savefig(pn.get())

    print(coeffs0_alpha)
    print(coeffs1_alpha)
    print(coeffs1_alpha - coeffs0_alpha)
    print(shc.isclose(coeffs1_alpha, coeffs0_alpha, rtol=1e-2, atol=1e-2))
    assert shc.allclose(coeffs1_alpha, coeffs0_alpha, rtol=1e-2, atol=1e-2)


def test_latlon_beta_conversion(request):
    """Test that the discretized field can be reconstructed as zdi components."""
    zg = geometry.ZdiGeometry(64)
    polar_centers, azimuth_centers = zg.centers()

    coeffs0_beta = shc.Coefficients()
    # coeffs0_alpha.append(0, 0, 1.1)  # Does not work with 0, 0 yet.
    coeffs0_beta.append(1, 1, 0.1)
    coeffs0_beta.append(2, 2, 0.5)
    coeffs0_beta.append(3, 3, 0.5)

    coeffs0_beta = shc.Coefficients()
    # coeffs0_beta.append(0, 0, 0.1)
    coeffs0_beta.append(1, 0, 1.0)
    # coeffs0_beta.append(2, 2, 1.0)  # Fails
    # coeffs0_beta.append(3, 2, 0.1)
    # coeffs0_alpha.append(2, 2, 0.1)
    coeffs0_beta.append(3, 3, 0.0)

    coeffs0_alpha = shc.zeros_like(coeffs0_beta)
    
    coeffs0 = shc.hstack((coeffs0_alpha, coeffs0_beta))

    zm0 = zdi_magnetogram.from_coefficients(coeffs0)
    field_r = zm0.get_radial_field(polar_centers, azimuth_centers)
    field_polar = zm0.get_polar_field(polar_centers, azimuth_centers)
    field_azimuthal = zm0.get_azimuthal_field(polar_centers, azimuth_centers)

    # import pdb; pdb.set_trace()

    zm1 = converter.convert_latlon_to_zdi(polar_centers, azimuth_centers,
                                          field_r, field_polar, field_azimuthal)

    coeffs1_beta = shc.from_arrays(zm1.degrees_l, zm1.orders_m, zm1.beta)
    zm1.gamma = np.zeros_like(zm1.beta)  # Drop any toroidal component


    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        from stellarwinds.magnetogram import plot_zdi

        fig, axs = plt.subplots(1, 2)
        ax = axs[0]
        plot_zdi.plot_energy_by_degree(zm0, ax)
        plt.draw()
        ylim = np.asarray(ax.get_ylim()) * np.array([.1, 10])
        ax.set_ylim(ylim)

        ax = axs[1]
        plot_zdi.plot_energy_by_degree(zm1, ax)
        ax.set_ylim(ylim)
        plt.savefig(pn.get())
        plt.close()

        fig, axs = plt.subplots(2, 3, figsize=(6*3, 2*3))
        plot_zdi.plot_zdi_components(zm0, zg=zg, axs=axs[0])
        plot_zdi.plot_zdi_components(zm1, zg=zg, axs=axs[1])
        plt.savefig(pn.get())

    print(coeffs1_beta - coeffs0_beta)
    assert shc.allclose(coeffs1_beta, coeffs0_beta, rtol=1e-3, atol=1e-3)
