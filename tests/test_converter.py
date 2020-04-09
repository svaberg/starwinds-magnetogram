import numpy as np
import numpy.random
import logging
import pytest
import cmath
from tests import context  # Test context

import stellarwinds.magnetogram.converter as cm
from stellarwinds.magnetogram import coefficients as shc
from stellarwinds.magnetogram import zdi_magnetogram
from stellarwinds.magnetogram import pfss_magnetogram
from stellarwinds.magnetogram import geometry

from stellarwinds.magnetogram import plots



def test_conversion():
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
    coeffs_pfss = coeffs_zdi.copy()
    coeffs_pfss.apply_scaling(cm.forward_conversion_factor)
    Br_pfss, *_ = pfss_magnetogram.evaluate_spherical(coeffs_pfss, 1, polar_centers, azimuth_centers)


    import matplotlib.pyplot as plt
    _, axs = plt.subplots(3, 1, figsize=(8, 14))
    for f, ax in zip([Br_zdi, Br_pfss, np.abs(Br_zdi-Br_pfss)], axs):
        plots.plot_magnetic_field(ax, polar_centers, azimuth_centers, f, legend_str='B_r', )
        plots.add_extrema(polar_centers, azimuth_centers, f, ax, legend_str='B_r', markers='12')
        ax.legend()

    axs[0].set_title("ZDI")
    axs[1].set_title("PFSS")
    axs[2].set_title("Error")
    plt.show()

    assert np.allclose(Br_zdi, Br_pfss)







def test_forward_conversion_factor():
    assert(np.isclose(cm.forward_conversion_factor(0, 0),  (4.0 * np.pi)**(-0.5)))
    assert(np.isclose(cm.forward_conversion_factor(0, 1), -(8.0 * np.pi)**(-0.5)))
    assert(np.isclose(cm.forward_conversion_factor(0, 2),  (8.0 * np.pi)**(-0.5)))
    assert(np.isclose(cm.forward_conversion_factor(0, 3), -(8.0 * np.pi)**(-0.5)))

    assert(np.isclose(cm.forward_conversion_factor(0, 98),  (8.0 * np.pi)**(-0.5)))
    assert(np.isclose(cm.forward_conversion_factor(0, 99), -(8.0 * np.pi)**(-0.5)))

    order_m = np.arange(0, 4)
    degree_l = np.zeros_like(order_m)
    expected = np.array([(4.0 * np.pi)**(-0.5),
                        -(8.0 * np.pi)**(-0.5),
                        (8.0 * np.pi)**(-0.5),
                        -(8.0 * np.pi)**(-0.5)])
    assert(np.allclose(cm.forward_conversion_factor(degree_l, order_m), expected))


@pytest.mark.parametrize("l", (0, 1, 2, 3, 10, 20))
@pytest.mark.parametrize("m", (0, 1, 2, 3, 10, 21))
def test_independent_implementation(l, m):
    """ Test if I still understand the implementation a few months later."""
    expected_result = ((2*l+1)**.5) * (1/np.sqrt(4*np.pi)) * ((-1)**m / np.sqrt(2-int(m==0)))

    cm_result = cm.forward_conversion_factor(l, m)

    assert(np.isclose(cm_result, expected_result))


def test_read():
    content = r"""General poloidal plus toroidal field
4 3 -3
 1  0 1. 1.
 1  1 1. 1.
 2  0 1. 1.
 2  1 1. 1.
 2  2 1. 1.

 1  0 100. 101.
 1  1 110. 111.
 2  0 200. 201.
 2  1 210. 211.
 2  2 220. 221.

 1  0 1000. 1010.
 1  1 1100. 1110.
 2  0 2000. 2010.
 2  1 2100. 2110.
 2  2 2200. 2210.
 """

    path = context.default_artifact_directory + '/test_field_zdipy.dat'

    with open(path, 'w') as f:
        f.write(content)

    coeffs = cm.read_magnetogram_file(path)

    coeffs.apply_scaling(cm.forward_conversion_factor)
    radial_coeffs, *_ = shc.hsplit(coeffs)

    cm.write_magnetogram_file(radial_coeffs, fname=context.default_artifact_directory + '/test_field_wso.dat')


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
            compare_values(request, original, cm.map_to_positive_orders(original))

    for degree_l in range(degree_l_max+1):
        for order_m in range(-degree_l, degree_l+1):
            original = shc.Coefficients()
            original.append(degree_l, order_m, 1+1j)
            compare_values(request, original, cm.map_to_positive_orders(original))

    for degree_l in range(degree_l_max + 1):
        for order_m in range(-degree_l, degree_l + 1):
            original = shc.Coefficients()
            original.append(degree_l, order_m, 3-2j)
            compare_values(request, original, cm.map_to_positive_orders(original))

    original = shc.Coefficients()
    for degree_l in range(degree_l_max + 1):
        for order_m in range(-degree_l, degree_l + 1):
            original.append(degree_l, order_m, 3-2j)
            compare_values(request, original, cm.map_to_positive_orders(original))

    # If this passes it should be correct.
    original = shc.Coefficients()
    for degree_l in range(degree_l_max + 1):
        for order_m in range(-degree_l, degree_l + 1):
            coeff = np.random.uniform(low=-10, high=10, size=2)
            coeff = coeff[0] + 1j * coeff[1]
            original.append(degree_l, order_m, coeff)
            compare_values(request, original, cm.map_to_positive_orders(original))

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        fig, axs = plt.subplots(1, 2)
        zg = geometry.ZdiGeometry()

        for magnetogram, ax in zip((original, cm.map_to_positive_orders(original)), axs):
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
    t, gamma = cm.collect_cosines(r, alpha, s, beta)

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


def test_read_full_magnetogram(input_file="/Users/u1092841/Documents/PHD/toupies-magnetograms-colin/coeff-TYC6349-0200-1.dat"):
    types = ("radial", "poloidal", "toroidal")
    magnetogram_data = cm.read_magnetogram_file(input_file, types)

    coefficients = []
    for type_id, type_name in enumerate(types):
        degrees, orders, coefficients = magnetogram_data.as_arrays()

    zdi_magnetogram.ZdiMagnetogram(degrees, orders, *coefficients.transpose())


