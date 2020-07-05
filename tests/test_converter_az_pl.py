import logging
log = logging.getLogger(__name__)

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
    """Test that the discretized field can be reconstructed as zdi components."""
    zg = geometry.ZdiGeometry()
    polar_centers, azimuth_centers = zg.centers()

    field_r = np.ones_like(polar_centers)
    field_polar = np.zeros_like(field_r)
    field_azimuthal = np.zeros_like(field_r)

    zm = converter.convert_latlon_to_zdi(polar_centers, azimuth_centers,
                                          field_r, field_polar, field_azimuthal)

    assert np.allclose(zm.alpha[1:], 0, rtol=1e-2, atol=1e-2)
    assert np.allclose(zm.beta, 0)
    assert np.allclose(zm.gamma, 0)


@pytest.mark.parametrize("coeff_name", ("low", "m0", "m01", "m10", "random"))
def test_alpha_conversion(coeff_name, request):
    """Test that the discretized radial field can be reconstructed as zdi components."""
    zg = geometry.ZdiGeometry(64)
    polar_centers, azimuth_centers = zg.centers()

    coeffs0_alpha = make_coeffs(coeff_name)

    zm0 = zdi_magnetogram.from_coefficients(coeffs0_alpha)
    field_r = zm0.get_radial_field(polar_centers, azimuth_centers)
    field_polar = zm0.get_polar_field(polar_centers, azimuth_centers)
    field_azimuthal = zm0.get_azimuthal_field(polar_centers, azimuth_centers)

    zm1 = converter.convert_latlon_to_zdi(polar_centers, azimuth_centers,
                                          field_r, field_polar, field_azimuthal,
                                          max_degree=coeffs0_alpha.degree_max)

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


def make_coeffs(coeff_name):
    if coeff_name == "low":
        coeffs = shc.Coefficients()
        coeffs.append(0, 0, 0.1 + .1j)  # Does not work with 0, 0 yet.
        coeffs.append(1, 1, 0.1 + 0.1j)
        coeffs.append(3, 2, 0.1)
        coeffs.append(3, 3, 0.1)
        coeffs.append(2, 2, 0.1)
    elif coeff_name == "m0":
        coeffs = shc.Coefficients()
        coeffs.append(0, 0, 0.1)  # Does not work with 0, 0 yet.
        coeffs.append(1, 0, 0.1)
        coeffs.append(2, 0, 0.13)
        coeffs.append(3, 3, 0.0)
    elif coeff_name == "m01":
        coeffs = shc.Coefficients()
        coeffs.append(0, 0, 0.1)  # Does not work with 0, 0 yet.
        coeffs.append(1, 0, 0.1 + .3j)
        coeffs.append(1, 1, -0.1 - .1j)
        # coeffs0_alpha.append(2, 0, 0.13)
        coeffs.append(3, 3, 0.0)
    elif coeff_name == "m10":
        coeffs = shc.Coefficients()
        coeffs.append(1, 0, -0.3 - 0.2j)
        coeffs.append(3, 3, 0.0)
    else:
        coeffs = shc.noise(degree_max=5, beta=1) * 1
        # coeffs.set(0, 0, 0.0)

    remove_m0_imaginary(coeffs)

    return coeffs


def remove_m0_imaginary(coeffs):
    # The imaginary component is undetermined when order_m is zero. This means that it is impossible to fully
    # recover the input coefficients, even when the magnetogram is fully reconstructed. Work around this here
    # by setting it to zero
    for (degree, order), val in coeffs.contents():
        if order == 0:
            coeffs.set(degree, order, np.real(val))


@pytest.mark.parametrize("coeff_name", ("low", "m0", "m01", "m10", "random"))
# @pytest.mark.parametrize("coeff_name", (None,))
def test_beta_conversion(coeff_name, request):
    """Test that the discretized field can be reconstructed as zdi components."""
    zg = geometry.ZdiGeometry(64)
    polar_centers, azimuth_centers = zg.centers()

    coeffs0_beta = make_coeffs(coeff_name)

    remove_m0_imaginary(coeffs0_beta)
    coeffs0_beta.set(0, 0, 0.0)  # This does not affect the field, so it cannot be reconstructed.

    coeffs0 = shc.hstack((shc.zeros_like(coeffs0_beta),
                          coeffs0_beta))

    zm0 = zdi_magnetogram.from_coefficients(coeffs0)
    field_r = zm0.get_radial_field(polar_centers, azimuth_centers)
    field_polar = zm0.get_polar_field(polar_centers, azimuth_centers)
    field_azimuthal = zm0.get_azimuthal_field(polar_centers, azimuth_centers)

    # import pdb; pdb.set_trace()

    zm1 = converter.convert_latlon_to_zdi(polar_centers, azimuth_centers,
                                          field_r, field_polar, field_azimuthal,
                                          max_degree=coeffs0.degree_max)

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

    with np.printoptions(precision=3, suppress=True):
        print(coeffs0_beta)
        print(coeffs1_beta)
        print(coeffs1_beta - coeffs0_beta)

    print(shc.isclose(coeffs1_beta, coeffs0_beta, rtol=1e-2, atol=1e-2))
    assert shc.allclose(coeffs1_beta, coeffs0_beta, rtol=1e-2, atol=1e-2)


def l_and_m(max_l):
    for l in range(1, max_l + 1):
        for m in range(0, l+1):
            yield l, m


@pytest.mark.parametrize("degree_l, order_m", l_and_m(5))
def test_beta_single_coeff(degree_l, order_m, request):
    """Test conversion of single coefficient"""

    zg = geometry.ZdiGeometry(64)
    polar_centers, azimuth_centers = zg.centers()

    coeffs0_beta = shc.Coefficients()
    coeffs0_beta.append(degree_l, order_m, 1.0)

    coeffs0 = shc.hstack((shc.zeros_like(coeffs0_beta),
                          coeffs0_beta,
                          shc.zeros_like(coeffs0_beta),))

    remove_m0_imaginary(coeffs0)

    zm0 = zdi_magnetogram.from_coefficients(coeffs0)
    field_r0 = zm0.get_radial_field(polar_centers, azimuth_centers)
    field_polar0 = zm0.get_polar_field(polar_centers, azimuth_centers)
    field_azimuthal0 = zm0.get_azimuthal_field(polar_centers, azimuth_centers)

    coeffs1 = converter.get_zdi_coeffs(polar_centers, azimuth_centers,
                                       degree_l, order_m,
                                       field_r0, field_polar0, field_azimuthal0)
    assert coeffs1.size == coeffs0.size

    _, coeffs1_beta, _ = shc.hsplit(coeffs1)

    coeffs1 = shc.hstack((shc.zeros_like(coeffs1_beta),
                          coeffs1_beta,
                          shc.zeros_like(coeffs1_beta),))

    zm1 = zdi_magnetogram.from_coefficients(coeffs1)
    field_r1 = zm1.get_radial_field(polar_centers, azimuth_centers)
    field_polar1 = zm1.get_polar_field(polar_centers, azimuth_centers)
    field_azimuthal1 = zm1.get_azimuthal_field(polar_centers, azimuth_centers)

    coeffs_delta_beta = coeffs1_beta / coeffs0_beta
    coeffs_delta_beta = shc.hstack((shc.zeros_like(coeffs_delta_beta),
                               coeffs_delta_beta,
                               shc.zeros_like(coeffs_delta_beta)))
    zm_delta = zdi_magnetogram.from_coefficients(coeffs_delta_beta)

    if not np.allclose(field_r0, field_r1):
        field_quotient_polar = field_r0 / field_r1
    else:
        field_quotient_r = np.zeros_like(field_r0)

    if not np.allclose(field_polar0, field_polar1):
        field_quotient_polar = field_polar1 / field_polar0
    else:
        field_quotient_polar = np.zeros_like(field_polar1)

    if not np.allclose(field_azimuthal0, field_azimuthal1):
        field_quotient_azimuthal = field_azimuthal1 / field_azimuthal0
    else:
        field_quotient_azimuthal = np.zeros_like(field_azimuthal1)

    # with context.PlotNamer(__file__, request.node.name) as (pn, plt):
    #     from stellarwinds.magnetogram import plot_zdi
    #
    #     fig, axs = plt.subplots(3, 3, figsize=(6 * 3, 3 * 3))
    #     plot_zdi.plot_zdi_components(zm0, zg=zg, axs=axs[0])
    #     plot_zdi.plot_zdi_components(zm1, zg=zg, axs=axs[1])
    #
    #     for pid, f in enumerate((field_quotient_r, field_quotient_polar, field_quotient_azimuthal)):
    #         img = axs[2, pid].pcolormesh(np.rad2deg(azimuth_centers),
    #                                      np.rad2deg(polar_centers), f)
    #
    #         fig.colorbar(img, ax=axs[2, pid], orientation="horizontal", pad=0.2)
    #
    #         axs[2, pid].xaxis.set_ticks(np.arange(0, 361, 45))
    #         axs[2, pid].yaxis.set_ticks(np.arange(0, 181, 30))
    #         axs[2, pid].grid()
    #         axs[2, pid].invert_yaxis()
    #         axs[2, pid].set_aspect('equal')
    #         axs[2, pid].set_xlabel("Azimuth angle [deg]")
    #         axs[2, pid].set_ylabel("Polar angle [deg]")
    #
    #     plt.savefig(pn.get())

    # The values are close, except for a small number of points where cancellation effects dominate.
    assert almost_all(field_quotient_r) >= .95
    assert almost_all(field_quotient_polar) >= .95
    assert almost_all(field_quotient_azimuthal) >= .95

    # log.error(f"{degree_l}, {order_m}, {np.median(field_quotient_polar)}")

def almost_all(array):
    flat = array.flatten()
    flat = flat[np.logical_not(np.isnan(flat))]
    return np.sum(np.isclose(array, np.median(flat))) / array.size


@pytest.mark.skip(reason="Based on false assumption of constant scaling in bad implementation.")
def test_beta_scale(request):

    zg = geometry.ZdiGeometry(64)
    polar_centers, azimuth_centers = zg.centers()

    for l in range(1, 4):
        for m in range(0, l+1):

            coeffs0_beta = shc.Coefficients()
            coeffs0_beta.append(l, m, 1.0)

            coeffs0 = shc.hstack((shc.zeros_like(coeffs0_beta), coeffs0_beta))

            zm0 = zdi_magnetogram.from_coefficients(coeffs0)
            field_r0 = zm0.get_radial_field(polar_centers, azimuth_centers)
            field_polar0 = zm0.get_polar_field(polar_centers, azimuth_centers)
            field_azimuthal0 = zm0.get_azimuthal_field(polar_centers, azimuth_centers)

            coeffs1 = converter.get_zdi_coeffs(polar_centers, azimuth_centers,
                                               l, m,
                                               field_r0, field_polar0, field_azimuthal0)

            zm1 = zdi_magnetogram.from_coefficients(coeffs1)
            field_r1 = zm1.get_radial_field(polar_centers, azimuth_centers)
            field_polar1 = zm1.get_polar_field(polar_centers, azimuth_centers)
            field_azimuthal1 = zm1.get_azimuthal_field(polar_centers, azimuth_centers)

            q_pl = field_polar0 - field_polar1
            q_az = field_azimuthal0 - field_azimuthal1

            mq_pl = np.mean(q_pl)
            mq_az = np.mean(q_az)

            with np.printoptions(precision=3, sign=' '):
                print(l, m, np.array([np.min(q_pl), np.max(q_pl), mq_pl, np.min(q_az), np.max(q_az), mq_az]))

            assert np.allclose(q_pl, mq_pl), f"Failed pl for {(l, m)}."
            assert np.allclose(q_az, mq_az), f"Failed az for {(l, m)}."


def test_beta_00(request):
    """Verify that beta_00 does not affect field"""
    zg = geometry.ZdiGeometry(64)
    polar_centers, azimuth_centers = zg.centers()

    coeffs_beta = shc.Coefficients()
    coeffs_beta.append(0, 0, +13 - .431j)

    coeffs = shc.hstack((shc.zeros_like(coeffs_beta),
                          coeffs_beta,
                          shc.zeros_like(coeffs_beta)))

    zm = zdi_magnetogram.from_coefficients(coeffs)
    field = zm.get_radial_field(polar_centers, azimuth_centers)

    assert np.allclose(field, 0)


def test_gamma_00(request):
    """Verify that beta_00 does not affect field"""
    zg = geometry.ZdiGeometry(64)
    polar_centers, azimuth_centers = zg.centers()

    coeffs_gamma = shc.Coefficients()
    coeffs_gamma.append(0, 0, +13 - .431j)

    coeffs = shc.hstack((shc.zeros_like(coeffs_gamma),
                          shc.zeros_like(coeffs_gamma),
                          coeffs_gamma))

    zm = zdi_magnetogram.from_coefficients(coeffs)
    field = zm.get_radial_field(polar_centers, azimuth_centers)

    assert np.allclose(field, 0)