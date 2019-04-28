import logging

import matplotlib as mpl
import numpy as np
import scipy.constants

log = logging.getLogger(__name__)

# Test "context"
import pytest
from tests import context  # Test context
from tests.magnetogram import magnetograms

# Local
from stellarwinds.magnetogram.geometry import ZdiGeometry
from stellarwinds.magnetogram.parker_solution import ParkerSolution
from stellarwinds.magnetogram import pfss_magnetogram
from stellarwinds.magnetogram import plot_pfss


def b_alfven(u, rho):
    return u * (scipy.constants.mu_0 * rho)**.5


def test_alfven(request):
    """
    Calculates the magnetic field required for the Alfven surface to fall at a given point.
    :param request:
    :return:
    """

    temperatures = (0.5e6, 0.75e6, 1e6, 1.5e6, 2e6, 3e6, 4e6)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        for _id, temperature in enumerate(temperatures):

            p = ParkerSolution(temperature=temperature)

            r = p.stellar_radius * np.geomspace(1, 215)
            c = plt.rcParams['axes.prop_cycle'].by_key()['color'][_id]

            u = p.speed(r)
            rho = p.density(r)

            plt.plot(p.radius_sonic/p.stellar_radius,
                     b_alfven(p.speed_sonic, p.density_sonic),
                     'o',
                     color=c)

            plt.plot(r/p.stellar_radius,
                     b_alfven(u, rho),
                     color=c,
                     label="T = %g K" % temperature)

        plt.xlabel(r'Height over chromosphere [$R_{\star}$]')
        plt.ylabel('Alfven field [T]')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.savefig(pn.get())


@pytest.mark.parametrize("magnetogram_name", ("dipole", "mengel"))
@pytest.mark.parametrize("plot_name", ("B_r", "B", "c_A"))
def test_alfven_slice(request,
                        magnetogram_name,
                        plot_name):

    normal = "x"
    radius_star = 1
    radius_source_surface = 3
    radius_max = 6

    # Points in slice plane xy coordinate system.
    p1 = np.linspace(-1, 1, 102) * radius_max
    p2 = np.linspace(-1, 1, 104) * radius_max
    p1, p2 = np.meshgrid(p1, p2)

    pxyz = pfss_magnetogram.normal_plane(p1, p2, normal)

    radial_coefficients = magnetograms.get_radial(magnetogram_name) * 10

    f_rpa_xyz = pfss_magnetogram.evaluate_on_slice(radial_coefficients, *pxyz,
                                                   radius_source_surface, radius_star)

    # Drop the extra dimension
    pxyz = [np.squeeze(p) for p in pxyz]
    px, py, pz = pxyz
    f_rpa_xyz = [np.squeeze(f) for f in f_rpa_xyz]
    fr, fp, fa, fx, fy, fz = f_rpa_xyz

    p = ParkerSolution()
    radial_distance = fr * 6.95510e8
    velocity = p.speed(radial_distance)
    density = p.density(radial_distance)

    bmag = (fr**2 + fp**2 + fa**2)**.5

    alfven_mach_number = velocity / (1e-4*bmag / np.sqrt(scipy.constants.mu_0 * density))

    # import pdb; pdb.set_trace()

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        fig, ax = plt.subplots(figsize=(9, 6))

        if plot_name == "B_r":
            from matplotlib.colors import SymLogNorm

            bmin = np.min(bmag[np.where(bmag > 0)])
            bmax = np.max(bmag)

            norm = mpl.colors.SymLogNorm(linthresh=100 * bmin,
                                         linscale=1,
                                         vmin=-bmax,
                                         vmax=bmax)

            im = ax.pcolormesh(p1, p2, fr,
                               norm=norm,
                               cmap='RdBu_r'
                               )

            ax.streamplot(p1, p2,
                          fy, fz,
                          color='gray')

            fig.colorbar(im).set_label('Radial field strength')

        elif plot_name == "B":
            norm = mpl.colors.LogNorm()

            im = ax.pcolormesh(p1, p2, bmag,
                               norm=norm,
                               cmap='viridis')
            fig.colorbar(im).set_label('Absolute field strength')

        else:

            norm = mpl.colors.LogNorm(vmin=1e-2,
                                      vmax=1e2)

            im = ax.pcolormesh(p1, p2,
                               alfven_mach_number,
                               norm=norm,
                               cmap='PuOr')

            fig.colorbar(im).set_label('Alfven number')

        # Plot Alfven surface
        r_a = ax.contour(p1, p2, alfven_mach_number, levels=(1,), colors='black')
        r_a.collections[0].set_label("Alfven surface")

        # Plot source surface
        pr = (px**2 + py**2 + pz**2)**.5
        r_ss = ax.contour(p1, p2, pr, levels=[radius_star, radius_source_surface], colors='white')
        r_ss.collections[0].set_label("Source surface")

        ax.set_aspect('equal')
        ax.set_xlabel('Distance $x/r_{\\star}$')
        ax.set_ylabel('Distance $z/r_{\\star}$')

        plt.legend()
        plt.savefig(pn.get())


@pytest.mark.parametrize("magnetogram_name", ("dipole",))
def test_slice(request,
               magnetogram_name):

    radial_coefficients = magnetograms.get_radial(magnetogram_name) * 10

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, ax = plot_pfss.plot_slice(radial_coefficients)
        fig.savefig(pn.get())
        plt.close()


def source_surface_field_maximum(degree_l, order_m, g_lm, h_lm, rs, rss):

    """
    Find the spherical coordinates of the point where the magnetic field strength is maximal
    on the source surface; outside of the source surface
    this will remain the strongest field location.
    :return:
    """
    points_polar, points_azimuth = ZdiGeometry().centers()
    field_radial, field_polar, field_azimuthal = pfss_magnetogram.evaluate_on_sphere(
        degree_l, order_m, g_lm, h_lm,
        points_polar, points_azimuth,
        radius=rss, radius_star=rs, radius_source_surface=rss)

    # On the source surface (and outside) the polar and azimuthal components are zero.
    assert np.allclose(field_polar, 0)
    assert np.allclose(field_azimuthal, 0)

    # Get the indices of the field maximum
    indices = np.unravel_index(np.argmax(field_radial, axis=None), field_radial.shape)

    return points_polar[indices], points_azimuth[indices], field_radial[indices]


def evaluate_along_ray(pr, pp, pa,
                       degree_l, order_m, g_lm, h_lm,
                       rs, rss):
    """
    Calculate field strength along ray
    :param pr:
    :param pp:
    :param pa:
    :param degree_l:
    :param order_m:
    :param g_lm:
    :param h_lm:
    :param rs:
    :param rss:
    :return:
    """
    pr, pp, pa = np.meshgrid(pr, pp, pa)
    field_radial, field_polar, field_azimuthal = pfss_magnetogram.evaluate_in_space(degree_l, order_m, g_lm, h_lm,
                                                                                    pr, pp, pa,
                                                                                    radius_star=rs,
                                                                                    radius_source_surface=rss)
    assert field_radial.shape == pr.shape
    assert field_polar.shape == pr.shape
    assert field_azimuthal.shape == pr.shape

    # assert np.all(field_polar[np.where(pr > rss)] == 0)  # This is NaN now.
    assert np.all(field_azimuthal[np.where(pr > rss)] == 0)

    pr = np.squeeze(pr)

    field_radial = np.squeeze(field_radial)
    field_polar = np.squeeze(field_polar)
    field_azimuthal = np.squeeze(field_azimuthal)

    return field_radial, field_polar, field_azimuthal


def test_max_alfven_radius_by_field_strength(request,
                                             magnetogram_name="dipole"):

    rss = 3
    rs = 1
    rmax = 2e3

    radial_points = np.geomspace(rs, rmax)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, ax = plt.subplots(figsize=(6, 9))

        for magnetogram_scale in (.1, 1, 10):
            magnetogram = magnetograms.get_radial(magnetogram_name)
            magnetogram *= magnetogram_scale
            degree_l, order_m, alpha_lm = magnetogram.as_arrays(include_unset=False)

            polar_max, azimuth_max, bmax = source_surface_field_maximum(degree_l, order_m,
                                                                        np.real(alpha_lm),
                                                                        np.imag(alpha_lm),
                                                                        rs, rss)

            field_radial, field_polar, field_azimuth = evaluate_along_ray(radial_points, polar_max, azimuth_max,
                                                                          degree_l, order_m,
                                                                          np.real(alpha_lm),
                                                                          np.imag(alpha_lm),
                                                                          rs, rss)

            p = ParkerSolution()
            velocity = p.speed(radial_points * p.stellar_radius)
            density = p.density(radial_points * p.stellar_radius)

            alfven_speed = (1e-4*field_radial / np.sqrt(scipy.constants.mu_0 * density))

            ax.plot(radial_points, alfven_speed,
                    label="Alfven speed, B=%4.2g" % bmax)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(radial_points, velocity, 'k',
                label="Parker flow speed")

        ax.grid(True)
        ax.set_title("Alfven surface maximum at intersection")
        plt.legend()
        plt.savefig(pn.get())
        plt.close()


def test_max_alfven_radius_by_density(request,
                                      magnetogram_name="dipole"):

    rss = 3
    rs = 1
    rmax = 2e3

    radial_points = np.geomspace(rs, rmax)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, ax = plt.subplots(figsize=(6, 9))

        p0 = ParkerSolution()

        fig, ax = plt.subplots(figsize=(6, 9))

        for density_scale in (.1, 1, 10):
            magnetogram = magnetograms.get_radial(magnetogram_name)
            degree_l, order_m, alpha_lm = magnetogram.as_arrays(include_unset=False)

            polar_max, azimuth_max, bmax = source_surface_field_maximum(degree_l, order_m,
                                                                        np.real(alpha_lm),
                                                                        np.imag(alpha_lm),
                                                                        rs, rss)

            field_radial, field_polar, field_azimuth = evaluate_along_ray(radial_points, polar_max, azimuth_max,
                                                                          degree_l, order_m,
                                                                          np.real(alpha_lm),
                                                                          np.imag(alpha_lm),
                                                                          rs, rss)

            p = ParkerSolution(base_density=p0.base_density * density_scale)
            velocity = p.speed(radial_points * p.stellar_radius)
            density = p.density(radial_points * p.stellar_radius)

            alfven_speed = (1e-4 * field_radial / np.sqrt(scipy.constants.mu_0 * density))

            ax.plot(radial_points, alfven_speed, label="Alfven speed rho=%4.2g" % p.base_density)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(radial_points, velocity, 'k',
                label="Parker flow speed")

        ax.grid(True)
        plt.legend()
        plt.savefig(pn.get())
        plt.close()


def test_max_alfven_radius_by_temperature(request,
                                          magnetogram_name="dipole"):

    rss = 3
    rs = 1
    rmax = 2e3

    radial_points = np.geomspace(rs, rmax)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        p0 = ParkerSolution()

        fig, ax = plt.subplots(figsize=(6, 9))

        for temperature_scale in 2**np.linspace(-2, 2, 5):
            magnetogram = magnetograms.get_radial(magnetogram_name)
            degree_l, order_m, alpha_lm = magnetogram.as_arrays(include_unset=False)

            polar_max, azimuth_max, bmax = source_surface_field_maximum(degree_l, order_m,
                                                                        np.real(alpha_lm),
                                                                        np.imag(alpha_lm),
                                                                        rs, rss)

            field_radial, field_polar, field_azimuth = evaluate_along_ray(radial_points, polar_max, azimuth_max,
                                                                          degree_l, order_m,
                                                                          np.real(alpha_lm),
                                                                          np.imag(alpha_lm),
                                                                          rs, rss)

            p = ParkerSolution(temperature=p0.temperature * temperature_scale)
            velocity = p.speed(radial_points * p.stellar_radius)
            density = p.density(radial_points * p.stellar_radius)

            alfven_speed = (1e-4*field_radial / np.sqrt(scipy.constants.mu_0 * density))

            line = ax.plot(radial_points, alfven_speed, label="Alfven speed T=%g" % p.temperature)

            ax.plot(radial_points, velocity,
                    '--',
                    color=line[0].get_color(),
                    label="Parker flow speed T=%g" % p.temperature)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        plt.legend()
        plt.savefig(pn.get())
        plt.close()
