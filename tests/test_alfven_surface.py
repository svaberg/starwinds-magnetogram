import numpy as np
import matplotlib as mpl
import scipy.constants
import logging
log = logging.getLogger(__name__)

# Test "context"
import pytest
from tests import context  # Test context
from tests.magnetogram import magnetograms

# Local
from stellarwinds.magnetogram.geometry import ZdiGeometry
from stellarwinds.magnetogram.parker_solution import ParkerSolution
from stellarwinds.magnetogram import pfss_magnetogram
from stellarwinds.coordinate_transforms import spherical_coordinates_from_rectangular
from stellarwinds.coordinate_transforms import spherical_to_rectangular_transformation_matrix


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
def test_alfven_surface(request,
                        magnetogram_name,
                        plot_name):

    rss = 3
    rs = 1
    rmax = 6

    _x = np.linspace(-1, 1, 302) * rmax
    _y = 0
    _z = np.linspace(-1, 1, 304) * rmax

    px, py, pz = np.meshgrid(_x, _y, _z)

    log.debug(_x.shape)

    assert px.shape == (1, len(_x), len(_z))  # np.meshgrid switches argument 1 and 2 in result.

    _shape = px.shape

    pr, pp, pa = spherical_coordinates_from_rectangular(px,py,pz)

    radial_coefficients = magnetograms.get_radial(magnetogram_name) * 10
    degree_l, order_m, alpha_lm = radial_coefficients.as_arrays(include_unset=False)
    field_radial, field_polar, field_azimuthal = pfss_magnetogram.evaluate_in_space(degree_l, order_m,
                                                                                    np.real(alpha_lm),
                                                                                    np.imag(alpha_lm),
                                                                                    pr, pp, pa,
                                                                                    radius_star=rs,
                                                                                    radius_source_surface=rss)

    assert field_radial.shape == pr.shape
    assert field_polar.shape == pr.shape
    assert field_azimuthal.shape == pr.shape

    # assert np.all(field_polar[np.where(pr > rss)] == 0)
    # assert np.all(field_azimuthal[np.where(pr > rss)] == 0)

    Frpa = np.stack([c.flatten() for c in (field_radial, field_polar, field_azimuthal)], axis=-1)

    polarity = np.sign(Frpa[:, 0] * pr.flatten()).reshape(_shape)

    transformation_matrix = spherical_to_rectangular_transformation_matrix(pp.flatten(),
                                                                           pa.flatten())
    Fxyz = transformation_matrix @ Frpa[:, :, np.newaxis]
    # Get rid of last dimension which is 1
    Fxyz = np.squeeze(Fxyz)

    bx = Fxyz[:, 0].reshape(_shape)
    by = Fxyz[:, 1].reshape(_shape)
    bz = Fxyz[:, 2].reshape(_shape)
    bmag = (bx ** 2 + by ** 2 + bz ** 2) ** .5
    bmin = np.min(bmag[np.where(bmag > 0)])
    bmax = np.max(bmag)

    p = ParkerSolution()
    radial_distance = field_radial * 6.95510e8
    velocity = p.speed(radial_distance)
    density = p.density(radial_distance)

    velocity = velocity.reshape(_shape)
    density = density.reshape(_shape)
    alfven_mach_number = velocity / (1e-4*bmag / np.sqrt(scipy.constants.mu_0 * density))

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        fig, ax = plt.subplots(figsize=(9, 6))
        # ax.plot(px[0,:,:], pz[0,:,:])

        if plot_name == "B_r":
            from matplotlib.colors import SymLogNorm
            b= (bx[0, :, :] ** 2 + by[0, :, :] ** 2 + bz[0, :, :] ** 2) ** .5
            bmin = np.min(b[np.where(b > 0)])
            bmax = np.max(b)

            norm = mpl.colors.SymLogNorm(linthresh=100 * bmin,
                                         linscale=1,
                                         vmin=-bmax,
                                         vmax=bmax)

            im = ax.pcolormesh(px[0, :, :], pz[0, :, :], field_radial[0, :, :],
                               norm=norm,
                               cmap='RdBu_r'
                               )
            fig.colorbar(im).set_label('Radial field strength')

        elif plot_name == "B":
            norm = mpl.colors.LogNorm()

            b = (bx ** 2 + by ** 2 + bz ** 2) ** .5

            im = ax.pcolormesh(px[0, :, :], pz[0, :, :], b[0, :, :],
                               norm=norm,
                               cmap='viridis')
            fig.colorbar(im).set_label('Absolute field strength')

        else:

            norm = mpl.colors.LogNorm(vmin=1e-2,
                                      vmax=1e2)

            im = ax.pcolormesh(px[0, :, :],
                               pz[0, :, :],
                               alfven_mach_number[0, :, :],
                               norm=norm,
                               cmap='PuOr')
            # im = ax.pcolormesh(px[0, :, :], pz[0, :, :], field_radial[0, :, :],
            #                    norm=norm,
            #                    cmap='PuOr')

            fig.colorbar(im).set_label('Alfven number')

        r_a = ax.contour(px[0, :, :], pz[0, :, :], alfven_mach_number[0, :, :], levels=(1,), colors='black')
        r_a.collections[0].set_label("Alfven surface")

        r_ss = ax.contour(px[0, :, :], pz[0, :, :], pr[0,:,:], levels=[rs, rss], colors='white')
        r_ss.collections[0].set_label("Source surface")

        # # ax.quiver(px[0,:,:], pz[0,:,:], fx[0,:,:], fz[0,:,:])
        ax.streamplot(px[0,:,:].transpose(), pz[0,:,:].transpose(),
                      bx[0,:,:].transpose(), bz[0,:,:].transpose(),
                      color='gray')
        # # ax.streamplot(pz[0,:,:], px[0,:,:], fz[0,:,:], fx[0,:,:])
        #
        # ax.contour(px[0, :, :], pz[0, :, :], pr[0,:,:], levels=[rs, rss], colors='k')
        #
        ax.set_aspect('equal')
        ax.set_xlabel('Distance $x/r_{\\star}$')
        ax.set_ylabel('Distance $z/r_{\\star}$')

        plt.legend()
        plt.savefig(pn.get())


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


def test_alfven_surface_max(request,
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

            ax.plot(radial_points, velocity/alfven_speed, label="B=%4.2g rho=%4.2g T=%4.2g" % (bmax,
                                                                                      p.base_density,
                                                                                      p.temperature))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        plt.legend()
        plt.savefig(pn.get())
        plt.close()



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

            ax.plot(radial_points, velocity/alfven_speed, label="B=%g rho=%g T=%g" % (bmax,
                                                                                      p.base_density,
                                                                                      p.temperature))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        plt.legend()
        plt.savefig(pn.get())
        plt.close()


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

            ax.plot(radial_points, velocity/alfven_speed, label="B=%g rho=%g T=%g" % (bmax,
                                                                                      p.base_density,
                                                                                      p.temperature))

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        plt.legend()
        plt.savefig(pn.get())
        plt.close()
