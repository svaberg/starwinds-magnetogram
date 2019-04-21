import numpy as np
import matplotlib as mpl
import scipy.constants
import logging
log = logging.getLogger(__name__)

# Test "context"
from tests import context  # Test context
from tests.magnetogram import magnetograms

# Local
import stellarwinds.magnetogram.parker_solution as parker
import stellarwinds.magnetogram.pfss_magnetogram as pfss_stanford
import stellarwinds.coordinate_transforms

def test_alfven(request):

    coronal_temperatures = (0.5e6, 0.75e6, 1e6, 1.5e6, 2e6, 3e6, 4e6)
    coronal_base_density = 2e16 * scipy.constants.proton_mass
    stellar_radius = 695.510e6
    stellar_mass = 1.989e30

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        for _id, coronal_temperature in enumerate(coronal_temperatures):
            r = np.geomspace(stellar_radius, 215 * stellar_radius)  # 20*r_S)
            c = plt.rcParams['axes.prop_cycle'].by_key()['color'][_id]

            u, rho, r_sonic, u_sonic, rho_sonic = parker.parker_solution(r,
                                                                         coronal_temperature=coronal_temperature,
                                                                         coronal_base_density=coronal_base_density,
                                                                         stellar_radius=stellar_radius,
                                                                         stellar_mass=stellar_mass)

            plt.plot(r/stellar_radius, u * (scipy.constants.mu_0 * rho)**.5, color=c)

        plt.xlabel(r'Height over chromosphere [$R_{\star}$]')
        plt.ylabel('Alfven field [T]')
        # plt.ylim((1e-2, 1e11))
        plt.yscale('log')
        # plt.xlim((r_S, 15*r_S))
        plt.grid(True)
        plt.savefig(pn.get())



def test_alfven_surface(request,
                        magnetogram_name="quadrupole"):



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

    pr, pp, pa = stellarwinds.coordinate_transforms.spherical_coordinates_from_rectangular(px,py,pz)

    degree_l, order_m, alpha_lm = magnetograms.get_radial(magnetogram_name).as_arrays(include_unset=False)
    field_radial, field_polar, field_azimuthal = pfss_stanford.evaluate_in_space(degree_l, order_m,
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


    transformation_matrix = stellarwinds.coordinate_transforms.spherical_to_rectangular_transformation_matrix(pp.flatten(),
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

    import stellarwinds.magnetogram.parker_solution as parker
    import scipy.constants
    # import pdb; pdb.set_trace()
    radial_distance = field_radial * 6.95510e8
    velocity, density, r_sonic, u_sonic, rho_sonic = parker.parker_solution(radial_distance)
    velocity = velocity.reshape(_shape)
    density = density.reshape(_shape)
    alfven_mach_number = velocity / (1e-4*bmag / np.sqrt(scipy.constants.mu_0 * density))

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, ax = plt.subplots(figsize=(6, 9))
        ax.contourf(np.squeeze(bz[0, :, :]))
        plt.savefig(pn.get())
        plt.close()


        fig, ax = plt.subplots(figsize=(9, 6))
        # ax.plot(px[0,:,:], pz[0,:,:])

        if True:
            from matplotlib.colors import SymLogNorm
            b= (bx[0, :, :] ** 2 + by[0, :, :] ** 2 + bz[0, :, :] ** 2) ** .5
            bmin = np.min(b[np.where(b > 0)])
            bmax = np.max(b)

            norm = mpl.colors.SymLogNorm(linthresh=100 * bmin,
                                         linscale=1,
                                         vmin=-bmax,
                                         vmax=bmax)

            # im = ax.pcolormesh(px[0, :, :], pz[0, :, :], alfven_mach_number[0, :, :],
            #                    norm=norm,
            #                    cmap='PuOr'
            im = ax.pcolormesh(px[0, :, :], pz[0, :, :], field_radial[0, :, :],
                               norm=norm,
                               cmap='RdBu_r'
                               )
            fig.colorbar(im).set_label('Radial field strength')

        else:

            norm = mpl.colors.LogNorm(vmin=1e-2,
                                      vmax=1e2)

            im = ax.pcolormesh(px[0, :, :], pz[0, :, :], field_radial[0, :, :],
                               norm=norm,
                               cmap='PuOr')
            # ax.contour(px[0, :, :], pz[0, :, :], field_radial[0,:,:], levels=6, colors='gray')  # symlog hack.


            fig.colorbar(im).set_label('Alfven number')
        #
        ax.contour(px[0, :, :], pz[0, :, :], alfven_mach_number[0, :, :], levels=(1,), colors='black')
        ax.contour(px[0, :, :], pz[0, :, :], pr[0,:,:], levels=[rs, rss], colors='black')
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
        plt.savefig(pn.get())


def bmax_location(degree_l, order_m, g_lm, h_lm, rs, rss):

    """
    Find the spherical coordinates of Bmax on the source surface; outside of the source surface
    this will remain the strongest field location.
    :return:
    """
    points_polar, points_azimuth = np.meshgrid(np.linspace(0, np.pi), np.linspace(0, 2*np.pi))
    field_radial, field_polar, field_azimuthal = pfss_stanford.evaluate_on_sphere(
        degree_l, order_m, g_lm, h_lm,
        points_polar, points_azimuth,
        radius=rss, radius_star=rs, radius_source_surface=rss)

    assert np.allclose(field_polar, 0)
    assert np.allclose(field_azimuthal, 0)

    _max_ids = np.unravel_index(np.argmax(field_radial, axis=None), field_radial.shape)

    return points_polar[_max_ids], points_azimuth[_max_ids], field_radial[_max_ids]


def line_strength(pr, pp, pa,
                  degree_l, order_m, g_lm, h_lm,
                  rs, rss):
    """
    Calculate total field strength along ray
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
    field_radial, field_polar, field_azimuthal = pfss_stanford.evaluate_in_space(degree_l, order_m, g_lm, h_lm,
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
    star_radius = 6.95510e8
    radial_points = np.geomspace(rs, rmax)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, ax = plt.subplots(figsize=(6, 9))

        for scale in (.1, 1, 10):
            magnetogram = magnetograms.get_radial(magnetogram_name)
            magnetogram *= scale
            degree_l, order_m, alpha_lm = magnetogram.as_arrays(include_unset=False)

            polar_max, azimuth_max, bmax = bmax_location(degree_l, order_m,
                                                         np.real(alpha_lm),
                                                         np.imag(alpha_lm),
                                                         rs, rss)

            field_radial, field_polar, field_azimuth = line_strength(radial_points, polar_max, azimuth_max,
                                                                     degree_l, order_m,
                                                                     np.real(alpha_lm),
                                                                     np.imag(alpha_lm),
                                                                     rs, rss)

            velocity, density, r_sonic, u_sonic, rho_sonic = parker.parker_solution(radial_points * star_radius)
            alfven_speed = (1e-4*field_radial / np.sqrt(scipy.constants.mu_0 * density))

            ax.plot(radial_points, velocity/alfven_speed)
            # ax.plot(radial_points, alfven_speed)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)

        plt.savefig(pn.get())
        plt.close()


        fig, ax = plt.subplots(figsize=(6, 9))

        for density_sacle in (.1, 1, 10):
            magnetogram = magnetograms.get_radial(magnetogram_name)
            magnetogram *= scale
            degree_l, order_m, alpha_lm = magnetogram.as_arrays(include_unset=False)

            polar_max, azimuth_max, bmax = bmax_location(degree_l, order_m,
                                                         np.real(alpha_lm),
                                                         np.imag(alpha_lm),
                                                         rs, rss)

            field_radial, field_polar, field_azimuth = line_strength(radial_points, polar_max, azimuth_max,
                                                                     degree_l, order_m,
                                                                     np.real(alpha_lm),
                                                                     np.imag(alpha_lm),
                                                                     rs, rss)

            velocity, density, r_sonic, u_sonic, rho_sonic = parker.parker_solution(radial_points * star_radius,
                                                                                    coronal_base_density=1.5e14*scipy.constants.proton_mass*density_sacle)
            alfven_speed = (1e-4*field_radial / np.sqrt(scipy.constants.mu_0 * density))

            ax.plot(radial_points, velocity/alfven_speed)
            # ax.plot(radial_points, alfven_speed)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)

        plt.savefig(pn.get())
        plt.close()


        fig, ax = plt.subplots(figsize=(6, 9))

        for temperature_sacle in 2**np.linspace(-2, 2, 5):
            magnetogram = magnetograms.get_radial(magnetogram_name)
            magnetogram *= scale
            degree_l, order_m, alpha_lm = magnetogram.as_arrays(include_unset=False)

            polar_max, azimuth_max, bmax = bmax_location(degree_l, order_m,
                                                         np.real(alpha_lm),
                                                         np.imag(alpha_lm),
                                                         rs, rss)

            field_radial, field_polar, field_azimuth = line_strength(radial_points, polar_max, azimuth_max,
                                                                     degree_l, order_m,
                                                                     np.real(alpha_lm),
                                                                     np.imag(alpha_lm),
                                                                     rs, rss)

            velocity, density, r_sonic, u_sonic, rho_sonic = parker.parker_solution(radial_points * star_radius,
                                                                                    coronal_temperature=1.5e6 * temperature_sacle)
            alfven_speed = (1e-4*field_radial / np.sqrt(scipy.constants.mu_0 * density))

            ax.plot(radial_points, velocity/alfven_speed, label="B=%g rho=%g T=%g" % (bmax,
                                                                                      0,
                                                                                      1.5e6 * temperature_sacle))
            # ax.plot(radial_points, alfven_speed)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)

        plt.legend()

        plt.savefig(pn.get())
        plt.close()
