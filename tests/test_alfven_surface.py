import logging

import matplotlib as mpl
import matplotlib.lines
import numpy as np
import scipy.constants

log = logging.getLogger(__name__)

# Test "context"
import pytest
from tests import context  # Test context
from tests import magnetograms

# Local
from starwinds_magnetogram.geometry import ZdiGeometry
from starwinds_magnetogram.parker_solution import ParkerSolution
from starwinds_magnetogram import pfss_magnetogram
from starwinds_magnetogram import plot_pfss
from matplotlib.colors import SymLogNorm


def b_alfven(u, rho):
    """
    Return the magnetic field required for the Alfven surface to lie at the point
    defined by the speed and density.
    :param u: speed (SI units)
    :param rho: density (SI units)
    :return: Magnetic field strength (SI units)
    """
    return u * (scipy.constants.mu_0 * rho)**.5


def get_alfven_mach_number(bmag, density, velocity):
    """
    NOTE CONVERTS FROM GAUSS TO TESLA!!
    :param bmag:
    :param density:
    :param velocity:
    :return:
    """
    with np.errstate(divide='ignore'):
        return velocity / (1e-4 * bmag / (scipy.constants.mu_0 * density)**.5)


def test_alfven_field_temperature(request):
    """
    Calculates the magnetic field required for the Alfven surface to fall at a given point.
    This shows that as the field drops, the Alfven surface moves outwards. Furthermore there
    are temperature/field combinations for which the Alfven surface does not exist (strong field/low
    temperature)
    :param request:
    :return:
    """

    temperatures = (0.5e6, 0.75e6, 1e6, 1.5e6, 2e6, 3e6, 4e6, 1e7)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        fig, ax = plt.subplots()
        for _id, temperature in enumerate(temperatures):

            p = ParkerSolution(temperature=temperature)

            r = p.stellar_radius * np.geomspace(1, 10)
            c = plt.rcParams['axes.prop_cycle'].by_key()['color'][_id]

            u = p.speed(r)
            rho = p.density(r)

            # ax.plot(p.radius_sonic/p.stellar_radius,
            #          b_alfven(p.speed_sonic, p.density_sonic),
            #          'o',
            #          color=c)

            ax.plot(r/p.stellar_radius,
                    b_alfven(u, rho),
                    color=c,
                    label="T = %2.3G K" % temperature)

        ax.set_xlabel(r'Height over chromosphere [$R_{\star}$]')
        ax.set_ylabel('Alfven field [T]')
        ax.set_yscale('log')
        ax.grid(True)
        ax.set_title("Required field for Alfven surface to lie at height")
        ax.legend()
        ax.set_xlim(left=1)

        s0 = "Parker solutions Rho_c=%2.2G kg/m3, " % (p.base_density,)
        s1 = "R=%2.2G, M=%2.2G" % (p.stellar_radius, p.stellar_mass)

        ax.text(1.0 - 0.01, 1.0 - 0.01, s0 + s1,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)

        fig.savefig(pn.get())
        plt.close(fig)


def test_alfven_field_density(request):
    """
    Calculates the magnetic field required for the Alfven surface to fall at a given point.
    This shows that as the field drops, the Alfven surface moves outwards. Furthermore there
    are temperature/density combinations for which the Alfven surface does not exist (strong field/low
    density). Also there are sometimes two solutions.
    :param request:
    :return:
    """

    # A common value is 10^15 electrons/m3 (for the Sun)
    base_number_densities = np.geomspace(1e13, 1e17, 5)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        fig, ax = plt.subplots()
        for _id, base_number_density in enumerate(base_number_densities):

            p = ParkerSolution(base_density=base_number_density * scipy.constants.proton_mass)

            r = p.stellar_radius * np.geomspace(1, 10)
            c = plt.rcParams['axes.prop_cycle'].by_key()['color'][_id]

            u = p.speed(r)
            rho = p.density(r)

            # ax.plot(p.radius_sonic/p.stellar_radius,
            #          b_alfven(p.speed_sonic, p.density_sonic),
            #          'o',
            #          color=c)

            ax.plot(r/p.stellar_radius,
                    b_alfven(u, rho),
                    color=c,
                    label=r"$n$=%2.3G, $\rho$ = %2.3G K" % (base_number_density, p.base_density),
                    )

        ax.set_xlabel(r'Height over chromosphere [$R_{\star}$]')
        ax.set_ylabel('Alfven field [T]')
        ax.set_yscale('log')
        ax.grid(True)
        ax.set_title("Required field for Alfven surface to lie at height")
        ax.legend(loc='lower right')
        ax.set_xlim(left=1)

        s0 = "Parker solutions Rho_c=%2.2G kg/m3, " % (p.base_density,)
        s1 = "R=%2.2G, M=%2.2G" % (p.stellar_radius, p.stellar_mass)

        ax.text(1.0 - 0.01, 1.0 - 0.01, s0 + s1,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)

        fig.savefig(pn.get())
        plt.close(fig)


@pytest.mark.parametrize("magnetogram_name", ("dipole",))
def test_slice(request,
               magnetogram_name):
    """
    Use the PFSS method to plot a two-dimensional slice of the magnetic field.
    This is in magnetogram units, i.e. Gauss
    TODO Should move this to test_plot_pfss (which does not exist).
    :param request:
    :param magnetogram_name:
    :return:
    """

    radial_coefficients = magnetograms.get_radial(magnetogram_name) * 10

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, ax = plot_pfss.plot_slice(radial_coefficients)
        ax.set_title("Radial field strength (magnetogram units)")
        fig.savefig(pn.get())
        plt.close()


def test_alfven_shape(request):
    """
    Observe how the Alfven surface moves with dipole field strength
    TODO check Gauss/Tesla carefully
    :param request:
    :return:
    """

    normal = "x"
    radius_star = pfss_magnetogram.default_radius_star  # The default is 1
    radius_source_surface = pfss_magnetogram.default_radius_source_surface
    radius_max = 10

    # Create points in slice plane xy coordinate system.
    p1 = np.linspace(-1, 1, 102) * radius_max
    p2 = np.linspace(-1, 1, 104) * radius_max
    p1, p2 = np.meshgrid(p1, p2)

    pxyz = pfss_magnetogram.normal_plane(p1, p2, normal)

    p = ParkerSolution()

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        fig, ax = plt.subplots(figsize=(9, 6))

        h, l = ax.get_legend_handles_labels()

        for _id, _scale in enumerate(np.geomspace(1/4, 4, 5)):
            # This returns a starwinds_magnetogram.coefficients.Coefficients object.
            radial_coefficients = magnetograms.get_radial("dipole") * _scale

            f_rpa_xyz = pfss_magnetogram.evaluate_cartesian(radial_coefficients,
                                                            *pxyz,
                                                            radius_star,
                                                            radius_source_surface,
                                                            )

            # Drop the extra dimension
            pxyz = [np.squeeze(p) for p in pxyz]
            px, py, pz = pxyz
            f_rpa_xyz = [np.squeeze(f) for f in f_rpa_xyz]
            fr, fp, fa, fx, fy, fz = f_rpa_xyz

            # These are all two-dimensional arrays
            radial_distance = fr * p.stellar_radius
            velocity = p.speed(radial_distance)
            density = p.density(radial_distance)

            # These are also two-dimensional arrays
            bmag = (fr**2 + fp**2 + fa**2)**.5
            alfven_mach_number = get_alfven_mach_number(bmag, density, velocity)

            # Add this only for the first iteration.
            if _id == 0:
                bmin = np.min(bmag[np.where(bmag > 0)])
                bmax = np.max(bmag)
                norm = mpl.colors.SymLogNorm(linthresh=100 * bmin,
                                             linscale=1,
                                             vmin=-bmax,
                                             vmax=bmax,
                                             base=10)
                ax.grid(False)
                im = ax.pcolormesh(p1, p2, fr,
                                   norm=norm,
                                   cmap='RdBu_r'
                                   )
                ax.streamplot(p1, p2,
                              fy, fz,
                              color='gray')

                # Add source surface to plot
                pr = (px**2 + py**2 + pz**2)**.5
                r_ss = ax.contour(p1, p2, pr, levels=[radius_source_surface], colors='black')
                h += [matplotlib.lines.Line2D([], [], color=r_ss.collections[0].get_edgecolor())]
                l += ["Source surface"]

            # Add Alfven surface to plot (for every iteration)
            r_a = ax.contour(p1, p2, alfven_mach_number, levels=(1,), colors=colors[_id])
            h += [matplotlib.lines.Line2D([], [], color=r_a.collections[0].get_edgecolor())]
            l += ["Alfven surface, scale=%2.3G" % _scale]

        ax.set_aspect('equal')
        ax.set_xlabel(r'Distance $x/R_{\star}$')
        ax.set_ylabel(r'Distance $z/R_{\star}$')

        plt.legend(h, l, loc="lower left")

        ax.text(1.0 - 0.01, 1.0 - 0.01, str(p),
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes,
                wrap=True)

        plt.savefig(pn.get())


def test_alfven_shape_simple(request):
    """
    Observe how the Alfven surface moves with dipole field strength
    TODO check Gauss/Tesla carefully
    :param request:
    :return:
    """

    normal = "x"
    radius_star = 1
    radius_source_surface = pfss_magnetogram.default_radius_source_surface
    radius_max = 10

    # Create points in slice plane xy coordinate system.
    p1 = np.linspace(-1, 1, 102) * radius_max
    p2 = np.linspace(-1, 1, 104) * radius_max
    p1, p2 = np.meshgrid(p1, p2)

    pxyz = pfss_magnetogram.normal_plane(p1, p2, normal)

    p = ParkerSolution()

    # This returns a starwinds_magnetogram.coefficients.Coefficients object.
    radial_coefficients = magnetograms.get_radial("dipole")

    f_rpa_xyz = pfss_magnetogram.evaluate_cartesian(radial_coefficients,
                                                    *pxyz,
                                                    radius_star,
                                                    radius_source_surface,)

    # Drop the extra dimension
    pxyz = [np.squeeze(p) for p in pxyz]
    px, py, pz = pxyz
    f_rpa_xyz = [np.squeeze(f) for f in f_rpa_xyz]
    fr, fp, fa, fx, fy, fz = f_rpa_xyz

    # These are all two-dimensional arrays
    radial_distance = fr * p.stellar_radius
    velocity = p.speed(radial_distance)
    density = p.density(radial_distance)

    # These are also two-dimensional arrays
    bmag = (fr**2 + fp**2 + fa**2)**.5
    alfven_mach_number = get_alfven_mach_number(bmag, density, velocity)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        fig, ax = plt.subplots(figsize=(9, 6))
    
        # Add this only for the first iteration.
        bmin = np.min(bmag[np.where(bmag > 0)])
        bmax = np.max(bmag)
        norm = mpl.colors.SymLogNorm(linthresh=100 * bmin,
                                     linscale=1,
                                     vmin=-bmax,
                                     vmax=bmax,
                                     base=10)
        # ax.grid(False)
        # im = ax.pcolormesh(p1, p2, fr,
        #                    norm=norm,
        #                    cmap='RdBu_r'
        #                    )
        # ax.streamplot(p1, p2,
        #               fy, fz,
        #               color='gray')


        # Add source surface to plot
        _handlers, _labels = ax.get_legend_handles_labels()
        pr = (px**2 + py**2 + pz**2)**.5
        r_ss = ax.contour(p1, p2, pr, levels=[radius_source_surface], colors='grey')
        _handlers += [matplotlib.lines.Line2D([], [], color=r_ss.collections[0].get_edgecolor())]
        _labels += ["Source surface"]

        # Add Alfven surface to plot (for every iteration)
        levels = (.125, .25, .5, 1, 2, 4, 8)
        r_a = ax.contour(p1, p2, alfven_mach_number, levels=levels)
        for _l, _c in zip(levels, r_a.collections):
            _handlers += [matplotlib.lines.Line2D([], [], color=_c.get_edgecolor())]
            _labels += ["Alfven surface l=%2.3G" % _l]

        ax.set_aspect('equal')
        ax.set_xlabel(r'Distance $x/R_{\star}$')
        ax.set_ylabel(r'Distance $z/R_{\star}$')

        ax.text(1.0 - 0.01, 1.0 - 0.01, str(p),
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes,
                wrap=True)

        ax.grid()

        ax.legend(_handlers, _labels, loc="lower left")
        plt.savefig(pn.get())

        #
        # New plot showing min, max, etc.
        #
        fig, ax = plt.subplots()
        levels =np.linspace(2**-5, 4)
        r_a = ax.contour(p1, p2, alfven_mach_number, levels=levels)
        fig.colorbar(r_a)
        fig.savefig(pn.get())

        fig, ax = plt.subplots()

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        _min = []
        _paths = []
        _max = []
        for _l, _c in zip(levels, r_a.collections):
            pos = np.vstack([p.vertices for p in _c.get_paths()])
            radius = np.sum(pos**2, axis=1)**.5
            ax.plot(_l * np.ones_like(radius), radius, 'k.')

            _min.append(np.min(radius))
            _paths.append(len(_c.get_paths()))
            _max.append(np.max(radius))

        ax.fill_between(levels, _min, _max, label="Alfven radial distance",
                        color=colors[0])
        ax.plot(levels, _paths, 'x', label="Segment count", color=colors[1])
        ax.plot(levels, _max, color='black')
        ax.grid()
        ax.set_title("Alfven surface distance from star")
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        ax.legend()
        ax.set_xlabel(r"Dipole $\alpha_{10}$ coefficient [Gauss] (all other coefficients are 0).")
        ax.set_ylabel(r"Radius [$R_\star$]")

        fig.savefig(pn.get())


@pytest.mark.parametrize("magnetogram_name", ("dipole", "mengel"))
@pytest.mark.parametrize("plot_name", ("B_r", "B", "M_A"))
def test_alfven_slice(request,
                      magnetogram_name,
                      plot_name):
    """
    Plot a two-dimensional slice of the magnetic field and the Alfven surface.
    TODO this is really a plot function.
    :param request:
    :param magnetogram_name:
    :param plot_name:
    :return:
    """

    normal = "x"
    radius_star = pfss_magnetogram.default_radius_star
    radius_source_surface = pfss_magnetogram.default_radius_source_surface
    radius_max = 6

    # Create points in slice plane xy coordinate system.
    p1 = np.linspace(-1, 1, 102) * radius_max
    p2 = np.linspace(-1, 1, 104) * radius_max
    p1, p2 = np.meshgrid(p1, p2)

    pxyz = pfss_magnetogram.normal_plane(p1, p2, normal)

    # This returns a starwinds_magnetogram.coefficients.Coefficients object.
    radial_coefficients = magnetograms.get_radial(magnetogram_name)

    f_rpa_xyz = pfss_magnetogram.evaluate_cartesian(radial_coefficients, *pxyz,
                                                    radius_star=radius_star,
                                                    radius_source_surface=radius_source_surface,
                                                    )

    if magnetogram_name == "dipole":
        f_rpa = pfss_magnetogram.evaluate_spherical(radial_coefficients,
                                                    np.atleast_3d(1),
                                                    np.atleast_3d(0),
                                                    np.atleast_3d(0),
                                                    radius_star=radius_star,
                                                    radius_source_surface=radius_source_surface,
                                                    )

        assert np.allclose(f_rpa[0], 1)  # Just one element inside a triple array

    # Drop the extra dimension
    pxyz = (p[..., 0] for p in pxyz)
    f_rpa_xyz = (f[..., 0] for f in f_rpa_xyz)

    px, py, pz = pxyz
    pr = (px**2 + py**2 + pz**2)**.5

    fr, fp, fa, fx, fy, fz = f_rpa_xyz

    p = ParkerSolution()
    radial_distance = pr * p.stellar_radius
    velocity = p.speed(radial_distance)
    density = p.density(radial_distance)

    bmag = (fx**2 + fy**2 + fz**2)**.5
    assert np.allclose(bmag, (fr**2 + fp**2 + fa**2)**.5)

    alfven_mach_number = get_alfven_mach_number(bmag, density, velocity)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        fig, ax = plt.subplots(figsize=(9, 6))

        # Some hacky branching here
        if plot_name == "B_r":
            from matplotlib.colors import SymLogNorm

            bmin = np.min(bmag[np.where(bmag > 0)])
            bmax = np.max(bmag)

            norm = mpl.colors.SymLogNorm(linthresh=100 * bmin,
                                         linscale=1,
                                         vmin=-bmax,
                                         vmax=bmax,
                                         base=10)

            ax.grid(False)
            im = ax.pcolormesh(p1, p2, fr,
                               norm=norm,
                               cmap='RdBu_r'
                               )

            ax.streamplot(p1, p2,
                          fy, fz,
                          color='gray')

            fig.colorbar(im).set_label('Radial field strength')

            max_pos = np.argwhere(bmag == bmax)
            for _id in range(max_pos.shape[0]):
                ax.plot(p1[max_pos[_id, 0], max_pos[_id, 1]],
                        p2[max_pos[_id, 0], max_pos[_id, 1]], 'kx')

        elif plot_name == "B":
            norm = mpl.colors.LogNorm()

            ax.grid(False)
            im = ax.pcolormesh(p1, p2, bmag,
                               norm=norm,
                               cmap='viridis')
            fig.colorbar(im).set_label('Absolute field strength')

        else:

            norm = mpl.colors.LogNorm(vmin=1e-2,
                                      vmax=1e2)

            ax.grid(False)
            im = ax.pcolormesh(p1, p2,
                               alfven_mach_number,
                               norm=norm,
                               cmap='PuOr')

            fig.colorbar(im).set_label('Alfven number')

        # Add Alfven surface to plot
        _h, _l = ax.get_legend_handles_labels()
        r_a = ax.contour(p1, p2, alfven_mach_number, levels=(1,), colors='magenta')
        _h += [matplotlib.lines.Line2D([], [], color=r_a.collections[0].get_edgecolor())]
        _l += ["Alfven surface"]

        # Add source surface to plot
        pr = (px**2 + py**2 + pz**2)**.5
        r_ss = ax.contour(p1, p2, pr, levels=[radius_source_surface], colors='green')
        _h += [matplotlib.lines.Line2D([], [], color=r_ss.collections[0].get_edgecolor())]
        _l += ["Source surface"]

        # Add stellar surface and source surface to plot
        r_s = ax.contour(p1, p2, pr, levels=[radius_star], colors='yellow')

        ax.set_aspect('equal')
        ax.set_xlabel(r'Distance $x/R_{\star}$')
        ax.set_ylabel(r'Distance $z/R_{\star}$')

        ax.legend(_h, _l, loc="lower left")
        ax.grid(True)

        ax.text(1.0 - 0.01, 1.0 - 0.01, str(p),
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes,
                wrap=True)


        plt.savefig(pn.get())


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

            polar_max, azimuth_max, bmax = source_surface_field_maximum(magnetogram,
                                                                        rs, rss)

            field_radial, field_polar, field_azimuth = evaluate_along_ray(magnetogram,
                                                                          radial_points,
                                                                          polar_max,
                                                                          azimuth_max,
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

            polar_max, azimuth_max, bmax = source_surface_field_maximum(magnetogram,
                                                                        rs, rss)

            field_radial, field_polar, field_azimuth = evaluate_along_ray(magnetogram,
                                                                          radial_points,
                                                                          polar_max,
                                                                          azimuth_max,
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

            polar_max, azimuth_max, bmax = source_surface_field_maximum(magnetogram,
                                                                        rs, rss)

            field_radial, field_polar, field_azimuth = evaluate_along_ray(magnetogram,
                                                                          radial_points,
                                                                          polar_max,
                                                                          azimuth_max,
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


def source_surface_field_maximum(magnetogram, radius_star, radius_source_surface):

    """
    Find the spherical coordinates of the point where the magnetic field strength is maximal
    on the source surface; outside of the source surface
    this will remain the strongest field location.
    :return:
    """
    points_polar, points_azimuth = ZdiGeometry().centers()
    field_radial, field_polar, field_azimuthal = pfss_magnetogram.evaluate_spherical(
        magnetogram,
        radius_source_surface, points_polar, points_azimuth,  # Use R_ss for radius
        radius_star=radius_star,
        radius_source_surface=radius_source_surface)

    # On the source surface (and outside) the polar and azimuthal components are zero.
    assert np.allclose(field_polar, 0)
    assert np.allclose(field_azimuthal, 0)

    # Get the indices of the field maximum
    indices = np.unravel_index(np.argmax(field_radial, axis=None), field_radial.shape)

    return points_polar[indices], points_azimuth[indices], field_radial[indices]


def evaluate_along_ray(magnetogram,
                       pr, pp, pa,
                       radius_star, radius_source_surface):
    """
    Calculate field strength along ray
    :param pr:
    :param pp:
    :param pa:
    :param degree_l:
    :param order_m:
    :param g_lm:
    :param h_lm:
    :param radius_star:
    :param radius_source_surface:
    :return:
    """
    pr, pp, pa = np.meshgrid(pr, pp, pa)

    field_radial, \
    field_polar, \
    field_azimuthal = pfss_magnetogram.evaluate_spherical(magnetogram,
                                                          pr, pp, pa,
                                                          radius_star=radius_star,
                                                          radius_source_surface=radius_source_surface)
    assert field_radial.shape == pr.shape
    assert field_polar.shape == pr.shape
    assert field_azimuthal.shape == pr.shape

    # assert np.all(field_polar[np.where(pr > radius_source_surface)] == 0)  # This is NaN now.
    assert np.all(field_azimuthal[np.where(pr > radius_source_surface)] == 0)

    pr = np.squeeze(pr)

    field_radial = np.squeeze(field_radial)
    field_polar = np.squeeze(field_polar)
    field_azimuthal = np.squeeze(field_azimuthal)

    return field_radial, field_polar, field_azimuthal

