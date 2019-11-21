import numpy as np
import scipy.constants
import logging
log = logging.getLogger(__name__)

import pytest

# Test "context"
from tests import context  # Test context

# Local
from stellarwinds.magnetogram.parker_solution import ParkerSolution


def test_dimensionless(request):
    """
    Test dimensionless Parker solution
    :param request:
    :return:
    """
    s = np.logspace(-1, np.log10(8), 100)

    w = ParkerSolution.find_parker_analytic(s)
    wp0, wn1 = ParkerSolution.find_parker_analytic(s, all_sols=True)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        plt.plot(s, w, 'k')
        plt.plot(s, wp0, 'b:', s, wn1, 'g:', linewidth=4)
        plt.xlabel('Distance [$r_c$]')
        plt.ylabel('Speed [$u_c$]')
        plt.legend(('Parker solution', 'Lambert $W_0(r)$ branch', 'Lambert $W_{-1}(r)$ branch'))
        plt.ylim((0, 3.5))
        plt.xlim((0, 8))
        plt.grid(True)
        plt.savefig(pn.get())
        plt.close()



def test_speed(request):
    """
    Calculate and plot solution speed
    :return:
    """

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        temperatures = (0.5e6, 0.75e6, 1e6, 1.5e6, 2e6, 3e6, 4e6)
        for T_id, temperature in enumerate(temperatures):

            parker_solution = ParkerSolution(temperature=temperature)

            r = np.geomspace(1, 215) * parker_solution.stellar_radius
            c = plt.rcParams['axes.prop_cycle'].by_key()['color'][T_id]

            u = parker_solution.speed(r)
            r_c, u_c = parker_solution.radius_sonic, parker_solution.speed_sonic

            r = r / parker_solution.stellar_radius
            r_c = r_c / parker_solution.stellar_radius

            plt.plot(r, u, color=c)
            plt.plot(r_c, u_c, 'o', color=c)
            plt.text(r[-1], u[-1], '%1.1f MK' % (1e-6 * temperature), color=c,
                     horizontalalignment='right', verticalalignment='bottom')

        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
        plt.xlabel(r'Height over chromosphere [$R_{\star}$]')
        plt.ylabel('Speed [m/s]')

        plt.grid(True)
        plt.savefig(pn.get())
        plt.close()


def test_density(request):
    """
    Calculate and plot solution density
    :param request:
    :return:
    """
    temperatures = (0.5e6, 0.75e6, 1e6, 1.5e6, 2e6, 3e6, 4e6)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        for _id, temperature in enumerate(temperatures):

            ps = ParkerSolution(temperature=temperature)

            r = np.geomspace(1, 215) * ps.stellar_radius
            c = plt.rcParams['axes.prop_cycle'].by_key()['color'][_id]

            rho = ps.density(r)
            r_sonic = ps.radius_sonic
            rho_sonic = ps.density_sonic

            plt.plot(r/ps.stellar_radius, rho, color=c)
            plt.plot(r_sonic/ps.stellar_radius, rho_sonic, 'o', color=c)

        plt.xlabel(r'Height over chromosphere [$R_{\star}$]')
        plt.ylabel('Density [kg/m3]')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(pn.get())
        plt.close()


def test_total_mass_flux(request):
    """
    Verify that the mass flux is constant at different radii.
    :param request:
    :return:
    """

    temperatures = (0.5e6, 0.75e6, 1e6, 1.5e6, 2e6, 3e6, 4e6)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        for _id, temperature in enumerate(temperatures):

            ps = ParkerSolution(temperature=temperature)

            r = np.geomspace(1, 215) * ps.stellar_radius
            c = plt.rcParams['axes.prop_cycle'].by_key()['color'][_id]

            u = ps.speed(r)
            rho = ps.density(r)
            r_sonic = ps.radius_sonic
            u_sonic = ps.speed_sonic
            rho_sonic = ps.density_sonic

            total_mass_flux = 4 * np.pi * r**2 * u * rho
            total_mass_flux_sonic = 4 * np.pi * r_sonic**2 * u_sonic * rho_sonic

            ax1.plot(r/ps.stellar_radius, total_mass_flux, color=c)
            ax1.plot(r_sonic/ps.stellar_radius, total_mass_flux_sonic, 'o', color=c,
                     label='T=%g' % temperature)

            ax2.plot(r / ps.stellar_radius,
                     total_mass_flux / ps.stellar_mass * 3600 * 24 * 365.25,
                     color=c)

            assert np.allclose(total_mass_flux, total_mass_flux_sonic)
            assert np.allclose(total_mass_flux_sonic, ps.total_mass_flux)


        for ax in (ax1, ax2):
            ax.set_yscale('log')

        ax1.legend()
        ax1.grid(True)
        ax1.set_xlabel(r'Radius [$R_{\star}$]')
        ax1.set_ylabel('Total mass loss [kg/s]')
        ax2.set_ylabel('Mass loss [Msun/yr]')

        plt.savefig(pn.get())
        plt.close()


def test_total_mass_flux_variation(request):
    """
    See how mass flux varies with coronal base temperature and density.
    :param request:
    :return:
    """

    temperatures = np.geomspace(0.5e6, 30e6)
    base_number_densities = np.geomspace(1.5e14, 2e16, 4)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        fig, ax = plt.subplots()
        ax2 = ax.twinx()

        for base_number_density in base_number_densities:

            ps = ParkerSolution(temperatures,
                                base_density=base_number_density * scipy.constants.proton_mass)

            ax.plot(temperatures, ps.total_mass_flux, label="Base number density %g" % base_number_density)
            ax2.plot(temperatures, ps.total_mass_flux / ps.stellar_mass * 3600 * 24 * 365.25, ',')

        ax.set_xlabel(r'Temperature [K]')
        ax.set_ylabel('Total mass loss [kg/s]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend()

        ax2.set_ylabel('Total mass loss [Msun/yr]')
        ax2.set_yscale('log')

        fig.savefig(pn.get())
        plt.close()
