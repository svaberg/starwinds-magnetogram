import numpy as np
import scipy.constants
import logging
log = logging.getLogger(__name__)

import pytest

# Test "context"
from tests import context  # Test context

# Local
import stellarwinds.magnetogram.parker_solution as parker


def test_0(request):
    s = np.logspace(-1, np.log10(8), 1e2)

    w = parker.find_parker_analytic(s)
    wp0, wn1 = parker.find_parker_analytic(s, all_sols=True)

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



def test_1(request):
    """

    :return:
    """

    r_S = 6.96e8  # m (solar/stellar radius)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):


        Ts = (0.5e6, 0.75e6, 1e6, 1.5e6, 2e6, 3e6, 4e6)
        for T_id, T in enumerate(Ts):
            r = np.geomspace(r_S, 215 * r_S)
            c = plt.rcParams['axes.prop_cycle'].by_key()['color'][T_id]

            u, r_c, u_c = parker.parker_speed(r, T)

            r = r / r_S
            r_c = r_c / r_S

            plt.plot(r, u, color=c)
            plt.plot(r_c, u_c, 'o', color=c)
            plt.text(r[-1], u[-1], '%1.1f MK' % (1e-6 * T), color=c, horizontalalignment='right', verticalalignment='bottom')

        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
        plt.xlabel('Height over chromosphere [$R_{\star}$]')
        plt.ylabel('Speed [m/s]')

        plt.grid(True)
        plt.savefig(pn.get())


def test_2(request):

    coronal_base_temperatures = (0.5e6, 0.75e6, 1e6, 1.5e6, 2e6, 3e6, 4e6)
    coronal_base_density = 2e16 * scipy.constants.proton_mass
    stellar_radius = 695.510e6
    stellar_mass = 1.989e30

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        for _id, coronal_base_temperature in enumerate(coronal_base_temperatures):
            r = np.geomspace(stellar_radius, 215 * stellar_radius)  # 20*r_S)
            c = plt.rcParams['axes.prop_cycle'].by_key()['color'][_id]

            u, rho, r_sonic, u_sonic, rho_sonic = parker.parker_solution(r,
                                                                         coronal_temperature=coronal_base_temperature,
                                                                         coronal_base_density=coronal_base_density,
                                                                         stellar_radius=stellar_radius,
                                                                         stellar_mass=stellar_mass)

            plt.plot(r/stellar_radius, rho, color=c)
            plt.plot(r_sonic/stellar_radius, rho_sonic, 'o', color=c)

        plt.xlabel(r'Height over chromosphere [$R_{\star}$]')
        plt.ylabel('Density [kg/m3]')
        # plt.ylim((1e-2, 1e11))
        plt.yscale('log')
        # plt.xlim((r_S, 15*r_S))
        plt.grid(True)
        plt.savefig(pn.get())


def test_total_mass_flux(request):

    coronal_base_temperatures = (0.5e6, 0.75e6, 1e6, 1.5e6, 2e6, 3e6, 4e6)
    coronal_base_density = 2e16 * scipy.constants.proton_mass
    stellar_radius = 695.510e6
    stellar_mass = 1.989e30

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        fig, [ax1, ax2] = plt.subplots(2, 1)
        # ax2 = plt.twinx()
        for _id, coronal_base_temperature in enumerate(coronal_base_temperatures):
            r = np.geomspace(stellar_radius, 215 * stellar_radius)  # 20*r_S)
            c = plt.rcParams['axes.prop_cycle'].by_key()['color'][_id]

            u, rho, r_sonic, u_sonic, rho_sonic = parker.parker_solution(r,
                                                                         coronal_temperature=coronal_base_temperature,
                                                                         coronal_base_density=coronal_base_density,
                                                                         stellar_radius=stellar_radius,
                                                                         stellar_mass=stellar_mass)

            total_mass_flux = r**2 * u * rho
            total_mass_flux_sonic = r_sonic**2 * u_sonic * rho_sonic

            ax1.plot(r/stellar_radius, total_mass_flux, color=c)
            ax1.plot(r_sonic/stellar_radius, total_mass_flux_sonic, 'o', color=c)

            ax2.plot(r / stellar_radius,
                     total_mass_flux / stellar_mass * 3600 * 24 * 365.25,
                     color=c)

        ax2.set_ylabel('Mass loss [Msun/yr]')

        ax1.set_xlabel(r'Radius [$R_{\star}$]')
        ax1.set_ylabel('Total mass loss [kg/s]')

        for ax in (ax1, ax2):
            ax.set_yscale('log')
            ax.grid(True)
        plt.savefig(pn.get())


def test_total_mass_flux2(request):

    coronal_base_temperatures = np.geomspace(0.5e6, 30e6)
    coronal_base_number_densities = np.geomspace(1.5e14, 2e16, 4)

    stellar_radius = 695.510e6
    stellar_mass = 1.989e30

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        fig, ax = plt.subplots()
        for coronal_base_number_density in coronal_base_number_densities:
            radius_sonic, speed_sonic, density_sonic = parker.parker_sonic_point(coronal_base_temperatures,
                                                                                 coronal_base_density=coronal_base_number_density*scipy.constants.proton_mass)

            total_mass_flux = radius_sonic**2 * speed_sonic * density_sonic

            ax.plot(coronal_base_temperatures, total_mass_flux)


        ax.set_xlabel(r'Temperature [K]')
        ax.set_ylabel('Total mass loss [kg/s]')
        ax.set_yscale('log')
        plt.savefig(pn.get())
        plt.close()


        fig, ax = plt.subplots()
        for coronal_base_number_density in coronal_base_number_densities:
            radius_sonic, speed_sonic, density_sonic = parker.parker_sonic_point(coronal_base_temperatures,
                                                                                 coronal_base_density=coronal_base_number_density*scipy.constants.proton_mass)

            total_mass_flux = radius_sonic**2 * speed_sonic * density_sonic

            ax.plot(coronal_base_temperatures, total_mass_flux / stellar_mass * 3600 * 24 * 365.25)

        ax.set_title('Rompe')
        ax.set_xlabel(r'Temperature [K]')
        ax.set_ylabel('Total mass loss [Msun/yr]')
        ax.set_xscale('log')
        ax.set_yscale('log')

        plt.savefig(pn.get())
        plt.close()
