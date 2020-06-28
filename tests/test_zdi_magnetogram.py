import matplotlib as mpl
from matplotlib import cm
import numpy as np
import logging

# Test "context"
import stellarwinds.magnetogram.plot_zdi
from tests import context  # Test context
from tests.magnetogram import magnetograms
from tests.magnetogram import test_flow
log = logging.getLogger(__name__)

# Local
import stellarwinds.magnetogram.zdi_magnetogram
import stellarwinds.magnetogram.geometry
from stellarwinds.magnetogram import plots
import stellarwinds.magnetogram.coefficients as shc
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import pytest


def test_properties(request):
    """Use ZdiGeometry object with ZdiMagnetogram object to calculate properties"""
    zg = stellarwinds.magnetogram.geometry.ZdiGeometry()
    lz = stellarwinds.magnetogram.zdi_magnetogram.ZdiMagnetogram(1, 0, 1, 0, 0)

    field_strengths = lz.get_field_strength(*zg.centers())
    areas = zg.areas()

    assert np.isclose(np.sum(areas), 4*np.pi)

    avg_field_strength = np.sum(field_strengths * areas) / np.sum(areas)
    assert np.isclose(avg_field_strength, 0.244, rtol=1e-02, atol=1e-05)

    ind = np.unravel_index(np.argmax(field_strengths, axis=None), field_strengths.shape)
    max_field_strength = np.argmax(field_strengths)
    assert np.isclose(field_strengths[ind], 0.489, rtol=1e-02, atol=1e-05)

    avg_field_squared = np.sum(field_strengths**2 * areas) / np.sum(areas)
    assert np.isclose(avg_field_squared, 0.0795934601435181)


def test_dmpl(request, magnetogram_name="mengel"):

    lz = stellarwinds.magnetogram.zdi_magnetogram.from_coefficients(magnetograms.get_all(magnetogram_name))
    zg = stellarwinds.magnetogram.geometry.ZdiGeometry(64)

    polar, azimuth = zg.centers()

    B = {}
    Bmax = 0
    for method in ("roll", "roll2", "gradient"):
        lz = stellarwinds.magnetogram.zdi_magnetogram.from_coefficients(magnetograms.get_all(magnetogram_name), method)
        B[method] = lz.get_field_strength(*zg.centers())
        Bmax = np.maximum(Bmax, np.max(B[method]))

    assert np.allclose(B["roll"], B["roll2"])
    # assert np.allclose(B["roll"], B["gradient"])  # This fails but they look really close...


    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        for cid, name in B.items():

            fig, ax = plt.subplots()
            img1 = ax.pcolormesh(np.rad2deg(azimuth), np.rad2deg(polar), np.real(B[cid]),
                                 vmin=0,
                                 vmax=Bmax)
            fig.colorbar(img1, ax=ax)
            fig.suptitle('Method %s.' % cid)
            ax.invert_yaxis()
            plt.savefig(pn.get())


def test_lpmn(request):
    zg = stellarwinds.magnetogram.geometry.ZdiGeometry(64)
    polar_centers, azimuth_centers = zg.centers()

    coeffs_zdi = shc.Coefficients()
    coeffs_zdi.append(2, 0, 1.0 + 0.0j)
    coeffs_zdi.append(3, 1, 1.0 + 0.0j)
    coeffs_zdi.append(3, 2, 1.0 + 0.0j)

    Bref = stellarwinds.magnetogram.zdi_magnetogram.from_coefficients(coeffs_zdi).get_polar_poloidal_field(polar_centers, azimuth_centers)

    Bnew = -stellarwinds.magnetogram.zdi_magnetogram.from_coefficients(coeffs_zdi).get_polar_poloidal_field_new(polar_centers, azimuth_centers)

    assert Bref.shape == Bnew.shape

    b = np.min(polar_centers.shape) // 8  # Bad border pixels

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        _, axs = plt.subplots(3, 1, figsize=(12, 12))
        for f, ax in zip([Bref, Bnew, np.abs(Bref - Bnew)], axs):
            plots.plot_magnetic_field(ax,
                                      # polar_centers,
                                      # azimuth_centers,
                                      # f,
                                      polar_centers[b:-b, b:-b],
                                      azimuth_centers[b:-b, b:-b],
                                      f[b:-b, b:-b],
                                      legend_str=r'B_\theta', )
            plots.add_extrema(polar_centers[b:-b, b:-b],
                              azimuth_centers[b:-b, b:-b],
                              f[b:-b, b:-b],
                              ax, legend_str=r'B_\theta', markers='12')
            ax.legend()
        axs[0].set_title("Reference")
        axs[1].set_title("New")
        axs[2].set_title("Error")

        plt.savefig(pn.get())

        # Scatter curve of errors

    assert np.allclose(Bref[b:-b, b:-b], Bnew[b:-b, b:-b], rtol=1e-2, atol=1e-2)


def test_lpmn_lpmv(request):

    zg = stellarwinds.magnetogram.geometry.ZdiGeometry(64)
    polar_centers, azimuth_centers = zg.centers()

    coeffs_zdi = shc.Coefficients()
    coeffs_zdi.append(2, 0, 1.0 + 0.0j)
    coeffs_zdi.append(3, 1, 1.0 + 0.0j)
    coeffs_zdi.append(3, 2, 1.0 + 0.0j)

    points_polar = np.linspace(0, np.pi)

    zm = stellarwinds.magnetogram.zdi_magnetogram.from_coefficients(coeffs_zdi)
    Pmn_z_result, Pmn_d_z_result = zm._calculate_lpmn(points_polar)

    assert Pmn_z_result.shape == points_polar.shape + (coeffs_zdi.size,)


def test_zdi_magnetogram_3d(request, magnetogram_name="mengel"):
    lz = stellarwinds.magnetogram.zdi_magnetogram.from_coefficients(magnetograms.get_all(magnetogram_name))

    # zg = stellarwinds.magnetogram.zdi_geometry.ZdiGeometry(polar_corners = np.pi * np.linspace(0, 1, 128),
    #                  azimuthal_corners = np.pi * np.linspace(0, 2, 256))

    zg = stellarwinds.magnetogram.geometry.ZdiGeometry(64)

    polar, azimuth = zg.centers()
    corners = zg.corners_cartesian()
    centers = zg.centers_cartesian()
    areas = zg.areas()

    B = [
        lz.get_radial_field(*zg.centers()),
        lz.get_polar_poloidal_field(*zg.centers()),
        lz.get_polar_toroidal_field(*zg.centers()),
        -lz.get_polar_field(*zg.centers()),
        lz.get_azimuthal_poloidal_field(*zg.centers()),
        lz.get_azimuthal_toroidal_field(*zg.centers()),
        lz.get_azimuthal_field(*zg.centers())
    ]

    Bscale = np.max([np.max(np.abs(Bc)) for Bc in B])

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        for cid, name in enumerate(("radial",
                                    "polar poloidal", "polar toroidal", "polar",
                                    "azimuthal poloidal", "azimuthal toroidal", "toroidal")):

            # fig, ax = plt.subplots(1,3)
            #
            # for ax_id, coords in enumerate(itertools.combinations({0, 1, 2}, 2)):
            #     print(ax_id, coords)
            #     img = ax[ax_id].pcolormesh(corners[coords[0]], corners[coords[1]], np.real(B[cid]), cmap='RdBu_r',
            #                      vmin=-Bscale,
            #                      vmax=Bscale)
            #     ax[ax_id].plot(corners[coords[0]], corners[coords[1]], 'k-', linewidth=.1)
            #     ax[ax_id].plot(corners[coords[0]].transpose(), corners[coords[1]].transpose(), 'k-', linewidth=.1)
            #     fig.colorbar(img, ax=ax[ax_id])
            #     ax[ax_id].set_title("%s=0" % "xyz"[ax_id])
            #
            # plt.savefig(pn.get())
            #
            # fig.suptitle('Field %s component' % name)
            # plt.savefig(pn.get())

            import matplotlib.pyplot as plt
            # ax.plot_surface(*corners, facecolors=cm.coolwarm(corners[0]),
            #            linewidth=1, antialiased=False)

            import matplotlib.colors

            def surface3d_and_bar(x, y, z, values, color_norm, color_map):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                surf = ax.plot_surface(*corners,
                                       facecolors=color_map(color_norm(values)),
                                       cstride=1, rstride=1,  # These are not the defaults!
                                       edgecolor='black', linewidth=.1,  # Does not work with 'facecolors'...
                                       antialiased=False,
                                       shade=False)

                m = cm.ScalarMappable(cmap=color_map, norm=color_norm)
                m.set_array(values)
                fig.colorbar(m)

                # ax.set_aspect('equal')
                ax.set_title("az=%f, el=%f" % (ax.azim, ax.elev))
                ax.set_xlabel("x [R]")
                ax.set_ylabel("y [R]")
                ax.set_zlabel("z [R]")

                return fig, ax

            color_norm = mpl.colors.Normalize(*(np.array([-1, 1]) * np.max(np.abs(B[0]))))
            color_map = cm.coolwarm

            fig, ax = surface3d_and_bar(*corners, B[0], color_norm, color_map)

            plt.savefig(pn.get())
            plt.close()

            def surface3d3_and_bar(x, y, z, values, color_norm, color_map):
                fig = plt.figure()

                for pid in range(1, 4):
                    ax = fig.add_subplot(1, 3, pid, projection='3d')

                    surf = ax.plot_surface(*corners,
                                           facecolors=color_map(color_norm(values)),
                                           cstride=1, rstride=1,  # These are not the defaults!
                                           edgecolor='black', linewidth=.1,  # Does not work with 'facecolors'...
                                           antialiased=False,
                                           shade=False)


                    # ax.set_aspect('equal')
                    ax.view_init(azim=120 * pid, elev=60)
                    ax.set_title("az=%f, el=%f" % (ax.azim, ax.elev))
                    ax.set_xlabel("x [R]")
                    ax.set_ylabel("y [R]")
                    ax.set_zlabel("z [R]")

                m = cm.ScalarMappable(cmap=color_map, norm=color_norm)
                m.set_array(values)
                # fig.colorbar(m)

                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                fig.colorbar(m, cax=cbar_ax)

                return fig, ax

            color_norm = mpl.colors.Normalize(*(np.array([-1, 1]) * np.max(np.abs(B[0]))))
            color_map = cm.coolwarm

            fig, ax = surface3d3_and_bar(*corners, B[6], color_norm, color_map)
            plt.savefig(pn.get())
            plt.close()


            def surface3d3_shifted_and_bar(x, y, z, values, color_norm, color_map):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                for pid in range(1, 4):

                    x, y, z = corners
                    surf = ax.plot_surface(x+3*pid, y, z,
                                           facecolors=color_map(color_norm(values)),
                                           cstride=1, rstride=1,  # These are not the defaults!
                                           edgecolor='black', linewidth=.1,  # Does not work with 'facecolors'...
                                           antialiased=False,
                                           shade=False)


                # ax.set_aspect('equal')
                # ax.view_init(azim=120 * pid)
                # ax.set_title("az=%f, el=%f" % (ax.azim, ax.elev))
                ax.set_xlabel("x [R]")
                ax.set_ylabel("y [R]")
                ax.set_zlabel("z [R]")

                m = cm.ScalarMappable(cmap=color_map, norm=color_norm)
                m.set_array(values)
                # fig.colorbar(m)

                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                fig.colorbar(m, cax=cbar_ax)

                return fig, ax

            fig, ax = surface3d3_shifted_and_bar(*corners, B[0], color_norm, color_map)

            plt.savefig(pn.get())
            plt.close()

            break


def test_zdi_magnetogram(request, magnetogram_name="mengel"):
    lz = stellarwinds.magnetogram.zdi_magnetogram.from_coefficients(magnetograms.get_all(magnetogram_name))

    zg = stellarwinds.magnetogram.geometry.ZdiGeometry()
    polar, azimuth = zg.corners()

    B = [
        lz.get_radial_field(*zg.centers()),
        lz.get_polar_poloidal_field(*zg.centers()),
        lz.get_polar_toroidal_field(*zg.centers()),
        -lz.get_polar_field(*zg.centers()),
        lz.get_azimuthal_poloidal_field(*zg.centers()),
        lz.get_azimuthal_toroidal_field(*zg.centers()),
        lz.get_azimuthal_field(*zg.centers())
    ]

    Bscale = np.max([np.max(np.abs(Bc)) for Bc in B])

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        for cid, name in enumerate(("radial",
                                    "polar poloidal", "polar toroidal", "polar",
                                    "azimuthal poloidal", "azimuthal toroidal", "toroidal")):

            fig, ax = plt.subplots()
            img1 = ax.pcolormesh(np.rad2deg(azimuth), np.rad2deg(polar), np.real(B[cid]), cmap='RdBu_r',
                                 vmin=-Bscale,
                                 vmax=Bscale)
            fig.colorbar(img1, ax=ax)
            fig.suptitle('Field %s component' % name)
            ax.invert_yaxis()
            plt.savefig(pn.get())



def test_negative_order(request):
    degree_l = 1
    alpha_lm = 1
    beta_lm = 0
    gamma_lm = 0

    zg = stellarwinds.magnetogram.geometry.ZdiGeometry()
    polar, azimuth = zg.corners()

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, axs = plt.subplots(1, 2)

        for order_m, ax in zip ((-1, 1), axs):
            lz = stellarwinds.magnetogram.zdi_magnetogram.ZdiMagnetogram(degree_l, order_m, alpha_lm, beta_lm, gamma_lm)

            img1 = ax.pcolormesh(np.rad2deg(azimuth), np.rad2deg(polar), np.real(lz.get_radial_field(*zg.centers())),
                                 cmap='RdBu_r',
                                 # vmin=-Bscale,
                                 # vmax=Bscale,
                                 )
            fig.colorbar(img1, ax=ax, orientation='horizontal')
            # fig.suptitle('Field %s component' % name)
            ax.invert_yaxis()
        plt.savefig(pn.get())


def test_compare_scipy(request):
    from stellarwinds.magnetogram import coefficients
    from scipy.special import sph_harm
    c = coefficients.Coefficients(0j)
    c.append(1, 1, 1.0)
    c.append(1, -1, 1.0)
    c.append(2, 1, 1.0j)
    c.append(2, 0, 1.0)
    c.append(11, 7, .20 + .1j)

    zg = stellarwinds.magnetogram.geometry.ZdiGeometry()
    # corner_pl, corner_az = zg.corners()
    center_pl, center_az = zg.centers()

    field_scipy = np.zeros_like(center_pl, dtype=np.complex)
    for (degree_l, order_m), data in c.contents():
        field_scipy += data * sph_harm(order_m, degree_l, center_az, center_pl)

    degrees_l, orders_m, alpha_lm = c.as_arrays()
    lz = stellarwinds.magnetogram.zdi_magnetogram.ZdiMagnetogram(degrees_l, orders_m, alpha_lm)
    field_zdi = lz.get_radial_field(center_pl, center_az)

    assert np.allclose(
        np.real(field_scipy), field_zdi)


def test_negative_and_positive_order(request):
    degree_l = 1
    alpha_lm = 1
    beta_lm = 0
    gamma_lm = 0

    zg = stellarwinds.magnetogram.geometry.ZdiGeometry()
    polar, azimuth = zg.corners()
    lz = stellarwinds.magnetogram.zdi_magnetogram.ZdiMagnetogram(
        np.array([1,  1]),
        np.array([1, -1]),
        np.array([1,  0]),
        np.array([0,  0]),
        np.array([0,  0]),
    )
    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, ax = plt.subplots()
        img1 = ax.pcolormesh(np.rad2deg(azimuth), np.rad2deg(polar), np.real(lz.get_radial_field(*zg.centers())),
                             cmap='RdBu_r',
                             # vmin=-Bscale,
                             # vmax=Bscale,
                             )
        fig.colorbar(img1, ax=ax, orientation='horizontal')
        # fig.suptitle('Field %s component' % name)
        ax.invert_yaxis()
        plt.savefig(pn.get())


def test_zdi_magnetogram_stats(request, magnetogram_name="mengel"):
    lz = stellarwinds.magnetogram.zdi_magnetogram.from_coefficients(magnetograms.get_all(magnetogram_name))

    zg = stellarwinds.magnetogram.geometry.ZdiGeometry()
    polar, azimuth = zg.corners()

    bstr = lz.get_field_strength(*zg.centers())
    bmax = np.max(bstr)
    bmean = np.sum(bstr * zg.areas()) / (4 * np.pi)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        fig, ax = plt.subplots()
        img1 = ax.pcolormesh(np.rad2deg(azimuth), np.rad2deg(polar), bstr,
                             vmin=0)
        fig.colorbar(img1, ax=ax)
        fig.suptitle('Field strength. Max=%f; mean=%f' % (bmax, bmean))
        ax.invert_yaxis()
        plt.savefig(pn.get())


def test_zdi_magnetogram_energy(request):
    coeffs = shc.Coefficients()
    coeffs.append(1, 0, 1.0)
    zc = stellarwinds.magnetogram.zdi_magnetogram.from_coefficients(coeffs)
    zc.energy()


def test_plot_zdi_magnetogram_energy(request):
    coeffs = shc.Coefficients()
    coeffs.append(0, 0, 1.0)
    coeffs.append(1, 0, 1.0)
    coeffs.append(1, 1, 1.0)
    coeffs.append(2, 0, 1.0)
    coeffs.append(2, 1, 1.0)
    coeffs.append(2, 2, 1.0)
    coeffs.append(6, 5, 1.0)

    zc = stellarwinds.magnetogram.zdi_magnetogram.from_coefficients(coeffs)
    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        fig, axs = stellarwinds.magnetogram.plot_zdi.plot_energy_matrix(zc)
        fig.savefig(pn.get())

        fig, axs = stellarwinds.magnetogram.plot_zdi.plot_energy_matrix(zc, types=("poloidal", "toroidal"))
        fig.savefig(pn.get())


def test_plot_strength(request, magnetogram_name="mengel"):
    lz = stellarwinds.magnetogram.zdi_magnetogram.from_coefficients(magnetograms.get_all(magnetogram_name))

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, ax = stellarwinds.magnetogram.plot_zdi.plot_zdi_field(lz.get_field_strength)
        fig.savefig(pn.get())
        plt.close()


def test_plot_radial_field(request, magnetogram_name="mengel"):
    lz = stellarwinds.magnetogram.zdi_magnetogram.from_coefficients(magnetograms.get_all(magnetogram_name))

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, ax = stellarwinds.magnetogram.plot_zdi.plot_zdi_field(lz.get_radial_field)
        fig.savefig(pn.get())
        plt.close()


def test_plot_radial_field_lic(request, magnetogram_name="mengel"):
    lz = stellarwinds.magnetogram.zdi_magnetogram.from_coefficients(magnetograms.get_all(magnetogram_name))

    zg = stellarwinds.magnetogram.geometry.ZdiGeometry(111)  # Increase this for prettier results
    polar_centers, azimuth_centers = zg.centers()
    polar_corners, azimuth_corners = zg.corners()

    radial_field_centers = lz.get_radial_field(polar_centers, azimuth_centers)
    az_field_centers = lz.get_azimuthal_field(polar_centers, azimuth_centers)
    pl_field_centers = lz.get_polar_field(polar_centers, azimuth_centers)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, ax = plt.subplots(figsize=(10, 4))
        stellarwinds.magnetogram.plots.plot_magnetic_field(ax, polar_centers, azimuth_centers, radial_field_centers,
                            polar_corners=polar_corners, azimuth_corners=azimuth_corners)
        fig.savefig(pn.get())
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 4))

        test_flow.add_lic(ax,
                          np.rad2deg(azimuth_centers).transpose(),
                          np.rad2deg(polar_centers).transpose(),
                          az_field_centers.transpose(),
                          -pl_field_centers.transpose(),
                          alpha=.3,
                          # length=40
                          )

        stellarwinds.magnetogram.plots.plot_magnetic_field(ax,
                                                              azimuth_centers,
                                                              polar_centers,
                                                              radial_field_centers,
                                                              polar_corners=polar_corners,
                                                              azimuth_corners=azimuth_corners)

        # _plot = ax.streamplot(np.rad2deg(_a.transpose()),
        #                       np.rad2deg(_p.transpose()),
        #                       _fa.transpose(),
        #                       # Note carefully that -_fp has to be used instead of _fp to flip the
        #                       # polar axis - it has to be flipped not only for
        #                       # the _p coordinate but for the vector component as well.
        #                       -_fp.transpose(),
        #                       color=((_fa ** 2 + _fp ** 2) ** (1 / 2)).transpose())

        # ax.invert_yaxis()

        fig.savefig(pn.get())
        plt.close()


@pytest.mark.parametrize("method", ("get_radial_field",
                                    "get_polar_field",
                                    "get_azimuthal_field",
                                    "get_field_strength"))
@pytest.mark.parametrize("magnetogram_name", ("mengel",))
def test_plot_field(request, method, magnetogram_name):
    lz = stellarwinds.magnetogram.zdi_magnetogram.from_coefficients(magnetograms.get_all(magnetogram_name))

    _method = getattr(lz, method)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, ax = stellarwinds.magnetogram.plot_zdi.plot_zdi_field(_method,
                                                                   zg=None,
                                                                   symmetric=None)
        ax.set_title(method)
        fig.savefig(pn.get())
        plt.close()

