import numpy as np
import scipy.special
import logging


log = logging.getLogger(__name__)

# Test "context"
from tests import context  # Test context
import pytest
from tests import magnetograms

# Local
import starwinds_magnetogram.pfss_magnetogram as pfss_stanford
from starwinds_magnetogram import plots  #TODO: remove this.
from starwinds_magnetogram import plot_pfss
from starwinds_magnetogram import geometry


def test_r_l_values(request):
    rss = 2
    rs = 1

    r = np.linspace(rs, rss)

    for deg_l in range(10):
        r_l, dr_l = pfss_stanford.r_l(deg_l, r, rs, rss)
        assert dr_l[0] == -1, "Expected analytical result -1"



def test_r_l_plot(request):
    rss = 2
    rs = 1

    r = np.linspace(rs, rss)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, axs = plt.subplots(2, 1)
        for deg_l in range(10):
            r_l, dr_l = pfss_stanford.r_l(deg_l, r, rs, rss)
            axs[0].plot(r, r_l)
            axs[1].plot(r, dr_l)
            axs[1].set_xlabel('radius')
            axs[0].set_ylabel('R(r)')
            axs[1].set_ylabel('d R(r) / dr')
            axs[0].set_title("rss=%f" % rss)
        plt.savefig(pn.get())
        plt.close(fig)


@pytest.mark.skip(reason="Never worked...")
def test_reference(request,
                   magnetogram_name="mengel"):

    magnetogram = magnetograms.get_radial(magnetogram_name)
    degree_l, order_m, alpha_lm = magnetogram.as_arrays(include_unset=False)
    polar, azimuth = geometry.ZdiGeometry().centers()

    B1 = evaluate_real_magnetogram_stanford_pfss_reference(
        degree_l,
        order_m,
        np.real(alpha_lm),
        np.imag(alpha_lm),
        polar, azimuth)


    B2 = pfss_stanford.evaluate_spherical(
        magnetogram,
        1, polar, azimuth)

    assert(np.allclose(B2[0], B1[0]))
    log.debug("Implementations match for B_radial.")

    assert(np.allclose(B2[1], B1[1]))
    log.debug("Implementations match for B_polar.")

    log.debug('B1 azimuth component range [%g, %g]' % (np.min(B1[2]), np.max(B1[2])))
    log.debug('B2 azimuth component range [%g, %g]' % (np.min(B2[2]), np.max(B2[2])))

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        for cid, name in enumerate(("radial", "polar", "azimuth")):
            fig, axs = plt.subplots(2, 2)

            img1 = axs[0, 0].pcolormesh(np.rad2deg(azimuth), np.rad2deg(polar), B1[cid], cmap='RdBu_r')
            fig.colorbar(img1, ax=axs[0, 0])

            img2 = axs[0, 1].pcolormesh(np.rad2deg(azimuth), np.rad2deg(polar), B2[cid], cmap='RdBu_r')
            fig.colorbar(img2, ax=axs[0, 1])

            img10 = axs[1, 0].pcolormesh(np.rad2deg(azimuth), np.rad2deg(polar), B2[cid]-B1[cid])
            fig.colorbar(img10, ax=axs[1, 0])

            img11 = axs[1, 1].pcolormesh(np.rad2deg(azimuth), np.rad2deg(polar), B2[cid]/B1[cid], vmin=-2, vmax=2)
            fig.colorbar(img11, ax=axs[1, 1])

            fig.suptitle('Field %s component' % name)

            for ax in axs.ravel():
                ax.invert_xaxis()
                ax.invert_yaxis()

            plt.savefig(pn.get())
            plt.close(fig)
    # Why do these not match??
    assert(np.allclose(B2[2], B1[2]))


def test_legendre_single(request):
    degree_l, order_m = 5, 2
    degree_l, order_m = np.atleast_1d(degree_l, order_m)

    points_polar = np.linspace(0, np.pi, 200)
    p0, dp0 = pfss_stanford.theta_lm(degree_l, order_m, points_polar)

    p1, dp1 = pfss_stanford.calculate_all_theta(degree_l, order_m, points_polar, scipy=True)
    p1 = p1.flatten()
    dp1 = dp1.flatten()

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, axs = plt.subplots(2, 1)
        ax = axs[0]
        ax.plot(points_polar, p0-p1)
        # ax.plot(points_polar, p1)

        ax = axs[1]
        ax.plot(points_polar, dp0-dp1)
        # ax.plot(points_polar, dp1)

        plt.savefig(pn.get())
        plt.close(fig)

    assert np.allclose(p0, p1)  # These should work
    assert not np.allclose(dp0, dp1)  # But these are inaccurate, that's why to use lpmn.
    assert np.allclose(dp0, dp1, atol=.1)  # This should still work.


@pytest.mark.skip(reason="Skip since some values differ. TODO investigate this.")
def test_legendre_multiple(request):
    degree_l = np.array([1, 1, 5, 20])
    order_m = np.array([0, 1, 3, 20])

    points_polar = np.linspace(0, np.pi, 400)

    p1, dp1 = pfss_stanford.calculate_all_theta(degree_l, order_m, points_polar, scipy=True)

    assert p1.shape == (len(points_polar), len(degree_l))
    assert dp1.shape == (len(points_polar), len(degree_l))

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, axs = plt.subplots(2, 1)

        for id_, (l, m) in enumerate(zip(degree_l, order_m)):
            ax = axs[0]

            p0, dp0 = pfss_stanford.theta_lm(l, m, points_polar)
            # line, = ax.plot(points_polar, p0)
            # ax.plot(points_polar, p1[..., id_], 'o:', color=line.get_color())
            ax.plot(points_polar, p0-p1[..., id_])

            ax = axs[1]
            ax.plot(points_polar, dp0-dp1[..., id_])

        plt.savefig(pn.get())
        plt.close(fig)

    for id_, (l, m) in enumerate(zip(degree_l, order_m)):
        p0, dp0 = pfss_stanford.theta_lm(l, m, points_polar)
        assert np.allclose(p0, p1[..., id_])  # These should work


@pytest.mark.parametrize("points_shape", ((1,), (2,), (2, 3), (2, 3, 5), (5, 8, 2, 2, 1, 3)))
@pytest.mark.parametrize("magnetogram_name", ("mengel",))
def test_evaluate_cartesian(request,
                            points_shape,
                            magnetogram_name):

    coeffs = magnetograms.get_radial(magnetogram_name)

    px = 1 + np.random.rand(*points_shape)
    py = 1 + np.random.rand(*points_shape)
    pz = 1 + np.random.rand(*points_shape)

    fr, fp, fa, fx, fy, fz = pfss_stanford.evaluate_cartesian(
        coeffs,
        px, py, pz)

    assert fr.shape == points_shape


@pytest.mark.parametrize("points_shape", ((1,), (2,), (2, 3), (2, 3, 5), (5, 8, 2, 2, 1, 3)))
@pytest.mark.parametrize("magnetogram_name", ("mengel",))
def test_evaluate_spherical(request,
                            points_shape,
                            magnetogram_name):

    coeffs = magnetograms.get_radial(magnetogram_name)

    pr = 1 + np.random.rand(*points_shape)
    pp = np.pi * np.random.rand(*points_shape)
    pa = 2 * np.pi * np.random.rand(*points_shape)

    fr, fp, fa = pfss_stanford.evaluate_spherical(
        coeffs,
        pr, pp, pa)

    assert fr.shape == points_shape


def test_plot_pfss_equirectangular(request,
                           magnetogram_name="mengel"):

    magnetogram = magnetograms.get_radial(magnetogram_name)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, axs = plot_pfss.plot_components(magnetogram)
        plt.savefig(pn.get())
        plt.close(fig)


@pytest.mark.parametrize("rss", (1.5, 3.0, 5.0))
def test_plot_pfss_slice(request,
                         rss,
                         magnetogram_name="dipole"):

    magnetogram = magnetograms.get_radial(magnetogram_name)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, axs = plot_pfss.plot_slice(magnetogram, radius_source_surface=rss)
        fig.savefig(pn.get())
        plt.close(fig)


def test_plot_pfss_magnitudes(request,
                magnetogram_name="mengel"):

    _geometry = geometry.ZdiGeometry()
    coefficients = magnetograms.get_radial(magnetogram_name)
    polar, azimuth = geometry.ZdiGeometry().centers()

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        fig, axs = plt.subplots(1, 2)

        for radius, ax in zip((1, 3), axs):
            field_rpa = pfss_stanford.evaluate_spherical(
                coefficients,
                radius, polar, azimuth)
            field_magnitude = np.sqrt(np.sum([f**2 for f in field_rpa], axis=0))
            img = plots.plot_equirectangular(_geometry, field_magnitude, ax, cmap='viridis')
            fig.colorbar(img, ax=ax, orientation='horizontal')
            ax.set_title("Bmag at r=%2.1f" % radius)

        plt.savefig(pn.get())
        plt.close(fig)

def test_plot_pfss_magnitudes_ss(request,
                magnetogram_name="mengel"):
    """Test different source surface radii"""

    _geometry = geometry.ZdiGeometry()
    coefficients = magnetograms.get_radial(magnetogram_name)
    polar, azimuth = geometry.ZdiGeometry().centers()

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        field_rpa_by_ss = []

        for radius_source_surface in [1.5, 2]:
            fig, axs = plt.subplots(1, 2)

            for radial, ax in zip((1, radius_source_surface), axs):
                field_rpa = pfss_stanford.evaluate_spherical(
                    coefficients,
                    radial, polar, azimuth,
                    radius_source_surface=radius_source_surface)

                field_rpa_by_ss.append(field_rpa)
                field_magnitude = np.sqrt(np.sum([f**2 for f in field_rpa], axis=0))
                img = plots.plot_equirectangular(_geometry, field_magnitude, ax, cmap='viridis')
                fig.colorbar(img, ax=ax, orientation='horizontal')
                ax.set_title("Bmag at r=%2.1f, rss=%2.1f" % (radial, radius_source_surface))

            plt.savefig(pn.get())
            plt.close(fig)


def test_plot_pfss_quiver(request,
                          magnetogram_name="mengel"):

    _geometry = geometry.ZdiGeometry()
    magnetogram = magnetograms.get_radial(magnetogram_name)
    polar, azimuth = geometry.ZdiGeometry().centers()

    field_rpa = pfss_stanford.evaluate_spherical(
        magnetogram,
        1, polar, azimuth)

    field_pa_mag = np.sqrt(field_rpa[1] ** 2 + field_rpa[2] ** 2)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        fig, ax = plt.subplots()
        hstride = int(2)
        vstride = int(hstride / 2)
        img = plots.plot_equirectangular(_geometry, field_pa_mag, ax, cmap='viridis')
        fig.colorbar(img, ax=ax, orientation='horizontal')

        ax.quiver(np.rad2deg(azimuth[::hstride, ::vstride]),
                  np.rad2deg(polar[::hstride, ::vstride]),
                  field_rpa[2][::hstride, ::vstride],
                  field_rpa[1][::hstride, ::vstride])

        ax.set_title("Surface tangential field strength")

        plt.savefig(pn.get())
        plt.close(fig)


def test_plot_pfss_streamtraces(request,
                                magnetogram_name="mengel"):

    magnetogram = magnetograms.get_radial(magnetogram_name)
    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, ax = plot_pfss.plot_streamtraces(magnetogram)
        fig.savefig(pn.get())
        plt.close(fig)


# TODO which reference implementation is this? Do we even have a reference implementation?
def evaluate_real_magnetogram_stanford_pfss_reference(degree_l, order_m, cosine_coefficients_g, sine_coefficients_h,
                                                      points_polar, points_azimuth, radius=None, r0=1, rss=3):
    if radius is None:
        radius = r0

    assert np.min(order_m) >= 0, "Stanford PFSS expects only positive orders (TBC)."
    field_radial = np.zeros_like(points_azimuth)
    field_polar = np.zeros_like(field_radial)
    field_azimuthal = np.zeros_like(field_radial)

    for row_id in range(len(degree_l)):

        deg_l = degree_l[row_id]
        ord_m = order_m[row_id]
        g_lm = cosine_coefficients_g[row_id]
        h_lm = sine_coefficients_h[row_id]

        p_lm = scipy.special.lpmv(ord_m, deg_l, np.cos(points_polar))

        if True:  # Apply correction
            # https://en.wikipedia.org/wiki/Spherical_harmonics#Condon%E2%80%93Shortley_phase
            # https://en.wikipedia.org/wiki/Spherical_harmonics#Conventions
            d0 = 0 + (ord_m == 0)
            p_lm *= (-1) ** ord_m * np.sqrt(scipy.special.factorial(deg_l - ord_m) / scipy.special.factorial(deg_l + ord_m)) * np.sqrt(2 - d0)

        # Estimate DPml
        DPml = (p_lm - np.roll(p_lm, 1)) / (np.cos(points_polar) - np.roll(np.cos(points_polar), 1)) * np.sin(points_polar)

        fixed = (r0 / radius) ** (deg_l + 2) / (deg_l + 1 + deg_l * (r0 / rss) ** (2 * deg_l + 1))

        field_radial += p_lm * (g_lm * np.cos(ord_m * points_azimuth) + h_lm * np.sin(ord_m * points_azimuth)) * (
                deg_l + 1 + deg_l * (radius / rss) ** (2 * deg_l + 1)) * fixed
        field_polar -= DPml * (g_lm * np.cos(ord_m * points_azimuth) + h_lm * np.sin(ord_m * points_azimuth)) * (
                1 - (radius / rss) ** (2 * deg_l + 1)) * fixed
        field_azimuthal -= p_lm * (g_lm * np.sin(ord_m * points_azimuth) - h_lm * np.cos(ord_m * points_azimuth)) * (
                1 - (radius / rss) ** (2 * deg_l + 1)) * fixed

    return field_radial, field_polar, field_azimuthal


def field_line_integral(points, coeffs, rs=1, rss=3, dir='both'):

    def _dxds(points):
        px, py, pz = points
        _r, _p, _a, *field_xyz = pfss_stanford.evaluate_cartesian(
            coeffs,
            px, py, pz,
            radius_star=rs,
            radius_source_surface=rss)

        field_xyz = np.stack(field_xyz)

        lengths = np.sum(field_xyz ** 2, axis=0) ** .5

        assert not np.any(np.isnan(field_xyz))

        with np.errstate(divide='ignore', invalid='ignore'):
            dxds_ = field_xyz / lengths
            dxds_[:, lengths == 0] = 0  # Remove NaNs from length==0

        return dxds_

    _dxds(points)

    ds = .05
    steps = 100


    trajectories = np.empty(shape=(steps,)+points.shape)
    trajectories.fill(np.nan)
    trajectories[0] = points

    if dir == 'forward':
        for id_ in range(1, trajectories.shape[0]):
            dxds = _dxds(trajectories[id_ - 1])
            trajectories[id_] = trajectories[id_ - 1] + dxds * ds
    elif dir == 'backwards':
        for id_ in range(1, trajectories.shape[0]):
            dxds = _dxds(trajectories[id_ - 1])
            trajectories[id_] = trajectories[id_ - 1] - dxds * ds
    else:
        tbw = np.empty_like(trajectories)
        tbw.fill(np.nan)
        tbw[-1] = points
        for id_ in reversed(range(1, tbw.shape[0])):
            dxds = _dxds(tbw[id_])
            tbw[id_-1] = tbw[id_] - dxds * ds

        for id_ in range(1, trajectories.shape[0]):
            dxds = _dxds(trajectories[id_ - 1])
            trajectories[id_] = trajectories[id_ - 1] + dxds * ds

        trajectories = np.concatenate((tbw, trajectories), axis=0)

    return trajectories

@pytest.mark.parametrize("rss", (1.5, 3.0, 5.0))
def test_fieldlines_3d(request, rss):

    rs = 1

    import starwinds_magnetogram.coefficients as shc
    from starwinds_magnetogram import geometry
    import matplotlib as mpl

    coeffs = shc.Coefficients()
    coeffs.append(0, 0, 0.0)
    coeffs.append(1, 0, 0.1)
    coeffs.append(2, 1, 0.7)

    x, y, z = geometry.ZdiGeometry(20).centers_cartesian()

    points = np.stack((x, y, z))
    trajectories = field_line_integral(points, coeffs, rs=rs, rss=rss)

    radial_distances = np.sum(trajectories**2, axis=1, keepdims=True)**.5
    # Don't plot past six radii
    mask = 1/(radial_distances < 6)
    trajectories = trajectories * mask

    # For coloring the lines
    max_radial_distances = np.max(radial_distances, axis=0).squeeze()

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(0, trajectories.shape[2]):
            for j in range(0, trajectories.shape[3]):
                col = mpl.cm.jet((max_radial_distances[i, j] - rs)/(rss-rs))

                ax.plot(trajectories[:, 0, i, j],
                        trajectories[:, 1, i, j],
                        trajectories[:, 2, i, j],
                        color=col)
                ax.set_title(f"Rss={rss:2.1}")
        fig.savefig(pn.get())
        plt.close(fig)

def test_fieldlines_3d_flat_points(request):

    rs = 1
    rss = 3

    import starwinds_magnetogram.coefficients as shc
    from starwinds_magnetogram import geometry
    import matplotlib as mpl
    from starwinds_magnetogram import fibonacci_sphere
    coeffs = shc.Coefficients()
    coeffs.append(0, 0, 0.0)
    # coeffs.append(1, 0, 0.1)
    coeffs.append(5, 3, 0.7)
    coeffs.append(5, 4, 0.7)

    x, y, z = geometry.ZdiGeometry(20).centers_cartesian()
    res = fibonacci_sphere.fibonacci_sphere(1000)
    x = res[:, 0]
    y = res[:, 1]
    z = res[:, 2]

    points = np.stack((x, y, z))
    trajectories = field_line_integral(points, coeffs, rs=rs, rss=rss)

    # Don't plot past source surface radius
    radial_distances = np.sum(trajectories**2, axis=1, keepdims=True)**.5
    with np.errstate(divide='ignore'):
        mask = 1/(radial_distances < rss)
    trajectories = trajectories * mask

    # For coloring the lines
    max_radial_distances = np.max(radial_distances, axis=0).squeeze()

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(0, trajectories.shape[2]):

            col = mpl.cm.jet((max_radial_distances[i] - rs)/(rss-rs))

            ax.plot(trajectories[:, 0, i],
                    trajectories[:, 1, i],
                    trajectories[:, 2, i],
                    color=col)
        fig.savefig(pn.get())
        plt.close(fig)


def single_field_line_integral(points, coeffs, rs=1, rss=3, dir='both'):

    def _dxds(points):
        px, py, pz = points
        _r, _p, _a, *field_xyz = pfss_stanford.evaluate_cartesian(
            coeffs,
            px, py, pz,
            radius_star=rs,
            radius_source_surface=rss)

        field_xyz = np.stack(field_xyz)

        lengths = np.sum(field_xyz ** 2, axis=0) ** .5

        assert not np.any(np.isnan(field_xyz))

        dxds_ = field_xyz / lengths
        dxds_[:, lengths == 0] = 0  # Remove NaNs from length==0

        return dxds_

    _dxds(points)

import starwinds_magnetogram.coefficients as shc
from starwinds_magnetogram import geometry
import matplotlib as mpl
from starwinds_magnetogram import fibonacci_sphere
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D


def to_uniform(sol_fwd, sol_bwd):

    uniform = []
    for fwd, bwd in zip(sol_fwd, sol_bwd):
        t_fwd = np.linspace(0, fwd.t[-1])
        y_fwd = fwd.sol(t_fwd)

        t_bwd = np.linspace(bwd.t[-1], 0)
        y_bwd = bwd.sol(t_bwd)

        assert np.allclose(y_bwd[:, -1], y_fwd[:, 0])

        y = np.hstack((y_bwd[:, :-2], y_fwd))
        uniform.append(y)

    return np.stack(uniform)

def plot_uniform(ax, uniform):
    rmax = np.max(uniform)
    rmin = np.min(uniform)

    for y in uniform:
        col = mpl.cm.viridis((np.max(y[0, :]) - rmin) / (rmax - rmin))
        ax.plot(*y[1:], '-', color=col)


def test_single_line(request):

    r = 1
    radius_star = 1
    radius_source_surface = 3
    trange = [0, 1000]

    n_points = 300

    coeffs = shc.Coefficients()
    coeffs.append(0, 0, 0.0)
    coeffs.append(1, 0, 0.1)
    coeffs.append(5, 3, 0.7)
    coeffs.append(5, 4, 0.7)

    def _dyds(t, rxyz):

        pr, px, py, pz = rxyz

        # assert np.isclose(pr**2, px**2 + py**2 + pz**2)

        drpaxyz = pfss_stanford.evaluate_cartesian(
            coeffs,
            px, py, pz,
            radius_star=radius_star,
            radius_source_surface=radius_source_surface)

        drpaxyz = np.stack(drpaxyz)

        strength = np.sum(drpaxyz[3:] ** 2, axis=0) ** .5

        assert not np.any(np.isnan(drpaxyz))

        dyds = drpaxyz[[0, 3, 4, 5]]
        with np.errstate(divide='ignore', invalid='ignore'):
            dyds = dyds / strength
            dyds[:, strength == 0] = 0  # Remove NaNs from length==0

        # Shape should be (3, 1); remove last dimension
        return dyds[:, 0]

    def _dyds_backwards(t, rxyz):
        return -1 * _dyds(t, rxyz)
    
    def _reach_rss(t, y):
        return y[0] - radius_source_surface
    _reach_rss.direction = 1  # Trigger when going from negative to positive
    _reach_rss.terminal = True

    def _reach_rs(t, y):
        return y[0] - radius_star
    _reach_rs.direction = -1  # Trigger when going from positive to negative
    _reach_rs.terminal = True


    points_rxyz = np.empty(shape=(n_points, 4))
    points_rxyz[:, 1:] = r * fibonacci_sphere.fibonacci_sphere(n_points)
    points_rxyz[:, 0] = np.sum(points_rxyz[:, 1:] ** 2, axis=1)**.5

    sols_fwd = []
    for fl_id in range(points_rxyz.shape[0]):
        point_rxyz = points_rxyz[fl_id, :]
        sol = solve_ivp(_dyds, trange, point_rxyz,
                        events=(_reach_rs, _reach_rss),
                        dense_output=True)
        sols_fwd.append(sol)

    sols_bwd = []
    for fl_id in range(points_rxyz.shape[0]):
        point_rxyz = points_rxyz[fl_id, :]
        sol = solve_ivp(_dyds_backwards, trange, point_rxyz,
                        events=(_reach_rs, _reach_rss),
                        dense_output=True)
        sols_bwd.append(sol)

    uniform = to_uniform(sols_fwd, sols_bwd)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for sols in zip(sols_fwd, sols_bwd):

            rxyz = sol.y
            line, *_ = ax.plot([rxyz[1, 0]], [rxyz[2, 0]], [rxyz[3, 0]], 'o')  # Starting point
            color = line.get_color()

            for sol in sols:
                rxyz = sol.y
                ax.plot(rxyz[1, :],
                        rxyz[2, :],
                        rxyz[3, :], ':', color=color)

                tmax = sol.t[-1]
                trange = np.linspace(0, tmax)
                rxyz = sol.sol(trange)
                ax.plot(rxyz[1, :],
                        rxyz[2, :],
                        rxyz[3, :], color=color)

                # e0_rxyz = sol.y_events[0]
                # if e0_rxyz.size > 0:
                #     ax.plot(e0_rxyz[:, 1], e0_rxyz[:, 2], e0_rxyz[:, 3], 'v', color=color)
                #
                # e1_rxyz = sol.y_events[1]
                # if e1_rxyz.size > 0:
                #     ax.plot(e1_rxyz[:, 1], e1_rxyz[:, 2], e1_rxyz[:, 3], '^', color=color)

        fig.savefig(pn.get())
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_uniform(ax, uniform)
        fig.savefig(pn.get())
        plt.close(fig)

@pytest.mark.skip("This is very slow and may not work as intended.")
def test_single_line_spherical(request):
    rs = 1
    rss = 3

    def _dyds(t, rpa):

        px, py, pz = rpa

        _r, _p, _a, *_ = pfss_stanford.evaluate_cartesian(
            coeffs,
            px, py, pz,
            radius_star=rs,
            radius_source_surface=rss)

        field_rpa = np.stack((_r, _p, _a))

        lengths = np.sum(field_rpa ** 2, axis=0) ** .5

        assert not np.any(np.isnan(field_rpa))

        dfds_ = field_rpa / lengths
        dfds_[:, lengths == 0] = 0  # Remove NaNs from length==0

        # Shape should be (3, 1); remove last dimension
        return dfds_[:, 0]

    def _reach_rss(t, y):
        return not np.any(y > rss)

    _reach_rss.direction = 1
    _reach_rss.terminal = True

    import starwinds_magnetogram.coefficients as shc
    from starwinds_magnetogram import geometry
    import matplotlib as mpl
    from starwinds_magnetogram import fibonacci_sphere
    from scipy.integrate import solve_ivp

    coeffs = shc.Coefficients()
    coeffs.append(0, 0, 0.0)
    # coeffs.append(1, 0, 0.1)
    coeffs.append(5, 3, 0.7)
    coeffs.append(5, 4, 0.7)

    trange = [0, 1000]

    res = fibonacci_sphere.fibonacci_sphere(30)

    sols = []
    for fl_id in range(res.shape[0]):
        rpa = res[fl_id, :]
        sol = solve_ivp(_dyds, trange, rpa, events=_reach_rss)
        sols.append(sol)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for sol in sols:
            ax.plot(sol.y[0, :],
                    sol.y[1, :],
                    sol.y[2, :])
        fig.savefig(pn.get())
        plt.close(fig)

# These are very slow:
# magnetogram/test_pfss_magnetogram.py::test_single_line PASSED                                                                           [ 95%]
# magnetogram/test_pfss_magnetogram.py::test_single_line_spherical