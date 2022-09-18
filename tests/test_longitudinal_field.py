import matplotlib as mpl
from matplotlib import cm
import numpy as np
import logging

# Test "context"
import starwinds_magnetogram.plot_zdi
from tests import context  # Test context
from tests.magnetogram import magnetograms
from tests.magnetogram import test_flow
log = logging.getLogger(__name__)

# Local
import starwinds_magnetogram.zdi_magnetogram
import starwinds_magnetogram.geometry
from starwinds_magnetogram import plots
starwinds_magnetogram import coordinate_transforms
from starwinds_magnetogram import longitudinal_field
from starwinds_magnetogram import coefficients as shc
import pytest


def test_normals():
    zg = starwinds_magnetogram.geometry.ZdiGeometry(11)
    unit_normals = zg.unit_normals()
    assert np.allclose(np.sum(unit_normals**2, axis=-1), 1)


def test_observation_angle(request):
    observer_direction = np.array([0, 1, 1])
    obs_dir_length = np.sum(observer_direction**2)**.5
    zg = starwinds_magnetogram.geometry.ZdiGeometry(64)
    unit_normals = zg.unit_normals()

    # Dot product of observer direction and normal vectors
    cos_angle = np.sum((observer_direction / obs_dir_length) * unit_normals, axis=-1)
    assert np.all(cos_angle <= 1)
    assert np.all(cos_angle >= -1)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        polar, azimuth = zg.centers()
        fig, ax = plt.subplots()
        img1 = ax.pcolormesh(np.rad2deg(azimuth), np.rad2deg(polar), np.rad2deg(np.arccos(cos_angle)),
                             vmin=0, vmax=180)
        fig.colorbar(img1, ax=ax)
        ax.invert_yaxis()
        ax.set_title("Observation angle. Observer at " + str(observer_direction))

        _, p, a = coordinate_transforms.spherical_coordinates_from_rectangular(*observer_direction)
        plt.plot(np.rad2deg(a), np.rad2deg(p), 'ko')

        plt.savefig(pn.get())


observer_directions = (np.array([1, 0, 0]),
                       np.array([0, 1, 0]),
                       np.array([0, 0, 1]),
                       )


@pytest.mark.parametrize("observer_direction", observer_directions)
def test_visible_area_fraction(request, observer_direction):

    zg = starwinds_magnetogram.geometry.ZdiGeometry(256)

    visible_fraction = zg.projected_visible_area_fraction(observer_direction)

    assert np.all(visible_fraction >= 0)
    assert np.all(visible_fraction <= 1)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        polar, azimuth = zg.centers()
        fig, ax = plt.subplots()
        img1 = ax.pcolormesh(np.rad2deg(azimuth), np.rad2deg(polar), visible_fraction, vmin=0, vmax=1)
        fig.colorbar(img1, ax=ax)
        ax.invert_yaxis()
        ax.set_title("Visible fraction. Observer at " + str(observer_direction))

        _, p, a = coordinate_transforms.spherical_coordinates_from_rectangular(*observer_direction)
        plt.plot(np.rad2deg(a), np.rad2deg(p), 'ko')

        plt.savefig(pn.get())


@pytest.mark.parametrize("observer_direction", observer_directions)
def test_radial_parallel_field(request, observer_direction):
    zg = starwinds_magnetogram.geometry.ZdiGeometry(64)
    field_xyz = zg.unit_normals()  # Create a field of unit normals.
    parallel_field = np.sum(field_xyz * observer_direction, axis=-1)
    average_proj_field = np.sum(zg.areas() * parallel_field) / np.sum(zg.areas())
    assert np.isclose(average_proj_field, 0)
    log.info(f"Sphere average parallel field {average_proj_field}")


@pytest.mark.parametrize("observer_direction", observer_directions)
def test_x_parallel_field(request, observer_direction):

    def expected(observer_direction):
        return np.sum(np.array([1, 0, 0]) * observer_direction)

    zg = starwinds_magnetogram.geometry.ZdiGeometry(64)
    field_xyz = np.zeros_like(zg.unit_normals())  # Create a field of unit normals.
    field_xyz[..., 0] = 1  # Set x components to 1

    parallel_field = np.sum(field_xyz * observer_direction, axis=-1)
    average_proj_field = np.sum(zg.areas() * parallel_field) / np.sum(zg.areas())
    assert np.isclose(average_proj_field, expected(observer_direction))
    log.info(f"Sphere average parallel field {average_proj_field}")


@pytest.mark.parametrize("magnetogram_name", ("mengel",))
@pytest.mark.parametrize("observer_direction", observer_directions)
def test_parallel_field(request, magnetogram_name, observer_direction):
    """Longitudinal refers to the direction towards the observer."""
    zg = starwinds_magnetogram.geometry.ZdiGeometry(64)

    lz = starwinds_magnetogram.zdi_magnetogram.from_coefficients(magnetograms.get_all(magnetogram_name))

    field_xyz = lz.get_cartesian_field(*zg.centers())

    parallel_field = np.sum(field_xyz * observer_direction, axis=-1)

    # Average field over entire surface (including backside). This does not average out to zero, e.g. for a dipole
    # facing towards the observer.
    average_proj_field = np.sum(zg.areas() * parallel_field) / np.sum(zg.areas())
    log.info(f"Sphere average parallel field {average_proj_field}")

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        polar, azimuth = zg.centers()
        fig, ax = plt.subplots()
        img1 = ax.pcolormesh(np.rad2deg(azimuth), np.rad2deg(polar), parallel_field,
                             vmin=-np.max(np.abs(parallel_field)),
                             vmax=+np.max(np.abs(parallel_field)),
                             )
        fig.colorbar(img1, ax=ax)
        ax.invert_yaxis()
        ax.set_title(f"Sphere average parallel field {average_proj_field}.")

        _, p, a = coordinate_transforms.spherical_coordinates_from_rectangular(*observer_direction)
        ax.plot(np.rad2deg(a), np.rad2deg(p), 'ko')

        plt.savefig(pn.get())


def test_longitudinal_field(request, magnetogram_name="mengel"):
    """Longitudinal refers to the direction towards the observer.
    This is the same as each facet's parallel field multiplied by its visible area"""
    observer_direction = np.array([0, 1, 1])
    zg = starwinds_magnetogram.geometry.ZdiGeometry(64)

    lz = starwinds_magnetogram.zdi_magnetogram.from_coefficients(magnetograms.get_all(magnetogram_name))

    field_xyz = lz.get_cartesian_field(*zg.centers())
    parallel_field = np.sum(field_xyz * observer_direction, axis=-1)
    projected_visible_area_fraction = zg.projected_visible_area_fraction(observer_direction)
    visible_areas = zg.areas() * projected_visible_area_fraction

    longitudinal_field = np.sum(parallel_field * visible_areas) / np.sum(visible_areas)
    log.info(f"Longitudinal field {longitudinal_field}")

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        polar, azimuth = zg.centers()
        fig, axs = plt.subplots(2, 1)
        ax = axs[0]
        img1 = ax.pcolormesh(np.rad2deg(azimuth), np.rad2deg(polar), parallel_field * projected_visible_area_fraction,
                             vmin=-np.max(np.abs(parallel_field * projected_visible_area_fraction)),
                             vmax=+np.max(np.abs(parallel_field * projected_visible_area_fraction)),
                             )
        fig.colorbar(img1, ax=ax)
        ax.invert_yaxis()

        _, p, a = coordinate_transforms.spherical_coordinates_from_rectangular(*observer_direction)
        ax.plot(np.rad2deg(a), np.rad2deg(p), 'ko')

        ax.set_title("Longitudinal field %f" % longitudinal_field)
        ax.grid()
        ax = axs[1]
        _summands = parallel_field * visible_areas / np.sum(visible_areas)
        img1 = ax.pcolormesh(np.rad2deg(azimuth), np.rad2deg(polar), _summands,
                             vmin=-np.max(np.abs(_summands)),
                             vmax=+np.max(np.abs(_summands)),
                             )
        fig.colorbar(img1, ax=ax)
        ax.invert_yaxis()
        ax.set_title(f"Area corrected longitudinal field summands. Sum %f" % np.sum(_summands))
        _, p, a = coordinate_transforms.spherical_coordinates_from_rectangular(*observer_direction)
        ax.plot(np.rad2deg(a), np.rad2deg(p), 'ko')
        ax.grid()
        plt.savefig(pn.get())


def test_longitudinal_field_curve(request, magnetogram_name="mengel"):

    observer_polar = np.deg2rad(50)

    zdi_geometry = starwinds_magnetogram.geometry.ZdiGeometry(64)
    zdi_magnetogram = starwinds_magnetogram.zdi_magnetogram.from_coefficients(magnetograms.get_all(magnetogram_name))

    observer_azimuths = np.deg2rad(np.linspace(0, 360, 20))


    longitudinal_fields = np.empty_like(observer_azimuths)

    areas = zdi_geometry.areas()

    field_xyz = zdi_magnetogram.get_cartesian_field(*zdi_geometry.centers())

    for i, observer_azimuth in enumerate(observer_azimuths):
        observer_direction = np.array(
            coordinate_transforms.rectangular_coordinates_from_spherical(1, observer_polar, observer_azimuth))

        parallel_field = np.sum(field_xyz * observer_direction, axis=-1)
        projected_visible_area_fraction = zdi_geometry.projected_visible_area_fraction(observer_direction)
        visible_areas = areas * projected_visible_area_fraction

        longitudinal_fields[i] = np.sum(parallel_field * visible_areas) / np.sum(visible_areas)

    assert np.allclose(longitudinal_field.get_longitudinal_field_curve(zdi_geometry, zdi_magnetogram, observer_polar,observer_azimuths), longitudinal_fields)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        polar, azimuth = zdi_geometry.corners()
        fig, axs = plt.subplots(2, 1, sharex=True)
        ax = axs[0]
        img1 = ax.pcolormesh(np.rad2deg(azimuth), np.rad2deg(polar), parallel_field * projected_visible_area_fraction,
                             vmin=-np.max(np.abs(parallel_field * projected_visible_area_fraction)),
                             vmax=+np.max(np.abs(parallel_field * projected_visible_area_fraction)),
                             )
        fig.colorbar(img1, ax=axs)
        ax.invert_yaxis()

        _, p, a = coordinate_transforms.spherical_coordinates_from_rectangular(*observer_direction)
        ax.plot(np.rad2deg(a), np.rad2deg(p), 'ko')

        # ax.set_title("Longitudinal field %f" % longitudinal_field)
        ax.grid()
        ax = axs[1]
        ax.plot(np.rad2deg(observer_azimuths), longitudinal_fields)
        ax.axhline(np.mean(longitudinal_fields))
        # ax.axhline(0, color='k')
        max_ampl = np.max(np.abs(longitudinal_fields))
        ax.set_ylim(np.array([-1, 1]) * max_ampl * 1.1)
        ax.grid(True)
        plt.savefig(pn.get())


def test_polar_view(request, magnetogram_name="mengel"):

    observer_polar = np.deg2rad(0)

    zdi_geometry = starwinds_magnetogram.geometry.ZdiGeometry(64)
    zdi_magnetogram = starwinds_magnetogram.zdi_magnetogram.from_coefficients(magnetograms.get_all(magnetogram_name))

    lf = longitudinal_field.get_longitudinal_field_curve(zdi_geometry, zdi_magnetogram, observer_polar)

    avg_lf = np.mean(lf)

    assert np.allclose(lf, avg_lf)


def test_poloidal(request, magnetogram_name="mengel"):

    observer_polar = np.deg2rad(10)
    zdi_geometry = starwinds_magnetogram.geometry.ZdiGeometry(256)

    coefficients = shc.Coefficients(np.zeros(3, dtype=complex))
    # coefficients.append(13, 7, np.array([1e6+0.0j, 0.0+0.0j, 1.0e6+0.0j]))
    # coefficients.append(11, 8, np.array([0.0+0.0j, 0.0+0.0j, 1.0e6+0.0j]))
    coefficients.append(3, 2, np.array([0.0+0.0j, 0.0+0.0j, 0.0+1.0j]))
    zdi_magnetogram = starwinds_magnetogram.zdi_magnetogram.from_coefficients(coefficients)

    # lf = longitudinal_field.get_longitudinal_field_curve(zdi_geometry, zdi_magnetogram, observer_polar)
    #
    # avg_lf = np.mean(lf)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, ax = plt.subplots()

        longitudinal_field.plot_longitudinal_field_curve(ax, zdi_geometry, zdi_magnetogram, observer_polar)

        plt.savefig(pn.get())

    # assert np.allclose(lf, 0)


def test_poloidal2(request, magnetogram_name="mengel"):

    observer_polar = np.deg2rad(60)
    zdi_geometry = starwinds_magnetogram.geometry.ZdiGeometry(256)

    coefficients = shc.Coefficients(np.zeros(3, dtype=complex))
    coefficients.append(1, 0, np.array([0.0+0.0j, 0.0+0.0j, 1.0+0.0j]))
    zdi_magnetogram = starwinds_magnetogram.zdi_magnetogram.from_coefficients(coefficients)

    lf = longitudinal_field.get_longitudinal_field_curve(zdi_geometry, zdi_magnetogram, observer_polar)
    assert np.allclose(lf, 0)


def test_plot_longitudinal_field(request, magnetogram_name="mengel"):
    """Why is the toroidal component of the mengel magnetogram so close to zero??"""
    observer_polar = np.deg2rad(30)

    zdi_geometry = starwinds_magnetogram.geometry.ZdiGeometry(64)
    zdi_magnetogram = starwinds_magnetogram.zdi_magnetogram.from_coefficients(magnetograms.get_all(magnetogram_name))

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, ax = plt.subplots()

        longitudinal_field.plot_longitudinal_field_curve(ax, zdi_geometry, zdi_magnetogram, observer_polar)

        plt.savefig(pn.get())
