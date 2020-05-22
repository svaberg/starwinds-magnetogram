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
from stellarwinds import coordinate_transforms
import pytest


def test_normals():
    zg = stellarwinds.magnetogram.geometry.ZdiGeometry(11)
    unit_normals = zg.unit_normals()
    assert np.allclose(np.sum(unit_normals**2, axis=-1), 1)


def test_observation_angle(request):
    observer_direction = np.array([0, 1, 1])
    obs_dir_length = np.sum(observer_direction**2)**.5
    zg = stellarwinds.magnetogram.geometry.ZdiGeometry(64)
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


def test_visible_area_fraction(request):
    observer_direction = np.array([0, 0, 1])
    zg = stellarwinds.magnetogram.geometry.ZdiGeometry(256)

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


def test_parallel_field(request, magnetogram_name="mengel"):
    """Longitudinal refers to the direction towards the observer."""
    observer_direction = np.array([0, 1, 1])
    zg = stellarwinds.magnetogram.geometry.ZdiGeometry(64)

    lz = stellarwinds.magnetogram.zdi_magnetogram.from_coefficients(magnetograms.get_all(magnetogram_name))

    parallel_field = get_parallel_field(zg, lz, observer_direction)

    # Average field over entire surface (including backside). This does not average out to 0.
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
    zg = stellarwinds.magnetogram.geometry.ZdiGeometry(64)

    lz = stellarwinds.magnetogram.zdi_magnetogram.from_coefficients(magnetograms.get_all(magnetogram_name))

    parallel_field = get_parallel_field(zg, lz, observer_direction)
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


def test_longitudinal_field_curve(request, magnetogram_name="quadrupole"):

    zg = stellarwinds.magnetogram.geometry.ZdiGeometry(64)
    lz = stellarwinds.magnetogram.zdi_magnetogram.from_coefficients(magnetograms.get_all(magnetogram_name))

    observer_polar = np.deg2rad(50)

    observer_azimuths = np.deg2rad(np.linspace(0, 360, 20))
    longitudinal_fields = np.empty_like(observer_azimuths)

    areas = zg.areas()

    for i, observer_azimuth in enumerate(observer_azimuths):
        observer_direction = np.array(
            coordinate_transforms.rectangular_coordinates_from_spherical(1, observer_polar, observer_azimuth))

        parallel_field = get_parallel_field(zg, lz, observer_direction)
        projected_visible_area_fraction = zg.projected_visible_area_fraction(observer_direction)
        visible_areas = areas * projected_visible_area_fraction

        longitudinal_fields[i] = np.sum(parallel_field * visible_areas) / np.sum(visible_areas)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        polar, azimuth = zg.corners()
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
        ax.grid(True)
        plt.savefig(pn.get())


def get_parallel_field(zdi_geometry, zdi_magnetogram, direction):
    """
    Calculate projected magnetic field on the surface facets of the zdi_geometry.
    To calculate the longitudinal magnetic field this must be averaged over the projected visible surface of each
    facet.
    :param zdi_geometry:
    :param zdi_magnetogram:
    :param direction: direction vector of projection
    :return:
    """
    # Field components in spherical coordinates
    fr = zdi_magnetogram.get_radial_field(*zdi_geometry.centers())
    fp = zdi_magnetogram.get_polar_field(*zdi_geometry.centers())
    fa = zdi_magnetogram.get_azimuthal_field(*zdi_geometry.centers())

    px, py, pz = [zdi_geometry.unit_normals()[..., i] for i in range(3)]

    field_xyz, *_ = mmm(fr, fp, fa, px, py, pz)

    pr, pp, pa = coordinate_transforms.spherical_coordinates_from_rectangular(px, py, pz)
    new_field_xyz, *_ = mmm2(fr, fp, fa, pr, pp, pa)
    assert np.allclose(field_xyz, new_field_xyz)

    return np.sum(field_xyz * direction, axis=-1)


def test_mmm(request, magnetogram_name="mengel"):
    direction = [0, 1, 1]
    zdi_geometry = stellarwinds.magnetogram.geometry.ZdiGeometry(64)
    zdi_magnetogram = stellarwinds.magnetogram.zdi_magnetogram.from_coefficients(magnetograms.get_all(magnetogram_name))

    """
    Calculate projected magnetic field on the surface facets of the zdi_geometry.
    To calculate the longitudinal magnetic field this must be averaged over the projected visible surface of each
    facet.
    :param zdi_geometry:
    :param zdi_magnetogram:
    :param direction: direction vector of projection
    :return:
    """
    # Field components in spherical coordinates
    fr = zdi_magnetogram.get_radial_field(*zdi_geometry.centers())
    fp = zdi_magnetogram.get_polar_field(*zdi_geometry.centers())
    fa = zdi_magnetogram.get_azimuthal_field(*zdi_geometry.centers())

    observer_polar = np.deg2rad(60)
    observer_azimuths = np.deg2rad(np.linspace(0, 360, 20))
    longitudinal_fields = np.empty_like(observer_azimuths)

    for i, observer_azimuth in enumerate(observer_azimuths):
        observer_direction = np.array(
            coordinate_transforms.rectangular_coordinates_from_spherical(1, observer_polar, observer_azimuth))

        px, py, pz = [zdi_geometry.unit_normals()[..., i] for i in range(3)]
        fxyz, frpa, t = mmm(fr, fp, fa, px, py, pz)

        pr, pp, pa = coordinate_transforms.spherical_coordinates_from_rectangular(px, py, pz)
        fxyz2, frpa2, t2 = mmm2(fr, fp, fa, pr, pp, pa)

        assert np.allclose(t, t2)
        assert np.allclose(frpa, frpa2)
        assert np.allclose(fxyz, fxyz2)



def mmm(fr, fp, fa, px, py, pz):
    pr, pp, pa = coordinate_transforms.spherical_coordinates_from_rectangular(px, py, pz)
    assert pr.shape == px.shape, "Expected matching shapes"
    assert pp.shape == px.shape, "Expected matching shapes"
    assert pa.shape == px.shape, "Expected matching shapes"
    assert fr.shape == pr.shape, "Expected matching shapes"
    assert fp.shape == pr.shape, "Expected matching shapes"
    assert fa.shape == pr.shape, "Expected matching shapes"
    # To carry out the transformation, flatten the polar coordinate arrays and stack them
    # calculate the transformation matrix, and apply it to the stack field_rpa.
    field_rpa = np.stack([c.flatten() for c in (fr, fp, fa)], axis=-1)
    transformation_matrix = coordinate_transforms.spherical_to_rectangular_transformation_matrix(pp.flatten(),
                                                                                                 pa.flatten())
    field_xyz = transformation_matrix @ field_rpa[:, :, np.newaxis]
    # Get rid of last dimension which is has length 1
    assert field_xyz.shape[-1] == 1, "Expected last dimension size to be 1"
    assert field_xyz.shape[-2] == 3, "Expected second last dimension size to be 3"
    field_xyz = field_xyz[..., 0].reshape(px.shape + (3,))
    return field_xyz, field_rpa, transformation_matrix


def mmm2(fr, fp, fa, pr, pp, pa):
    assert fr.shape == pr.shape, "Expected matching shapes"
    assert fp.shape == pr.shape, "Expected matching shapes"
    assert fa.shape == pr.shape, "Expected matching shapes"
    # To carry out the transformation, flatten the polar coordinate arrays and stack them
    # calculate the transformation matrix, and apply it to the stack field_rpa.
    field_rpa = np.stack([c.flatten() for c in (fr, fp, fa)], axis=-1)
    transformation_matrix = coordinate_transforms.spherical_to_rectangular_transformation_matrix(pp.flatten(),
                                                                                                 pa.flatten())
    field_xyz = transformation_matrix @ field_rpa[:, :, np.newaxis]
    # Get rid of last dimension which is has length 1
    assert field_xyz.shape[-1] == 1, "Expected last dimension size to be 1"
    assert field_xyz.shape[-2] == 3, "Expected second last dimension size to be 3"
    field_xyz = field_xyz[..., 0].reshape(pr.shape + (3,))
    return field_xyz, field_rpa, transformation_matrix
