import numpy as np
import logging
log = logging.getLogger(__name__)
import quaternion
import pytest
from tests import context  # Test context
from tests import magnetograms
from tests.test_converter import get_field_values
import starwinds_magnetogram.rotate as rm
from starwinds_magnetogram import converter
from starwinds_magnetogram import zdi_magnetogram
import starwinds_magnetogram.geometry as zdi_geometry

from starwinds_magnetogram import fibonacci_sphere
from starwinds_magnetogram import coordinate_transforms

_magnetograms = "mengel",
_angles_deg = np.array([15, 30, 45, 90, 180, 360])


@pytest.mark.parametrize("magnetogram_name", _magnetograms)
@pytest.mark.parametrize("alpha_deg", _angles_deg)
def test_rotate_magnetogram_around_z(request, magnetogram_name, alpha_deg):

    original = magnetograms.get_radial(magnetogram_name)
    rotated = rm.rotate_magnetogram_euler_zyz_deg(original, (alpha_deg, 0, 0))
    folded = converter.map_to_positive_orders(rotated)

    zg = zdi_geometry.ZdiGeometry()
    field_values = get_field_values(zg, original, rotated, folded)

    assert np.allclose(np.imag(field_values[0]), 0), "No imaginary component expected."
    assert np.allclose(np.imag(field_values[1]), 0), "No imaginary component expected."
    assert np.allclose(np.imag(field_values[2]), 0), "No imaginary component expected."

    # At least the min, max should not change. A bit tolerance as exact values depend on the discretization.
    assert np.isclose(np.min(field_values[0]), np.min(field_values[1]), rtol=1e-2), "Minimum value should be unchanged."
    assert np.isclose(np.max(field_values[0]), np.max(field_values[1]), rtol=1e-2), "Maximum value should be unchanged."

    #
    # Values at specific points.
    #
    zl0 = zdi_magnetogram.from_coefficients(original)
    zl1 = zdi_magnetogram.from_coefficients(folded)

    # Existing functionality to get points - yay!
    px, py, pz = fibonacci_sphere.fibonacci_sphere(10).transpose()
    _, _pl, _az = coordinate_transforms.spherical_coordinates_from_rectangular(px, py, pz)

    az0 = np.mod(_az, 2*np.pi)  # Just to correct for -np.pi to np.pi
    pl0 = _pl
    az1 = az0 - np.deg2rad(alpha_deg)
    pl1 = pl0

    values0 = zl0.get_radial_field(pl0, az0)
    values1 = zl1.get_radial_field(pl1, az1)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, axs = plt.subplots(1, 2)

        for fv, ax in zip(field_values, axs):
            img1 = ax.pcolormesh(*zg.corners()[::-1], np.real(fv))
            ax.plot(az0, pl0, 'ko')
            ax.plot(az1, pl1, 'k>')
            fig.colorbar(img1, ax=ax, orientation='horizontal')

        plt.savefig(pn.get())
        plt.close()

    assert np.allclose(values0, values1)


_euler_angles = (np.array([   0,    0,    0]),
                 np.array([  45,    0,    0]),
                 np.array([  60,    0,    0]),
                 np.array([   0,    0,    0]),
                 np.array([   0,   30,    0]),
                 np.array([   0,  180,    0]),
                 np.array([ +67,   43,   13]))


@pytest.mark.parametrize("magnetogram_name", _magnetograms)
@pytest.mark.parametrize("euler_deg", _euler_angles)
def test_rotate_magnetogram(request, magnetogram_name, euler_deg):

    original = magnetograms.get_radial(magnetogram_name)
    rotated = rm.rotate_magnetogram_euler_zyz_deg(original, euler_deg)
    rotated = converter.map_to_positive_orders(rotated)
    assert rotated.order_min >= 0

    zg = zdi_geometry.ZdiGeometry(128)
    field_values = get_field_values(zg, original, rotated)

    assert np.allclose(np.imag(field_values[0]), 0), "No imaginary component expected."
    assert np.allclose(np.imag(field_values[1]), 0), "No imaginary component expected."

    # At least the min, max should not change. A bit tolerance as exact values depend on the discretization.
    assert np.isclose(np.min(field_values[0]), np.min(field_values[1]), rtol=1e-2), "Minimum value should be unchanged."
    assert np.isclose(np.max(field_values[0]), np.max(field_values[1]), rtol=1e-2), "Maximum value should be unchanged."

    #
    # Values at specific points.
    #
    zl0 = zdi_magnetogram.from_coefficients(original)
    zl1 = zdi_magnetogram.from_coefficients(rotated)

    # Existing functionality to get points - yay!
    original_points = np.stack(fibonacci_sphere.fibonacci_sphere(100)).transpose()

    # Rotate points with rotation quaternion
    rotation_quaternion = quaternion.from_euler_angles(np.deg2rad(euler_deg))
    rotated_points = quaternion.as_rotation_matrix(rotation_quaternion).transpose() @ original_points
    _, _pl, _az = coordinate_transforms.spherical_coordinates_from_rectangular(*original_points)
    az0 = np.mod(_az, 2*np.pi)  # Just to correct for -np.pi to np.pi
    pl0 = _pl
    _, _pl, _az = coordinate_transforms.spherical_coordinates_from_rectangular(*rotated_points)
    az1 = np.mod(_az, 2*np.pi)  # Just to correct for -np.pi to np.pi
    pl1 = _pl

    values0 = zl0.get_radial_field(pl0, az0)
    values1 = zl1.get_radial_field(pl1, az1)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, axs = plt.subplots(1, 2)

        for fv, ax in zip(field_values, axs):
            abs_max_fv = np.max(np.abs(fv))
            img1 = ax.pcolormesh(*zg.corners()[::-1], np.real(fv),
                                 vmin=-abs_max_fv, vmax=abs_max_fv, cmap='RdBu_r')
            fig.colorbar(img1, ax=ax, orientation='horizontal')
            ax.invert_yaxis()
            ax.set_xlabel("Azimuth [rad]")

        for _pid in range(len(pl0)):
            axs[0].text(az0[_pid], pl0[_pid], '%d' % _pid)

            axs[1].plot((az0[_pid], az1[_pid]), (pl0[_pid], pl1[_pid]))
            axs[1].text(az1[_pid], pl1[_pid], '%d' % _pid)

        axs[0].set_title("Original magnetogram")
        axs[0].set_ylabel("Polar angle [rad]")

        axs[1].set_title("Euler angles zyz [deg] (%4.1f, %4.1f, %4.1f)" % tuple(euler_deg))
        plt.savefig(pn.get())
        plt.close()

    assert np.allclose(values0, values1)


def test_quaternion_rotation():
    # See the log to see the effect of the rotations.

    # Half rotations
    euler = (np.pi, 0, 0)
    q = quaternion.from_euler_angles(euler)
    mat = quaternion.as_rotation_matrix(q)
    log.info("Euler:\n" + str(euler) + "Quaternion:\n" + str(q) + "Matrix: \n" + str(mat))

    euler = (0, np.pi, 0)
    q = quaternion.from_euler_angles(euler)
    mat = quaternion.as_rotation_matrix(q)
    log.info("Euler:\n" + str(euler) + "Quaternion:\n" + str(q) + "Matrix: \n" + str(mat))

    euler = (0, 0, np.pi)
    q = quaternion.from_euler_angles(euler)
    mat = quaternion.as_rotation_matrix(q)
    log.info("Euler:\n" + str(euler) + "Quaternion:\n" + str(q) + "Matrix: \n" + str(mat))

    # Quarter rotations
    euler = (np.pi/2, 0, 0)
    q = quaternion.from_euler_angles(euler)
    mat = quaternion.as_rotation_matrix(q)
    log.info("Euler:\n" + str(euler) + "Quaternion:\n" + str(q) + "Matrix: \n" + str(mat))

    euler = (0, np.pi/2, 0)
    q = quaternion.from_euler_angles(euler)
    mat = quaternion.as_rotation_matrix(q)
    log.info("Euler:\n" + str(euler) + "Quaternion:\n" + str(q) + "Matrix: \n" + str(mat))

    euler = (0, 0, np.pi/2)
    q = quaternion.from_euler_angles(euler)
    mat = quaternion.as_rotation_matrix(q)
    log.info("Euler:\n" + str(euler) + "Quaternion:\n" + str(q) + "Matrix: \n" + str(mat))
