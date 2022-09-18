import numpy as np
import logging
log = logging.getLogger(__name__)

import matplotlib.ticker

# Local
from starwinds_magnetogram import coordinate_transforms


def get_longitudinal_field_curve(zdi_geometry, zdi_magnetogram, observer_polar, observer_azimuths=None):
    """
    Calculate visible longitudinal field.
    :param zdi_geometry:
    :param zdi_magnetogram:
    :param observer_polar: Observer polar angle in stellar coordinate system (corresponds to inclination)
    :param observer_azimuths: Observer azimuth angle(s)
    :return:
    """

    if observer_azimuths is None:
        observer_azimuths = np.deg2rad(np.linspace(0, 360, zdi_geometry.shape[0]))

    longitudinal_fields = np.empty_like(observer_azimuths)

    areas = zdi_geometry.areas()

    # if field == "full":
    #     field_xyz = zdi_magnetogram.get_cartesian_field(*zdi_geometry.centers())
    # elif field == "poloidal":
    #     field_xyz = zdi_magnetogram.get_cartesian_poloidal_field(*zdi_geometry.centers())
    # elif field == "toroidal":
    #     field_xyz = zdi_magnetogram.get_cartesian_toroidal_field(*zdi_geometry.centers())
    # else:
    #     raise TypeError("Unexpected field type \"%s\"" % field)
    field_xyz = zdi_magnetogram.get_cartesian_field(*zdi_geometry.centers())

    for i, observer_azimuth in enumerate(observer_azimuths):
        observer_direction = np.array(
            coordinate_transforms.rectangular_coordinates_from_spherical(1, observer_polar, observer_azimuth))

        parallel_field = np.sum(field_xyz * observer_direction, axis=-1)
        projected_visible_area_fraction = zdi_geometry.projected_visible_area_fraction(observer_direction)
        visible_areas = areas * projected_visible_area_fraction

        longitudinal_fields[i] = np.sum(parallel_field * visible_areas) / np.sum(visible_areas)

    return longitudinal_fields


def plot_longitudinal_field_curve(ax, zdi_geometry, zdi_magnetogram, observer_polar, observer_azimuths=None):

    if observer_azimuths is None:
        observer_azimuths = np.deg2rad(np.linspace(0, 360, zdi_geometry.shape[0]))

    lf_full = get_longitudinal_field_curve(zdi_geometry, zdi_magnetogram, observer_polar, observer_azimuths)

    plot_style(ax, lf_full, observer_azimuths)

    return ax, np.stack((observer_azimuths, lf_full))


def plot_style(ax, lf_full, observer_azimuths):
    ax.plot(np.rad2deg(observer_azimuths), lf_full)
    ax.axhline(np.mean(lf_full), linestyle='--')
    max_ampl = np.max(np.abs(lf_full))
    ax.set_ylim(np.array([-1, 1]) * max_ampl * 1.1)
    ax.grid(True)
    loc = matplotlib.ticker.MultipleLocator(base=60.0)  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
