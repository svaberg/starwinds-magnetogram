import numpy as np
import logging
log = logging.getLogger(__name__)

import matplotlib.ticker

# Local
from stellarwinds import coordinate_transforms


def get_longitudinal_field_curve(zdi_geometry, zdi_magnetogram, observer_polar, observer_azimuths=None, field="full"):
    """
    Calculate visible longitudinal field.
    :param zdi_geometry:
    :param zdi_magnetogram:
    :param observer_polar: Observer polar angle in stellar coordinate system (corresponds to inclination)
    :param observer_azimuths: Observer azimuth angle(s)
    :param field: full field, or toroidal field, or poloidal field.
    :return:
    """

    if observer_azimuths is None:
        observer_azimuths = np.deg2rad(np.linspace(0, 360, 8 * zdi_magnetogram.degree()))

    longitudinal_fields = np.empty_like(observer_azimuths)

    areas = zdi_geometry.areas()

    if field == "full":
        field_xyz = zdi_magnetogram.get_cartesian_field(*zdi_geometry.centers())
    elif field == "poloidal":
        field_xyz = zdi_magnetogram.get_cartesian_poloidal_field(*zdi_geometry.centers())
    elif field == "toroidal":
        field_xyz = zdi_magnetogram.get_cartesian_toroidal_field(*zdi_geometry.centers())
    else:
        raise TypeError("Unexpected field type \"%s\"" % field)

    for i, observer_azimuth in enumerate(observer_azimuths):
        observer_direction = np.array(
            coordinate_transforms.rectangular_coordinates_from_spherical(1, observer_polar, observer_azimuth))

        parallel_field = np.sum(field_xyz * observer_direction, axis=-1)
        projected_visible_area_fraction = zdi_geometry.projected_visible_area_fraction(observer_direction)
        visible_areas = areas * projected_visible_area_fraction

        longitudinal_fields[i] = np.sum(parallel_field * visible_areas) / np.sum(visible_areas)

        if False and field == "toroidal" and i%5==0:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            pl, az = [np.rad2deg(v) for v in zdi_geometry.corners()]
            _v = parallel_field * projected_visible_area_fraction
            ax.pcolormesh(az, pl,
                          _v,
                          vmin=-np.max(np.abs(_v)),
                          vmax=+np.max(np.abs(_v)))
            ax.set_xticks(np.arange(0, 361, 60))
            ax.set_yticks(np.arange(0, 181, 30))
            ax.grid()

            ax.plot(np.rad2deg(observer_azimuth), np.rad2deg(observer_polar), 'ko')
            ax.set_title(f"{field}, i={i}. LF={longitudinal_fields[i]}")

            plt.show()

    return longitudinal_fields


def plot_longitudinal_field_curve(ax, zdi_geometry, zdi_magnetogram, observer_polar, observer_azimuths=None):

    if observer_azimuths is None:
        observer_azimuths = np.deg2rad(np.linspace(0, 360, 8 * zdi_magnetogram.degree()))

    lf_full = get_longitudinal_field_curve(zdi_geometry, zdi_magnetogram, observer_polar, observer_azimuths)
    lf_pol = get_longitudinal_field_curve(zdi_geometry, zdi_magnetogram, observer_polar, observer_azimuths, field="poloidal")
    lf_tor = get_longitudinal_field_curve(zdi_geometry, zdi_magnetogram, observer_polar, observer_azimuths, field="toroidal")

    plot_style(ax, lf_full, lf_pol, lf_tor, observer_azimuths)

    return ax, np.stack((observer_azimuths, lf_full, lf_pol, lf_tor))


def plot_style(ax, lf_full, lf_pol, lf_tor, observer_azimuths):
    ax.plot(np.rad2deg(observer_azimuths), lf_full, label="full", linewidth=2)
    ax.axhline(np.mean(lf_full))
    ax.plot(np.rad2deg(observer_azimuths), lf_pol, '--', label="poloidal")
    ax.plot(np.rad2deg(observer_azimuths), lf_tor, '--', label="toroidal")
    max_ampl = np.max(np.abs(lf_full))
    ax.set_ylim(np.array([-1, 1]) * max_ampl * 1.1)
    ax.grid(True)
    loc = matplotlib.ticker.MultipleLocator(base=60.0)  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    ax.legend()

