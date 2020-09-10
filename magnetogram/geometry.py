import numpy as np
import logging
log = logging.getLogger(__name__)
from functools import cached_property
import stellarwinds.coordinate_transforms


class ZdiGeometry:
    def __init__(self,
                 polar_corners=None,
                 azimuthal_corners=None):

        if type(polar_corners) is int and type(azimuthal_corners) is int:
            polar_corners = np.linspace(0, np.pi, polar_corners)
            azimuthal_corners = np.linspace(0, 2 * np.pi, azimuthal_corners)
        elif type(polar_corners) is int and azimuthal_corners is None:
            polar_corners = np.linspace(0, np.pi, polar_corners)
            azimuthal_corners = np.linspace(0, 2 * np.pi, 2 * len(polar_corners))
        elif polar_corners is None and azimuthal_corners is None:
            polar_corners = np.linspace(0, np.pi, 64 + 1)
            azimuthal_corners = np.linspace(0, 2 * np.pi, 2 * len(polar_corners))

        # Only go around once.
        assert(np.abs(np.max(polar_corners) - np.min(polar_corners)) <= np.pi)
        assert(np.abs(np.max(azimuthal_corners) - np.min(azimuthal_corners)) <= 2*np.pi)

        self.polar_corners, self.azimuthal_corners = np.meshgrid(polar_corners, azimuthal_corners)

    @property
    def shape(self):
        return self.polar_corners.shape

    def corners(self):
        """Facet corner polar coordinates (polar, azimuth)"""
        return self.polar_corners, self.azimuthal_corners

    # @cached_property
    def centers(self):
        """Facet center polar coordinates (polar, azimuth)"""
        def make_centers(x):
            return 0.25 * (x[:-1, :-1] + x[:-1, 1:] + x[1:, :-1] + x[1:, 1:])

        polar_centers = make_centers(self.polar_corners)
        azimuthal_centers = make_centers(self.azimuthal_corners)

        assert azimuthal_centers.shape == polar_centers.shape
        return polar_centers, azimuthal_centers

    def unit_normals(self):
        """Facet unit normal vector cartesian coordinates as ndarray"""
        return np.stack(self.centers_cartesian(), axis=-1)

    # def normals(self):
    #     """Facet normal vector cartesian coordinates as ndarray. The lengths correspond to the facet area."""
    #     return self.unit_normals() * self.areas()[..., np.newaxis]

    def areas(self):
        """Calculate area as difference between two spherical caps
        https://en.wikipedia.org/wiki/Spherical_cap
        TODO this is inconsistent with the faceted approach used elsewhere in this class."""

        height_delta = (np.cos(self.polar_corners[:-1, :-1]) - np.cos(self.polar_corners[:-1, 1:]))
        azimuth_delta = (self.azimuthal_corners[1:, 1:] - self.azimuthal_corners[:-1, 1:])

        return height_delta * azimuth_delta

    # def visible_areas(self, projection_direction):
    #     """Calculate projected (visible) area in the direction of projection_direction.
    #     Invisible facets have zero area."""
    #     proj_dir_length = np.sum(projection_direction ** 2) ** .5
    #     projected_areas = np.sum((projection_direction / proj_dir_length) * self.normals(), axis=-1)
    #     return np.where(projected_areas > 0, projected_areas, 0)

    def projected_visible_area_fraction(self, projection_direction):
        """
        Calculate each facet's projected visible area divided by its total area.
        Obscured facets have zero projected visible area.
        :param projection_direction: direction of the projection
        :return:
        """
        proj_dir_length = np.sum(projection_direction ** 2) ** .5  # Length of projection_direction

        # Calculate dot product of unit projection direction and unit surface normals.
        proj_area_frac = np.sum((projection_direction / proj_dir_length) * self.unit_normals(), axis=-1)

        # Set obscured faces (faces with negative dot product) to zero.
        return np.where(proj_area_frac > 0, proj_area_frac, 0)


    def corners_cartesian(self):
        """Facet corner cartesian coordinates"""
        x_corners, y_corners, z_corners = \
            stellarwinds.coordinate_transforms.rectangular_coordinates_from_spherical(
                np.ones(self.polar_corners.shape),
                self.polar_corners,
                self.azimuthal_corners)

        return x_corners, y_corners, z_corners

    def centers_cartesian(self):
        """Facet center cartesian coordinates"""
        polar_centers, azimuthal_centers = self.centers()
        x_centers, y_centers, z_centers = \
            stellarwinds.coordinate_transforms.rectangular_coordinates_from_spherical(
                np.ones(polar_centers.shape),
                polar_centers,
                azimuthal_centers)

        return x_centers, y_centers, z_centers


def numerical_description(zdi_geometry, zdi_magnetogram, dest=None):
    """
    Describe field by numerically evaluating it at a set of points, then taking sums and
    averages.
    TODO test that this still works.
    :param zdi_geometry:
    :param zdi_magnetogram:
    :return:
    """

    if dest is None:
        dest = dict()

    def describe(name, values):
        """Calculate area-weighted average (mean) and find maximum value."""
        log.info("Describing %s component." % name)
        abs_max_indices = np.unravel_index(np.argmax(np.abs(values), axis=None), values.shape)
        abs_max_polar = zdi_geometry.centers()[0][abs_max_indices]
        abs_max_azimuth = zdi_geometry.centers()[1][abs_max_indices]
        abs_max = np.abs(values[abs_max_indices])

        abs_mean = np.sum(np.abs(values) * zdi_geometry.areas()) / (4 * np.pi)
        abs_rms = (np.sum(values**2 * zdi_geometry.areas()) / (4 * np.pi))**.5
        abs_std = (np.sum((np.abs(values) - abs_mean)**2 * zdi_geometry.areas()) / (4 * np.pi))**.5

        # This is a statistical identity.
        assert np.isclose(abs_rms**2, abs_mean**2 + abs_std**2), "RMS does not match mean and std."

        dest[f"magnetogram.{name}.abs.max"] = abs_max
        log.info(f"{name} |B|_max = %4.4g Gauss" % abs_max)
        log.info(f"{name} |B|_max at az=%2.2f deg, pl=%3.2f deg" % (np.rad2deg(abs_max_azimuth),
                                                                    np.rad2deg(abs_max_polar)))
        log.info(f"{name} |B|_mean = %4.4g Gauss" % abs_mean)
        log.info(f"{name} |B|_var = %4.4g Gauss" % abs_std)
        dest[f"magnetogram.{name}.abs.mean"] = abs_mean
        dest[f"magnetogram.{name}.abs.rms"] = abs_rms
        dest[f"magnetogram.{name}.abs.std"] = abs_std

        return abs_mean

    _dict = zdi_magnetogram.get_all()

    accumulated_strength_squared = np.zeros_like(zdi_geometry.centers()[0])
    for sph_dir, field in _dict.items():
        accumulated_component = np.zeros_like(zdi_geometry.centers()[0])
        for key_1, method_1 in field.items():
            values_1 = method_1(*zdi_geometry.centers())
            describe(sph_dir + "-" + key_1, values_1)
            accumulated_component += values_1

        describe(sph_dir, accumulated_component)

        accumulated_strength_squared += accumulated_component ** 2

    describe("field strength", accumulated_strength_squared**.5)

    return dest

