import numpy as np
import logging
log = logging.getLogger(__name__)

import stellarwinds.tecplot.coordinate_transforms


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
            polar_corners = np.linspace(0, np.pi, 32 + 1)
            azimuthal_corners = np.linspace(0, 2 * np.pi, 64 + 1)

        # Only go around once.
        assert(np.abs(np.max(polar_corners) - np.min(polar_corners)) <= np.pi)
        assert(np.abs(np.max(azimuthal_corners) - np.min(azimuthal_corners)) <= 2*np.pi)

        self.polar_corners, self.azimuthal_corners = np.meshgrid(polar_corners, azimuthal_corners)

    def corners(self):
        return self.polar_corners, self.azimuthal_corners

    def centers(self):
        def make_centers(x):
            return 0.25 * (x[:-1, :-1] + x[:-1, 1:] + x[1:, :-1] + x[1:, 1:])

        polar_centers = make_centers(self.polar_corners)
        azimuthal_centers = make_centers(self.azimuthal_corners)

        assert azimuthal_centers.shape == polar_centers.shape
        return polar_centers, azimuthal_centers

    def areas(self):
        """Calculate area as difference between two spherical caps
        https://en.wikipedia.org/wiki/Spherical_cap"""

        height_delta = (np.cos(self.polar_corners[:-1, :-1]) - np.cos(self.polar_corners[:-1, 1:]))
        azimuth_delta = (self.azimuthal_corners[1:, 1:] - self.azimuthal_corners[:-1, 1:])

        return height_delta * azimuth_delta

    def corners_cartesian(self):
        x_corners, y_corners, z_corners = \
            stellarwinds.tecplot.coordinate_transforms._rectangular_coordinates_from_spherical(
                np.ones(self.polar_corners.shape),
                self.polar_corners,
                self.azimuthal_corners)

        return x_corners, y_corners, z_corners

    def centers_cartesian(self):
        polar_centers, azimuthal_centers = self.centers()
        x_centers, y_centers, z_centers = \
            stellarwinds.tecplot.coordinate_transforms._rectangular_coordinates_from_spherical(
                np.ones(polar_centers.shape),
                polar_centers,
                azimuthal_centers)

        return x_centers, y_centers, z_centers


def numerical_description(geometry, zdi):
    """
    Describe field by numerically evaluating it at a set of points, then taking sums and
    averages.
    :param geometry:
    :param zdi:
    :return:
    """

    def describe(name, values):
        log.info("Describing %s component." %  name)
        abs_max_indices = np.unravel_index(np.argmax(np.abs(values), axis=None), values.shape)
        abs_max_polar = geometry.centers()[0][abs_max_indices]
        abs_max_azimuth = geometry.centers()[1][abs_max_indices]
        abs_max = values[abs_max_indices]

        mean = np.sum(values * geometry.areas()) / (4 * np.pi)
        abs_mean = np.sum(np.abs(values) * geometry.areas()) / (4 * np.pi)

        log.info("|B|_max = %4.4g Gauss" % (abs_max))
        log.info("|B|_max at az=%2.2f deg, pl=%3.2f deg" % (np.rad2deg(abs_max_azimuth),
                                                            np.rad2deg(abs_max_polar)))
        log.info("|B|_mean = %4.4g Gauss" % abs_mean)

    _dict = zdi.get_all()

    accumulated_strength_squared = np.zeros_like(geometry.centers()[0])
    for key_0, value_0 in _dict.items():
        accumulated_component = np.zeros_like(geometry.centers()[0])
        for key_1, method_1 in value_0.items():
            values_1 = method_1(*geometry.centers())
            describe(key_0 + "-" + key_1, values_1)
            accumulated_component += values_1

        describe(key_0, accumulated_component)

        accumulated_strength_squared += accumulated_component ** 2

    describe("field strength", accumulated_strength_squared**.5)

