import numpy as np
import logging
log = logging.getLogger(__name__)

import stellarwinds.tecplot.coordinate_transforms as ct


class ZdiGeometry:
    def __init__(self,
                 polar_corners=np.linspace(0, np.pi, 32+1),
                 azimuthal_corners=np.linspace(0, 2*np.pi, 64+1)):

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
        x_corners, y_corners, z_corners = ct._rectangular_coordinates_from_spherical(
            np.ones(self.polar_corners.shape),
            self.polar_corners,
            self.azimuthal_corners)

        return x_corners, y_corners, z_corners

    def centers_cartesian(self):
        polar_centers, azimuthal_centers = self.centers()
        x_centers, y_centers, z_centers = ct._rectangular_coordinates_from_spherical(
            np.ones(polar_centers.shape),
            polar_centers,
            azimuthal_centers)

        return x_centers, y_centers, z_centers

