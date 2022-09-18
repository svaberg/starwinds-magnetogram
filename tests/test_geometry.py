import matplotlib as mpl
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.special import lpmv
from scipy.special import factorial
import logging
import pytest

# Test "context"
from tests import context  # Test context
log = logging.getLogger(__name__)
import itertools

# Local
import starwinds_magnetogram.geometry


def test_make_corners(request):
    zg = starwinds_magnetogram.geometry.ZdiGeometry()

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):
        fig, ax = plt.subplots()
        pc, ac = zg.centers()
        img1 = ax.pcolormesh(pc, ac, np.sin(pc)*np.sin(ac), edgecolor='black', facecolor=None)
        ax.invert_yaxis()
        img2 = ax.pcolormesh(ac, pc, 0*pc, edgecolor='blue', facecolor=None, alpha = .2)
        plt.savefig(pn.get())


def test_make_geometry(request):
    zg = starwinds_magnetogram.geometry.ZdiGeometry()

    polar_corners, azimuthal_corners = zg.corners()
    polar_centers, azimuthal_centers = zg.centers()

    areas = zg.areas()
    assert np.isclose(np.sum(areas), 4 * np.pi)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        fig, ax = plt.subplots()
        img1 = ax.pcolormesh(np.rad2deg(azimuthal_corners), np.rad2deg(polar_corners), areas)
        fig.colorbar(img1, ax=ax)
        fig.suptitle('Areas')
        ax.plot(np.rad2deg(azimuthal_corners), np.rad2deg(polar_corners), 'ko')
        ax.plot(np.rad2deg(azimuthal_centers), np.rad2deg(polar_centers), 'kx')
        ax.set_ylim(0, 180)
        ax.invert_yaxis()
        ax.set_xlim(0, 360)
        plt.savefig(pn.get())


def test_make_areas(request):
    zg = starwinds_magnetogram.geometry.ZdiGeometry()

    polar, azimuth = zg.corners()
    areas = zg.areas()
    assert np.isclose(np.sum(areas), 4 * np.pi)

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        fig, ax = plt.subplots()
        img1 = ax.pcolormesh(np.rad2deg(azimuth), np.rad2deg(polar), areas)
        fig.colorbar(img1, ax=ax)
        fig.suptitle('Areas')
        ax.invert_yaxis()
        plt.savefig(pn.get())


def test_make_geometry_3d(request):

    zg = starwinds_magnetogram.geometry.ZdiGeometry(
        polar_corners = np.pi * np.linspace(.25, .5, 11),
        azimuthal_corners = np.pi * np.linspace(0, 2, 21))
    corners = zg.corners_cartesian()
    centers = zg.centers_cartesian()
    areas = zg.areas()

    with context.PlotNamer(__file__, request.node.name) as (pn, plt):

        fig, ax = plt.subplots(1, 3)

        for ax_id, coords in enumerate(itertools.combinations({0, 1, 2}, 2)):
            print(ax_id, coords)
            img = ax[ax_id].pcolormesh(corners[coords[0]], corners[coords[1]], centers[0])
            ax[ax_id].plot(corners[coords[0]], corners[coords[1]], 'k-', linewidth=.1)
            # ax[ax_id].plot(centers[coords[0]], centers[coords[1]], 'o')
            ax[ax_id].plot(corners[coords[0]].transpose(), corners[coords[1]].transpose(), 'k-', linewidth=.1)
            # fig.colorbar(img, ax=ax[ax_id])
            ax[ax_id].set_xlabel("xyz"[coords[0]])
            ax[ax_id].set_ylabel("xyz"[coords[1]])

        plt.savefig(pn.get())

