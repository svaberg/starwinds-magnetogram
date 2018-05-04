import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.special import lpmv
from scipy.special import factorial


# Evaluate spherical harmonics for on a polar, azimuthal grid)
def evaluate_spherical(data, polar, azimuth, r=None, r0=1, rss=3):
    if r is None:
        r = r0

    B_radial = np.zeros(azimuth.shape)
    B_polar = np.zeros(azimuth.shape)
    B_azimuthal = np.zeros(azimuth.shape)

    for row_id in range(data.shape[0]):

        l = data.item(row_id, 0);
        m = data.item(row_id, 1);
        g = data.item(row_id, 2);
        h = data.item(row_id, 3);

        Pml = lpmv(m, l, np.cos(polar));

        if True:  # Apply correction
            # https://en.wikipedia.org/wiki/Spherical_harmonics#Condon%E2%80%93Shortley_phase
            # https://en.wikipedia.org/wiki/Spherical_harmonics#Conventions
            d0 = 0 + (m == 0)
            Pml *= (-1) ** m * np.sqrt(factorial(l - m) / factorial(l + m)) * np.sqrt(2 - d0)

        # Estimate DPml
        DPml = (Pml - np.roll(Pml, 1)) / (np.cos(polar) - np.roll(np.cos(polar), 1)) * np.sin(polar)

        fixed = (r0 / r) ** (l + 2) / (l + 1 + l * (r0 / rss) ** (2 * l + 1))

        B_radial += Pml * (g * np.cos(m * azimuth) + h * np.sin(m * azimuth)) * (
                l + 1 + l * (r / rss) ** (2 * l + 1)) * fixed;
        B_polar -= DPml * (g * np.cos(m * azimuth) + h * np.sin(m * azimuth)) * (1 - (r / rss) ** (2 * l + 1)) * fixed;
        B_azimuthal += Pml * (g * np.sin(m * azimuth) - h * np.cos(m * azimuth)) * (
                1 - (r / rss) ** (2 * l + 1)) * fixed;

    return B_radial, B_polar, B_azimuthal


def pretty_plot(polar, azimuth, z, ax, crange=(), cmap='RdBu_r'):
    if len(crange) == 1:
        cmin = -crange[0]
        cmax = crange[0]
    elif len(crange) == 2:
        cmin = crange[0]
        cmax = crange[1]
    else:
        cmax = np.absolute(z[~np.isnan(z)]).max()
        cmin = np.absolute(z[~np.isnan(z)]).min()
        # cmax = np.ceil(cmax)
        # cmin = np.floor(cmin)

    # print ('Color map %s, range [%f, %f].' % (cmap, cmin, cmax))

    # Print zero contour
    ax.contour(180 / np.pi * azimuth.T, 180 / np.pi * polar.T, z.T, levels=[0], colors=('g',), linewidths=.25)

    q = ax.contourf(180 / np.pi * azimuth.T, 180 / np.pi * polar.T, z.T, 128, cmap=cmap,
                    vmin=cmin, vmax=cmax)

    ax.set_xticks(180 / np.pi * np.linspace(azimuth[0, 0], azimuth[-1, -1], 9))
    ax.set_label('Azimuth angle $\phi$')
    ax.set_yticks(180 / np.pi * np.linspace(polar[0, 0], polar[-1, -1], 5))
    ax.set_label('Polar angle $\\theta$')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    # ax.invert_xaxis() # To look more like Matthew
    ax.grid()
