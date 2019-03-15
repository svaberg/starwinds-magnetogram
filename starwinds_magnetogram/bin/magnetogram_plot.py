import argparse
import logging
import os.path

log = logging.getLogger(__name__)

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # So that a plot window actually opens.
import matplotlib.pyplot as plt
import scipy.constants
from stellarwinds.magnetogram import energy_spectrum
from stellarwinds.magnetogram import convert_magnetogram
from stellarwinds.magnetogram import zdi_lehmann
from stellarwinds.magnetogram import zdi_geometry
#
# Main method. Use -h for usage and help.
#
def main():
    parser = argparse.ArgumentParser(description='Plot magnetograms')
    parser.add_argument('input_file', type=str, help='input magnetogram file')
    parser.add_argument('output_file', type=str, nargs='?', help='output image file')
    parser.add_argument('-p', '--plot_type', type=str, help='plot type', default='map')
    parser.add_argument('-q', '--quiet', dest='log_level', action='store_const',
                        const=logging.WARNING, default=logging.INFO, help='only log warnings and errors')
    parser.add_argument('-v', '--verbose', dest='log_level', action='store_const',
                        const=logging.DEBUG, help='generate and log detailed debug output')
    args = parser.parse_args()

    logging.getLogger("stellarwinds").setLevel(args.log_level)  # Set for entire stellarwinds package.

    if args.plot_type == 'map':
        fig = plot_map(args.input_file)
    elif args.plot_type == 'spectrum':
        fig = plot_spectrum(args.input_file)

    #
    # Show or save plot
    #
    if args.output_file is None:
        plt.show()
    else:
        plt.savefig(args.output_file)

def latex_float(f):
    float_str = "{0:+.3g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

def read_lehmann_zdi(input_file):
    # Read input file
    coeffs = convert_magnetogram.read_magnetogram_file(input_file,
                                                       types=("radial",
                                                               "poloidal",
                                                               "toroidal"))
    degree_l, order_m, alpha_lm = coeffs[0].as_zdi()
    _, _, beta_lm = coeffs[1].as_zdi()
    _, _, gamma_lm = coeffs[2].as_zdi()

    alpha_lm = alpha_lm[:, 0] + 1.0j * alpha_lm[:, 1]
    beta_lm = beta_lm[:, 0] + 1.0j * beta_lm[:, 1]
    gamma_lm = gamma_lm[:, 0] + 1.0j * gamma_lm[:, 1]

    lz = zdi_lehmann.LehmannZdi(degree_l, order_m, alpha_lm, beta_lm, gamma_lm)
    return lz

def plot_map(input_file):

    lz = read_lehmann_zdi(input_file)
    zg = zdi_geometry.ZdiGeometry(61)

    polar_centers, azimuth_centers = zg.centers()

    zdi_geometry.numerical_description(zg, lz)

    b_radial = lz.get_radial_field(polar_centers, azimuth_centers)
    b_radial_max_indices = np.unravel_index(np.argmax(b_radial, axis=None), b_radial.shape)
    b_radial_max_polar = polar_centers[b_radial_max_indices]
    b_radial_max_azimuth = azimuth_centers[b_radial_max_indices]
    b_radial_max = b_radial[b_radial_max_indices]

    b_radial_min_indices = np.unravel_index(np.argmin(b_radial, axis=None), b_radial.shape)
    b_radial_min_polar = polar_centers[b_radial_min_indices]
    b_radial_min_azimuth = azimuth_centers[b_radial_min_indices]
    b_radial_min = b_radial[b_radial_min_indices]

    if np.abs(b_radial_max) > np.abs(b_radial_min):
        abs_b_radial_max = np.abs(b_radial_max)
        abs_b_radial_max_polar = b_radial_max_polar
        abs_b_radial_max_azimuth = b_radial_max_azimuth
    else:
        abs_b_radial_max = np.abs(b_radial_min)
        abs_b_radial_max_polar = b_radial_min_polar
        abs_b_radial_max_azimuth = b_radial_min_azimuth

    abs_b_radial_mean = np.sum(np.abs(b_radial) * zg.areas()) / (4 * np.pi)

    log.info("|B_r|_max = %4.4g Gauss" % abs_b_radial_max)
    log.info("|B_r|_max at az=%2.2f deg, pl=%3.2f deg" % (np.rad2deg(abs_b_radial_max_azimuth),
                                                        np.rad2deg(abs_b_radial_max_polar)))
    log.info("|B_r|_mean = %4.4g Gauss" % abs_b_radial_mean)

    fig, ax = plt.subplots(figsize=(9,6))
    polar_corners, azimuth_corners = zg.corners()
    img1 = ax.pcolormesh(np.rad2deg(azimuth_corners),
                         np.rad2deg(polar_corners),
                         b_radial,
                         vmin=-abs_b_radial_max, vmax=abs_b_radial_max,
                         cmap="RdBu_r")
    cb = fig.colorbar(img1, ax=ax)

    i1 = ax.contour(np.rad2deg(zg.centers()[1]),
                    np.rad2deg(zg.centers()[0]),
                    b_radial,
                    0,
                    linewidths=1,
                    colors='k',
                    )

    i1.collections[1].set_label('$B_r=0$ G')  # Even if there is only one line, the collections array has 3 elements.
    cb.add_lines(i1)

    i2 = ax.contour(np.rad2deg(zg.centers()[1]),
                    np.rad2deg(zg.centers()[0]),
                    b_radial,
                    cb.get_ticks(),
                    linewidths=.5,
                    colors='k',
                    linestyles='dashed'
                    )

    i2.collections[0].set_label('$\\Delta B_r = %g$ G' % (cb.get_ticks()[1] - cb.get_ticks()[0]))
    # import pdb; pdb.set_trace()
    # ax.clabel(i2, fmt='%2.1f', colors='k', fontsize=6)
    # h2, _ =
    # for _id, h2l in enumerate(i2.collections):
    #     h2l.set_label(_id)


    cb.add_lines(i2)

    # _ticks = cb.get_ticks()
    if np.abs(b_radial_max) > np.abs(b_radial_min):
        cb.ax.axhline(y=b_radial_min, color='g')
        _value_to_add = b_radial_min
    else:
        _value_to_add = b_radial_max
        cb.ax.axhline(y=b_radial_max, color='y')

    # _ticks = np.append(_ticks, _value_to_add)
    # # import pdb; pdb.set_trace()
    # cb.set_ticks(_ticks)

    ax.xaxis.set_ticks(np.arange(0, 361, 45))
    ax.yaxis.set_ticks(np.arange(0, 181, 30))
    plt.grid()

    plt.plot(np.rad2deg(b_radial_max_azimuth, ), np.rad2deg(b_radial_max_polar), 'g^', label='Max $B_r=%s$ G' % latex_float(b_radial_max))
    plt.plot(np.rad2deg(b_radial_min_azimuth, ), np.rad2deg(b_radial_min_polar), 'yv', label='Min $B_r=%s$ G' % latex_float(b_radial_min))

    star_name = os.path.basename(input_file)[6:-4]
    plt.title('%s: $|B_r|_\mathrm{mean}=%4.4g$ G' % (star_name, abs_b_radial_mean))
    ax.invert_yaxis()



    ## Dipole max
    b_radial = lz.as_dipole().get_radial_field(polar_centers, azimuth_centers)
    b_radial_max_indices = np.unravel_index(np.argmax(b_radial, axis=None), b_radial.shape)
    b_radial_max_polar = polar_centers[b_radial_max_indices]
    b_radial_max_azimuth = azimuth_centers[b_radial_max_indices]
    b_radial_max = b_radial[b_radial_max_indices]

    b_radial_min_indices = np.unravel_index(np.argmin(b_radial, axis=None), b_radial.shape)
    b_radial_min_polar = polar_centers[b_radial_min_indices]
    b_radial_min_azimuth = azimuth_centers[b_radial_min_indices]
    b_radial_min = b_radial[b_radial_min_indices]

    log.info("Dipole polar angle %g deg" % np.rad2deg(np.min([b_radial_max_polar, b_radial_min_polar])))

    plt.plot(np.rad2deg(b_radial_max_azimuth, ), np.rad2deg(b_radial_max_polar), 'g^', fillstyle='none', label='Dipole $B_r=%s$ G' % latex_float(b_radial_max))
    plt.plot(np.rad2deg(b_radial_min_azimuth, ), np.rad2deg(b_radial_min_polar), 'yv', fillstyle='none', label='Dipole $B_r=%s$ G' % latex_float(b_radial_min))

    plt.legend(ncol=3, loc='lower left')

    plt.title('%s: $|B_r|_\mathrm{mean}=%4.4g$ G. Dipole at %4.4g deg' % (star_name,
                                                                abs_b_radial_mean,
                                                                np.rad2deg(np.min([b_radial_max_polar, b_radial_min_polar]))))

    return fig


def pole_walk(lz, zg):
    """
    Forget this; the pole walks all over the place
    :param lz:
    :param zg:
    :return:
    """

    polar_centers, azimuth_centers = zg.centers()

    ## Dipole max
    b_radial_max_polar = []
    b_radial_max_azimuth = []
    b_radial_min_polar = []
    b_radial_min_azimuth = []

    for max_degree in range(lz.degree()):
        b_radial = lz.as_restricted(degree_l_range=(0, max_degree)).get_radial_field(polar_centers, azimuth_centers)
        b_radial_max_indices = np.unravel_index(np.argmax(b_radial, axis=None), b_radial.shape)
        b_radial_max_polar.append(polar_centers[b_radial_max_indices])
        b_radial_max_azimuth.append(azimuth_centers[b_radial_max_indices])
        # b_radial_max.append(b_radial[b_radial_max_indices])

        b_radial_min_indices = np.unravel_index(np.argmin(b_radial, axis=None), b_radial.shape)
        b_radial_min_polar.append(polar_centers[b_radial_min_indices])
        b_radial_min_azimuth.append(azimuth_centers[b_radial_min_indices])
        # b_radial_min = b_radial[b_radial_min_indices]

    plt.plot(np.rad2deg(b_radial_max_azimuth, ), np.rad2deg(b_radial_max_polar), 'g:')#, label='Max $B_r=%s$ G' % latex_float(b_radial_max))
    plt.plot(np.rad2deg(b_radial_min_azimuth, ), np.rad2deg(b_radial_min_polar), 'y:')#, label='Min $B_r=%s$ G' % latex_float(b_radial_min))



def plot_spectrum(input_file):
    """
    Plot the spectrum (not working)
    :param input_file:
    :return:
    """

    lz = read_lehmann_zdi(input_file)

    energy_alpha_lm, energy_beta_lm, energy_gamma_lm = lz.energy()

    ea = np.zeros(lz.degree()+1)
    eb = np.zeros(lz.degree()+1)
    eg = np.zeros(lz.degree()+1)

    for line_id, degree in enumerate(lz.degrees_l):
        ea[degree] += energy_alpha_lm[line_id]
        eb[degree] += energy_beta_lm[line_id]
        eg[degree] += energy_gamma_lm[line_id]

    degrees = np.arange(1, lz.degree()+1)
    ea = ea[1:] / (2 * scipy.constants.mu_0)
    eb = eb[1:] / (2 * scipy.constants.mu_0)
    eg = eg[1:] / (2 * scipy.constants.mu_0)




    fig, axs = plt.subplots(2, 1)

    ax=axs[0]
    ax.stackplot(degrees,
                 np.cumsum(ea)/1,
                 np.cumsum(eb)/1,
                 np.cumsum(eg)/1,
                 # labels=('$\\alpha$', '$\\beta$', '$\gamma$'),
                 )

    ax.fill_between(degrees, np.cumsum(ea + eb), y2=0, facecolor="none", hatch="||", edgecolor='k', label='poloidal')
    ax.fill_between(degrees, np.cumsum(ea + eb + eg), y2=np.cumsum(ea + eb), facecolor="none", hatch="///", edgecolor='k', label='toroidal')
    # ax.plot(degrees, np.cumsum(ea)/np.sum(ea+eb+eg), 'x-', fillstyle='none',  label='$\\alpha$')
    # ax.plot(degrees, np.cumsum(eb)/np.sum(ea+eb+eg), '+-', fillstyle='none',  label='$\\beta$')
    # ax.plot(degrees, np.cumsum(eg)/np.sum(ea+eb+eg), 'o-', fillstyle='none',  label='$\gamma$')
    # ax.plot(degrees, np.cumsum(ea + eb + eg)/np.sum(ea+eb+eg), 'k-', label='total')

    ax.set_ylabel("Cumulative pressure [Pa]")
    ax.grid()
    ax.legend(ncol=2, loc="lower right")


    ax = axs[1]


    # ax.semilogy(lz.degrees_l, energy_alpha_lm, 'x', label='$\\alpha$')
    # ax.semilogy(lz.degrees_l, energy_beta_lm, 'x',  label='$\\beta$')
    # ax.semilogy(lz.degrees_l, energy_gamma_lm, 'x', label='$\gamma$')

    # shift = np.random.rand(len(lz.degrees_l), 3) * .3
    shift = np.tile([-1, 0, 1], (len(lz.degrees_l), 1)) * .15
    ax.semilogy(lz.degrees_l + shift[:, 0], energy_alpha_lm / (2 * scipy.constants.mu_0)/np.sum(ea+eb+eg), '>', fillstyle='full', label='$\\alpha$', markersize=3)
    ax.semilogy(lz.degrees_l + shift[:, 1], energy_beta_lm / (2 * scipy.constants.mu_0)/np.sum(ea+eb+eg), 'D', fillstyle='full', label='$\\beta$', markersize=2, zorder=10)
    ax.semilogy(lz.degrees_l + shift[:, 2], energy_gamma_lm / (2 * scipy.constants.mu_0)/np.sum(ea+eb+eg), '<', fillstyle='full', label='$\\gamma$', markersize=3)
    # ax.semilogy(degrees, ea/np.sum(ea+eb+eg), 'x--', fillstyle='none', label='$\\alpha$')
    # ax.semilogy(degrees, eb/np.sum(ea+eb+eg), '+--', fillstyle='none',   label='$\\beta$')
    # ax.semilogy(degrees, eg/np.sum(ea+eb+eg), 'o--', fillstyle='none',  label='$\gamma$')
    # ax.semilogy(degrees, (ea + eb + eg)/np.sum(ea+eb+eg), 'k-', label='Sum')
    ax.set_ylabel("Contribution at $\ell$")
    ax.grid()
    ax.legend(ncol=3, loc="lower left")
    ax.set_xlabel("Degree $\ell$")


    star_name = os.path.basename(input_file)[6:-4]
    axs[0].set_title('Average magnetic pressure of %s' % star_name)


    return fig


if __name__ == "__main__":
    main()
