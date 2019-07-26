import logging
log = logging.getLogger(__name__)
import argparse
import os.path

from stellarwinds.magnetogram import converter
from stellarwinds.magnetogram import zdi_magnetogram
from stellarwinds.magnetogram import plot_zdi


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

    coefficients = converter.read_magnetogram_file(args.input_file)
    lz = zdi_magnetogram.from_coefficients(coefficients)

    if args.plot_type == 'map':
        fig, _ = plot_zdi.plot_map(lz, guess_star_name_from_filename(args.input_file))
    elif args.plot_type == 'spectrum':
        fig, _ = plot_zdi.plot_energy_by_degree(lz)
    elif args.plot_type == 'matrix':
        fig, _ = plot_zdi.plot_energy(lz)
    elif args.plot_type == 'walk':
        fig, _ = plot_zdi.pole_walk(lz)

    #
    # Show or save plot
    #
    if args.output_file is None:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        fig.savefig(args.output_file, bbox_inches="tight")


def guess_star_name_from_filename(input_file):
    return os.path.basename(input_file)[6:-4]


if __name__ == "__main__":
    main()
