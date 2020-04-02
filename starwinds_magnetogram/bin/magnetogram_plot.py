import logging
logging.basicConfig()
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
    parser.add_argument('-y', '--type', type=str, help='magnetogram type (zdi/pfss)', default='zdi')
    args = parser.parse_args()

    logging.getLogger("stellarwinds").setLevel(args.log_level)  # Set for entire stellarwinds package.

    coefficients = converter.read_magnetogram_file(args.input_file, )

    if args.type == "pfss":
        log.debug("Converting from PFSS scaling to ZDI format.")
        coefficients.apply_scaling(converter.forward_conversion_factor, -1)
    elif args.type == "zdi":
        log.debug("Using ZDI format (native).")
    else:
        raise NotImplementedError(f"Unknown magnetogram type \"{args.type}\".")

    lz = zdi_magnetogram.from_coefficients(coefficients)

    _star_name = guess_star_name_from_filename(args.input_file)

    _getters = {'radial': (lz.get_radial_field, 'B_r'),
                'polar': (lz.get_polar_field, r'B_\phi'),
                'azimuthal': (lz.get_azimuthal_field, r'B_\theta'),
                'strength': (lz.get_field_strength, '|B|')}
    _getters['azimuth'] = _getters['azimuthal']

    if args.plot_type == 'map':
        plot_zdi.plot_zdi_components(lz)
    elif args.plot_type in _getters:
        getter, latex_name = _getters[args.plot_type]
        _field_name = " ".join(getter.__name__.split("_")[1:])
        fig, ax = plot_zdi.plot_zdi_field(getter, legend_str=latex_name)
        ax.set_title(f"{_star_name} {_field_name}")
    elif args.plot_type == 'spectrum':
        fig, _ = plot_zdi.plot_energy_by_degree(lz)
    elif args.plot_type == 'matrix':
        fig, _ = plot_zdi.plot_energy(lz)
    elif args.plot_type == 'energy':
        lz.energy()
        quit(0)
    else:
        raise NotImplementedError(f"Unknown plot type \"{args.plot_type}\".")

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
