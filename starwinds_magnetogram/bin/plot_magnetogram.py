import logging
logging.basicConfig()
log = logging.getLogger(__name__)

import argparse
import os.path
import matplotlib.pyplot as plt

from starwinds_magnetogram import reader_writer
from starwinds_magnetogram import converter
from starwinds_magnetogram import zdi_magnetogram
from starwinds_magnetogram import plot_zdi

from . import generic_arguments

def main():
    parser = argparse.ArgumentParser(
        description="""Plot magnetograms. 
        Plot type can be map, radial, polar, azimuthal, strength,
        energy-by-degree, energy-matrix, or energy-summary""")

    with generic_arguments.defaults(parser):
        parser.add_argument('input_file', type=str, help='input magnetogram file')
        parser.add_argument('output_file', type=str, nargs='?', help='output image file')
        parser.add_argument('-p', '--plot_type', type=str, help='plot type', default='map')
        parser.add_argument('-y', '--type', type=str, help='magnetogram type (zdi/pfss)', default='zdi')

    args = parser.parse_args(); args.debug and breakpoint()

    # Processing starts here
    coefficients = reader_writer.read_magnetogram_file(args.input_file, )

    if args.type == "pfss":
        log.debug("Converting from PFSS scaling to ZDI format.")
        coefficients = converter.convert_pfss_to_zdi(coefficients)
    elif args.type == "zdi":
        log.debug("Using ZDI format (native).")
    else:
        raise ValueError(f"Unknown magnetogram type \"{args.type}\".")

    lz = zdi_magnetogram.from_coefficients(coefficients)

    _star_name = guess_star_name_from_filename(args.input_file)

    _getters = {'radial': (lz.get_radial_field, 'B_r'),
                'polar': (lz.get_polar_field, r'B_\phi'),
                'azimuthal': (lz.get_azimuthal_field, r'B_\theta'),
                'strength': (lz.get_field_strength, '|B|')}
    _getters['azimuth'] = _getters['azimuthal']

    if args.plot_type == 'map':
        axs = plot_zdi.plot_zdi_components(lz)
        fig = axs[0].figure
    elif args.plot_type in _getters:
        getter, latex_name = _getters[args.plot_type]
        _field_name = " ".join(getter.__name__.split("_")[1:])
        fig, ax = plot_zdi.plot_zdi_field(getter, legend_str=latex_name)
        ax.set_title(f"{_star_name} {_field_name}")
    elif args.plot_type == 'energy-by-degree':
        fig, _ = plot_zdi.plot_energy_by_degree(lz)
    elif args.plot_type == 'energy-matrix':
        fig, _ = plot_zdi.plot_energy_matrix(lz)
    elif args.plot_type == 'energy-summary':
        fig, _ = plot_zdi.plot_energy_summary(lz)
    else:
        raise ValueError(f"Unknown plot type \"{args.plot_type}\".")

    #
    # Show or save plot
    #
    if args.output_file is None:
        plt.show()
    else:
        fig.savefig(args.output_file, bbox_inches="tight")
        plt.close()


def guess_star_name_from_filename(input_file):
    return os.path.basename(input_file)[6:-4]


if __name__ == "__main__":
    main()
