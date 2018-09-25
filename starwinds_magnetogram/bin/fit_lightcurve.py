import argparse
import logging

log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from stellarwinds.magnetogram import profile_shapes
from stellarwinds.magnetogram import fit_lightcurve


#
# Main method. Use -h for usage and help.
#
def main():
    available_shapes = {'Voigt': profile_shapes.FaddeevaVoigt,
                      'Gauss': profile_shapes.Gaussian,
                      'Lorentz': profile_shapes.Lorentzian}

    parser = argparse.ArgumentParser(description='Fit analytical profile to light curve.')
    parser.add_argument('profile_type', help='profile type',
                        choices=available_shapes.keys(), default='Voigt')
    parser.add_argument('input_file', type=str, help='input LSD file')
    parser.add_argument('output_file', type=str, nargs='?', help='output image file')

    parser.add_argument('--quiet', dest='log_level', action='store_const',
                        const=logging.WARNING, default=logging.INFO, help='only log warnings and errors')
    parser.add_argument('--debug', dest='log_level', action='store_const',
                        const=logging.DEBUG, help='generate and log detailed debug output')
    args = parser.parse_args()

    log.setLevel(args.log_level)
    ch = logging.StreamHandler()
    ch.setLevel(args.log_level)
    log.addHandler(ch)

    fit_lightcurve.fit_curve(args.input_file, profile=available_shapes[args.profile_type], skip_header=2)
    if args.output_file is None:
        plt.show()
    else:
        plt.savefig(args.output_file)


if __name__ == "__main__":
    main()
