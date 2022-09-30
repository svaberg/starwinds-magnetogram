import argparse
import logging
log = logging.getLogger(__name__)
import numpy as np
import matplotlib.pyplot as plt

from starwinds_magnetogram import plot_lsd

from . import generic_arguments

_def = " (default: %(default)s)"
#
# Main method. Use -h for usage and help.
#
def main():
    parser = argparse.ArgumentParser(description='Visualise LSD file by plotting plot Stokes V/Ic, I/Ic, and/or null '
                                                 'signal N/Ic against Doppler velocity.')
    with generic_arguments.defaults(parser):
        parser.add_argument('input_file', type=str, help='input LSD file')
        parser.add_argument('output_file', type=str, nargs='?', help='output image file')
        parser.add_argument('-p', '--params', type=str, default='VI', help='parameters, one or more from VIN' + _def)
        parser.add_argument('-s', '--skip-rows', type=int, default=2, help='number of header rows' + _def)

    args = parser.parse_args(); args.debug and breakpoint()

    # Processing starts here
    base_data = np.genfromtxt(args.input_file, skip_header=args.skip_rows)

    parameters = args.params
    _, axs = plt.subplots(len(parameters), figsize=(8, 6), sharex=True)
    plot_lsd.plot_lsd_profile(base_data, parameters=parameters)
    axs[0].set_title('Data in %s' % args.input_file)

    if args.output_file is None:
        plt.show()
    else:
        plt.savefig(args.output_file)


if __name__ == "__main__":
    main()
