import argparse
import logging
log = logging.getLogger(__name__)
import numpy as np


import matplotlib.pyplot as plt
from stellarwinds.magnetogram import plot_lsd

_def = " (default: %(default)s)"
#
# Main method. Use -h for usage and help.
#
def main():
    parser = argparse.ArgumentParser(description='Visualise LSD file by plotting plot Stokes V/Ic, I/Ic, and/or null '
                                                 'signal N/Ic against Doppler velocity.')
    parser.add_argument('input_file', type=str, help='input LSD file')
    parser.add_argument('output_file', type=str, nargs='?', help='output image file')
    parser.add_argument('-p', '--params', type=str, default='VI', help='parameters, one or more from VIN' + _def)
    parser.add_argument('-s', '--skip-rows', type=int, default=2, help='number of header rows' + _def)
    parser.add_argument('-q', '--quiet', dest='log_level', action='store_const',
                        const=logging.WARNING, default=logging.INFO, help='only log warnings and errors')
    parser.add_argument('-v', '--verbose', dest='log_level', action='store_const',
                        const=logging.DEBUG, help='generate and log detailed debug output')
    args = parser.parse_args()

    log.setLevel(args.log_level)
    ch = logging.StreamHandler()
    ch.setLevel(args.log_level)
    log.addHandler(ch)

    skip_header = args.skip_rows
    parameters = args.params

    base_data = np.genfromtxt(args.input_file, skip_header=skip_header)

    fig, axs = plt.subplots(len(parameters), figsize=(8, 6), sharex=True)
    plot_lsd.plot_lsd_profile(base_data, parameters=parameters)
    axs[0].set_title('Data in %s' % args.input_file)

    if args.output_file is None:
        plt.show()
    else:
        plt.savefig(args.output_file)


if __name__ == "__main__":
    main()
