import argparse
import logging

log = logging.getLogger('stellarwinds')

import matplotlib.pyplot as plt
from stellarwinds.magnetogram import energy_spectrum


#
# Main method. Use -h for usage and help.
#
def main():
    parser = argparse.ArgumentParser(description='Plot magnetograms')
    parser.add_argument('plot_type', type=str, help='plot type')
    parser.add_argument('input_file', type=str, help='input magnetogram file')
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

    if (args.plot_type == 'spectrum'):
        fig, ax = plt.subplots()
        energy_spectrum.spectrum_plot(args.input_file, energy_spectrum.normalisation_schmidt, ax=ax)
        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file)


if __name__ == "__main__":
    main()
