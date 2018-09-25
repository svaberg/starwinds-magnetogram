import argparse
import logging

log = logging.getLogger('stellarwinds')

import matplotlib.pyplot as plt
from stellarwinds.magnetogram import plot_lsd


#
# Main method. Use -h for usage and help.
#
def main():
    parser = argparse.ArgumentParser(description='Plot LSD file')
    parser.add_argument('plot_type', type=str, help='plot type')
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

    if (args.plot_type == 'lsd'):
        plot_lsd.plot_lsd_a(args.input_file, skip_header=2)
        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file)


if __name__ == "__main__":
    main()
