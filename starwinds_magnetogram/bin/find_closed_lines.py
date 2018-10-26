import argparse
import logging
log = logging.getLogger(__name__)

from stellarwinds.hypercube_rejection import hypercube_rejection
from stellarwinds.fibonacci_sphere import fibonacci_sphere
from stellarwinds.tecplot import find_closed_fieldlines


#
# Main method. Use -h for usage and help.
#
def main():
    parser = argparse.ArgumentParser(description='Find closed field lines in a TecPlot layout.')
    parser.add_argument('input_layout', type=str, help='the input  layout file')
    parser.add_argument('output_layout', type=str, nargs='?', help='the output layout file')
    parser.add_argument('--candidates', type=int, default=512, help='number of candidate field lines to consider')
    parser.add_argument('--hypercube-rejection', dest='sphere_points', action='store_const',
                        const=hypercube_rejection, default=fibonacci_sphere,
                        help='use hypercube rejection sampling to generate sphere points.')
    parser.add_argument('--keep-open-lines', dest='keep_open_lines', action='store_const',
                        const=True, default=False, help='retain the open field lines and store them in a separate zone')
    parser.add_argument('-q', '--quiet', dest='log_level', action='store_const',
                        const=logging.WARNING, default=logging.INFO, help='only log warnings and errors')
    parser.add_argument('-v', '--verbose', dest='log_level', action='store_const',
                        const=logging.DEBUG, help='generate and log detailed debug output')
    args = parser.parse_args()

    log.setLevel(args.log_level) #If the logging disappears look at the convert_magnetogram.py.

    find_closed_fieldlines(args.input_layout,
                           args.output_layout,
                           num_candidates=args.candidates,
                           keep_open_lines=args.keep_open_lines,
                           sphere_points=args.sphere_points)


if __name__ == "__main__":
    main()
