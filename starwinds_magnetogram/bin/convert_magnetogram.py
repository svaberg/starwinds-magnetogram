import argparse
import logging
log = logging.getLogger(__name__)

import stellarwinds.magnetogram.converter


#
# Main method. Use -h for usage and help.
#
def main():
    parser = argparse.ArgumentParser(description='Convert from zdipy magnetogram to wso magnetogram')
    parser.add_argument('input_file', type=str, help='input magnetogram file')
    parser.add_argument('output_file', type=str, nargs='?', help='output magnetogram file')
    parser.add_argument('--inverse', dest='power', action='store_const',
                        const=-1, default=1, help='Convert wso magnetogram back to zdipy magnetogram')
    parser.add_argument('--format-only', dest='power', action='store_const',
                        const=0, help='Convert file format but do not change coefficients')
    parser.add_argument('--degree_max', type=int, default=None, help='Pad magnetogram with zeros up to given degree')
    parser.add_argument('-q', '--quiet', dest='log_level', action='store_const',
                        const=logging.WARNING, default=logging.INFO, help='only log warnings and errors')
    parser.add_argument('-v', '--verbose', dest='log_level', action='store_const',
                        const=logging.DEBUG, help='generate and log detailed debug output')
    args = parser.parse_args()

    logging.getLogger("stellarwinds").setLevel(args.log_level)  # Set for entire stellarwinds package.

    convert_magnetogram_file(args.input_file, args.output_file, power=args.power, degree_max=args.degree_max)


def convert_magnetogram_file(input_file, output_name=None, power=1, degree_max=None):
        # Make an output file name if none was given
        if output_name is None:
            file_tokens = input_file.split(".")
            file_tokens[0] += "_wso"
            output_name = ".".join(file_tokens)

        # Read input file
        coeffs = stellarwinds.magnetogram.converter.read_magnetogram_file(input_file)

        coeffs.apply_scaling(stellarwinds.magnetogram.converter.forward_conversion_factor, power)

        stellarwinds.magnetogram.converter.write_magnetogram_file(coeffs,
                                                                  fname=output_name,
                                                                  degree_max=degree_max)


if __name__ == "__main__":
    main()


