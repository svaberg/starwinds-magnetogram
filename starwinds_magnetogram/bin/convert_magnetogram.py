import argparse
import logging
log = logging.getLogger(__name__)

from starwinds_magnetogram import reader_writer
from starwinds_magnetogram import coefficients
from starwinds_magnetogram import converter

from . import generic_arguments

#
# Main method. Use -h for usage and help.
#

def main():
    parser = argparse.ArgumentParser(description='Convert from ZDIPy magnetogram to WSO/Stanford PFSS magnetogram')
    
    generic_arguments.add(parser)
    parser.add_argument('input_file', type=str, help='input magnetogram file')
    parser.add_argument('output_file', type=str, nargs='?', help='output magnetogram file')
    parser.add_argument('--pfss-to-zdi', dest='pfss_to_zdi', action='store_const',
                        const=True, default=False, help='Convert WSO magnetogram back to zdipy magnetogram')
    parser.add_argument('--degree-max', type=int, default=None, help='Pad magnetogram with zeros up to given degree')
    parser.add_argument('--no-header', dest='write_swmf_header', action='store_const', const=False, default=True,
                        help='Do not create new style SWMF header when converting coefficients from ZDIPy to WSO format')

    args = generic_arguments.handle(parser); args.debug and breakpoint()

    # Processing starts here
    if args.pfss_to_zdi:
        convert_pfss_to_zdi(args.input_file, args.output_file, degree_max=args.degree_max,)
    else:
        convert_zdi_to_pfss(args.input_file, args.output_file, degree_max=args.degree_max,
                            write_swmf_header=args.write_swmf_header)


def convert_zdi_to_pfss(input_file, output_name=None, degree_max=None, write_swmf_header=False):

    zdi_coefficients = reader_writer.read_magnetogram_file(input_file)
    zdi_radial_coefficients, *_ = coefficients.hsplit(zdi_coefficients)
    pfss_coefficients = converter.convert_zdi_to_pfss(zdi_radial_coefficients)

    output_name = _make_output_file(input_file, output_name, postfix="wso")
    reader_writer.write_magnetogram_file(pfss_coefficients, file_name=output_name, degree_max=degree_max,
                                         write_swmf_header=write_swmf_header)


def convert_pfss_to_zdi(input_file, output_name=None, degree_max=None):

    pfss_coefficients = reader_writer.read_magnetogram_file(input_file)
    if pfss_coefficients.default_coefficients.shape != (1,):
        raise IndexError(f"Incorrect shape for PFSS formatted file {input_file}.")

    zdi_coefficients = converter.convert_pfss_to_zdi(pfss_coefficients)

    output_name = _make_output_file(input_file, output_name, postfix="zdi")
    reader_writer.write_magnetogram_file(zdi_coefficients, file_name=output_name,
                                         order_min=0,
                                         degree_max=degree_max)


def _make_output_file(input_file, output_name, postfix):
    # Make an output file name if none was given
    if output_name is None:
        file_tokens = input_file.split(".")
        file_tokens[0] += "_" + postfix
        output_name = ".".join(file_tokens)
    return output_name


if __name__ == "__main__":
    main()


