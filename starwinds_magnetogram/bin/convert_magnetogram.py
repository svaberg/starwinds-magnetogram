import argparse
import logging
import os
log = logging.getLogger(__name__)

from starwinds_magnetogram import reader_writer
from starwinds_magnetogram import coefficients
from starwinds_magnetogram import converter

from . import generic_arguments

#
# Main method. Use -h for usage and help.
#

def main():
    parser = argparse.ArgumentParser(description='Convert magnetogram files from the ZDIPy convention to WSO/Stanford PFSS convention and add the header information expected by SWMF/BATSRUS.')
    
    # generic_arguments.add(parser)
    with generic_arguments.defaults(parser):
        parser.add_argument('input_file', type=str, help='input magnetogram file')
        parser.add_argument('output_file', type=str, nargs='?', help='output magnetogram file')
        parser.add_argument('--pfss-to-zdi', dest='pfss_to_zdi', action='store_const',
                            const=True, default=False, help='Convert WSO magnetogram back to zdipy magnetogram')
        parser.add_argument('--degree-max', type=int, default=None, help='Pad magnetogram with zeros up to given degree')
        parser.add_argument('--no-header', dest='write_swmf_header', action='store_const', const=False, default=True,
                            help='Do not create new style BATSRUS/SWMF header when converting coefficients from ZDIPy to WSO format')
        parser.add_argument('--no-conversion', dest='no_conversion', action='store_const', const=True, default=False,
                            help='Do not alter the coefficients but make the format suitable for ingestion into SWMF/BATSRUS.'
                            'This is useful for converting the format of e.g. GONG files to the SWMF/BATSRUS format.')

    args = parser.parse_args(); args.debug and breakpoint()

    # Processing starts here
    if args.no_conversion:
        no_conversion(args.input_file, args.output_file, degree_max=args.degree_max, 
                      write_swmf_header=args.write_swmf_header)
    elif args.pfss_to_zdi:
        convert_pfss_to_zdi(args.input_file, args.output_file, degree_max=args.degree_max,)
    else:
        convert_zdi_to_pfss(args.input_file, args.output_file, degree_max=args.degree_max,
                            write_swmf_header=args.write_swmf_header)


def no_conversion(input_file, output_name=None, degree_max=None, write_swmf_header=False):
    """
    Convert the format of the coefficients to the SWMF/BATSRUS format without altering the coefficients.
    This is useful for converting the format of e.g. GONG files to the SWMF/BATSRUS format.
    This is equivalent to converting a WSO magnetogram to ZDIPy format and back to WSO format.
    """
    pfss_coefficients = reader_writer.read_magnetogram_file(input_file)
    if pfss_coefficients.default_coefficients.shape != (1,):
        raise IndexError(f"Incorrect shape for PFSS formatted file {input_file}.")

    output_name = _make_output_filename(input_file, output_name, postfix="wso")
    reader_writer.write_magnetogram_file(pfss_coefficients, file_name=output_name,
                                         order_min=0,
                                         degree_max=degree_max, 
                                         write_swmf_header=write_swmf_header)


def convert_zdi_to_pfss(input_file, output_name=None, degree_max=None, write_swmf_header=False):
    """
    Convert the coefficients from the ZDIPy convention to WSO/Stanford PFSS convention and add the header information expected by SWMF/BATSRUS.
    """

    zdi_coefficients = reader_writer.read_magnetogram_file(input_file)
    zdi_radial_coefficients, *_ = coefficients.hsplit(zdi_coefficients)
    pfss_coefficients = converter.convert_zdi_to_pfss(zdi_radial_coefficients)

    output_name = _make_output_filename(input_file, output_name, postfix="wso")
    reader_writer.write_magnetogram_file(pfss_coefficients, file_name=output_name, degree_max=degree_max,
                                         write_swmf_header=write_swmf_header)


def convert_pfss_to_zdi(input_file, output_name=None, degree_max=None):
    """
    Convert the coefficients from the WSO/Stanford PFSS convention to the ZDIPy convention.
    """
    pfss_coefficients = reader_writer.read_magnetogram_file(input_file)
    if pfss_coefficients.default_coefficients.shape != (1,):
        raise IndexError(f"Incorrect shape for PFSS formatted file {input_file}.")

    zdi_coefficients = converter.convert_pfss_to_zdi(pfss_coefficients)

    output_name = _make_output_filename(input_file, output_name, postfix="zdi")
    reader_writer.write_magnetogram_file(zdi_coefficients, file_name=output_name,
                                         order_min=0,
                                         degree_max=degree_max)


def _make_output_filename(input_file, output_name, postfix):
    """
    Make sure that the output file name is different from the input file name.
    Create a new file name if no output file name is given (i.e. output_name is None).
    """
    if output_name is None:
        input_file_name = os.path.basename(input_file)
        # Split the file name into the name and the extension
        input_file_name, input_file_extension = os.path.splitext(input_file_name)
        input_file_name += "_" + postfix
        output_name = input_file_name + input_file_extension

        if os.path.exists(output_name):
            raise ValueError(f"Output file {output_name} already exists. Please specify a different output file name.")

    if input_file == output_name:
        raise ValueError("Input and output file names are the same. Please provide a (different) output file name.")

    return output_name


if __name__ == "__main__":
    main()


