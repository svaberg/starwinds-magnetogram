import os.path
from tests import context  # Test context

from stellarwinds.magnetogram import reader_writer
from stellarwinds.magnetogram import converter
from stellarwinds.magnetogram import coefficients
from stellarwinds.magnetogram import zdi_magnetogram


def test_read(zdi_file):

    coeffs = reader_writer.read_magnetogram_file(zdi_file).scale(converter.forward_conversion_factor)
    radial_coeffs, *_ = coefficients.hsplit(coeffs)

    reader_writer.write_magnetogram_file(radial_coeffs, file_name=os.path.dirname(zdi_file) + '/test_field_wso.dat')


def test_read_full_magnetogram(zdi_file):
    magnetogram_data = reader_writer.read_magnetogram_file(zdi_file)

    degrees, orders, coeffs = magnetogram_data.as_arrays()

    zdi_magnetogram.ZdiMagnetogram(degrees, orders, *coeffs.transpose())


