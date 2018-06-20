import numpy as np
import logging

log = logging.getLogger(__name__)


class SphericalHarmonicsCoefficients(object):
    """Spherical harmonics coefficient data container class"""
    def __init__(self, default_coefficients):
        """Create an empty set of coefficients"""
        self.coefficients = {}
        self._default_coefficients = default_coefficients
        self._degree_max = -1

    def append(self, degree, order, data):
        """Append coefficients for a given degree and order (cannot already exist)."""
        assert (degree, order) not in self.coefficients
        self.set(degree, order, data)

    def set(self, degree, order, data):
        """Set (overwrite) coefficients for a given degree and order."""
        assert order <= degree
        assert data.shape == self._default_coefficients.shape
        self.coefficients[(degree, order)] = data
        self._degree_max = max(self._degree_max, degree)

    def get(self, degree, order):
        """Return coefficients for at given degree and order"""
        return self.coefficients.get((degree, order), self._default_coefficients)

    def degree_max(self):
        """Return highest degree in coefficient set"""
        return self._degree_max

    def size(self):
        """Return number of coefficients."""
        return len(self.coefficients.keys())

    def __str__(self):
        return str(sorted(self.coefficients))

    def apply_scaling(self, scale_function, power=1):
        """Scale by applying function to each element"""
        for (degree, order) in self.coefficients:
            factor = scale_function(degree, order) ** power
            self.coefficients[(degree, order)] *= factor

    def contents(self):
        def iterator():
            for key in sorted(self.coefficients):
                yield key, self.coefficients[key]
        return iterator()


def read_magnetogram_file(fname):
    """
    Read zdipy magnetogram file.
    :return:

    The file looks like this

    General poloidal plus toroidal field
    135 3 -3
    1  0 -2.375224e+02 -0.000000e+00
    1  1  1.931724e+01 -1.110055e+02
    (...)
    15 15 -2.021262e+01  6.270638e-01

    Only the radial coefficients are (currently) read as they are the only ones used.
    """

    log.debug("Begin reading magnetogram file \"%s\"..." % fname)
    with open(fname) as f:
        magnetogram_file_lines = f.readlines()
        # Remove whitespace characters like `\n` at the end of each line
        magnetogram_file_lines = [x.strip() for x in magnetogram_file_lines]

    header_lines = []

    coeffs = SphericalHarmonicsCoefficients(np.array([0.0, 0.0]))

    for line in magnetogram_file_lines:
        try:
            line_tokens = line.split()
            coeffs.append(int(line_tokens[0]),
                          int(line_tokens[1]),
                          np.array([float(line_tokens[2]), float(line_tokens[3])]))
            log.debug("Read coefficient line %d: \"%s\"" % (coeffs.size() + 1, line))
        except:
            if coeffs.size() == 0:
                log.debug("Read header line: %d: \"%s\"" % (len(header_lines), line))
                header_lines.append(line)
            else:
                log.debug("Read non-coefficient line \"%s\", finished reading." % line)
                break

    log.info("Read %d header lines and %d radial coefficient lines." % (len(header_lines), coeffs.size()))
    log.debug("l\tm\tg_lm\th_lm")
    for coeff in coeffs.contents():
        log.debug("%d\t%d\t%e\t%e" % (coeff[0][0],coeff[0][1],coeff[1][0],coeff[1][1]))

    log.info("Finished reading magnetogram file \"%s\"." % fname)

    return coeffs


def write_magnetogram_file(coeffs, fname="test_field_wso.dat", degree_max=None):
    log.debug("Begin writing magnetogram file \"%s\"..." % fname)

    if degree_max is None:
        degree_max = coeffs.degree_max()

    with open(fname, 'w') as f:
        f.write("Output of %s\n" % __file__)
        f.write("Order:%d\n" % degree_max)

        for degree in range(0, degree_max + 1):
            for order in range(0, degree + 1):
                data = coeffs.get(degree, order)
                f.write("%d %d %e %e\n" % (degree, order, data[0], data[1]))

    log.info("Finished writing magnetogram file \"%s\"." % fname)


def forward_conversion_factor(degree_l, order_m):
    """Conversion from zdipy format to wso format"""

    #
    # Calculate the complex-to-real rescaling factor $\sqrt{2-\delta_{m,0}$
    #
    #  The Dirac delta function $\delta_{m0}$ has
    # $\delta_{m,0} = 1$ for $m = 0$ and
    # $\delta_{m,0} = 0$ for $m \neq 0$
    delta_m0 = np.where(order_m == 0, 1, 0)
    complex_to_real_rescaling = np.sqrt(-delta_m0+2)

    #
    # Calculate the value of the Corton-Shortley phase, $(-1)^m$
    #
    corton_shortley_phase = (-1)**(order_m % 2)

    #
    # Calculate the unit sphere area compensation $\sqrt{4\pi}$
    #
    unit_sphere_factor = np.sqrt(4.0 * np.pi)

    #
    # Calculate the Schmidt scaling factor $\sqrt{2\ell+1}$
    #
    schmidt_scaling = np.sqrt(2 * degree_l + 1)

    #
    # The full conversion factor.
    #
    conversion_factor = schmidt_scaling / (corton_shortley_phase * complex_to_real_rescaling * unit_sphere_factor)

    return conversion_factor


def convert_magnetogram_file(input_file, output_name=None, power=1, degree_max=None):

    # Make an output file name if none was given
    if output_name is None:
        file_tokens = input_file.split(".")
        file_tokens[0] += "_wso"
        output_name = ".".join(file_tokens)

    # Read input file
    coeffs = read_magnetogram_file(input_file)

    coeffs.apply_scaling(forward_conversion_factor, power)

    write_magnetogram_file(coeffs, fname=output_name, degree_max=degree_max)

