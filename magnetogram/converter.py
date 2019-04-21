import numpy as np
import cmath
import logging

log = logging.getLogger(__name__)

import stellarwinds.magnetogram.coefficients as shc


def collect_cosines(r, alpha, s, beta):
    """

    :param r:
    :param alpha:
    :param s:
    :param beta:
    :return:
    """
    cos_terms = r * np.cos(alpha) + s * np.cos(beta)
    sin_terms = r * np.sin(alpha) - s * np.sin(beta)

    t = np.sqrt(cos_terms**2 + sin_terms**2)
    gamma = np.arctan2(sin_terms, cos_terms)

    return t, gamma


def collect_sines(r, alpha, s, beta):
    """

    :param r:
    :param alpha:
    :param s:
    :param beta:
    :return:
    """
    cos_terms = r * np.cos(alpha) - s * np.cos(beta)
    sin_terms = r * np.sin(alpha) + s * np.sin(beta)

    t = np.sqrt(cos_terms**2 + sin_terms**2)
    gamma = np.arctan2(sin_terms, cos_terms)

    return t, gamma


def map_to_positive_orders(magnetogram):
    """

    :param magnetogram:
    :return:
    """
    output = shc.empty_like(magnetogram)
    for degree_l in range(magnetogram.degree_max + 1):
        output.append(degree_l, 0, magnetogram.get(degree_l, 0))
        for order_m in range(1, degree_l + 1):  # No need to map m=0 as it has no negative partner.
            c_pos = magnetogram.get(degree_l, + order_m)
            c_neg = magnetogram.get(degree_l, - order_m)

            r, alpha = cmath.polar(c_pos)
            s, beta  = cmath.polar(c_neg)

            t, gamma = collect_cosines(r, alpha, (-1) ** order_m * s, beta)
            # t, gamma = collect_cosines(r, alpha, s, beta)
            # log.debug('collect_c', (r, alpha), (s, beta), (t, gamma))

            c = cmath.rect(t, gamma)
            output.append(degree_l, order_m, c)
    return output


# TODO small change here to return 3 shparm objects
def read_magnetogram_file(fname, types=("radial",)):
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

    1  0 -2.375224e+02 -0.000000e+00
    1  1  1.931724e+01 -1.110055e+02
    (...)
    15 15 -2.021262e+01  6.270638e-01

    1  0 -2.375224e+02 -0.000000e+00
    1  1  1.931724e+01 -1.110055e+02
    (...)
    15 15 -2.021262e+01  6.270638e-01
    """

    log.debug("Begin reading magnetogram file \"%s\"..." % fname)
    with open(fname) as f:
        magnetogram_file_lines = f.readlines()
        # Remove whitespace characters like `\n` at the end of each line
        magnetogram_file_lines = [x.strip() for x in magnetogram_file_lines]


    full_coeffs = []
    line_offset = 0

    for coeffs_types in types:
        header_lines = []

        coeffs = shc.Coefficients()

        for __line_id, line in enumerate(magnetogram_file_lines[line_offset:]):
            line_no = __line_id + line_offset
            try:
                line_tokens = line.split()

                degree_l = int(line_tokens[0])
                order_m  = int(line_tokens[1])
                real_coeff = float(line_tokens[2])
                imag_coeff = float(line_tokens[3])

                coeffs.append(degree_l, order_m, real_coeff + 1j * imag_coeff)
                log.debug("Read coefficient line %d: \"%s\"" % (line_no, line))
            except (ValueError, IndexError):
                if coeffs.size == 0:
                    log.debug("Read header line: %d: \"%s\"" % (len(header_lines), line))
                    header_lines.append(line)
                else:
                    log.debug("Read non-coefficient line \"%s\", finished reading %s." % (line, coeffs_types))
                    line_offset = line_no
                    break

        log.debug("Read %d header lines and %d %s coefficient lines." % (len(header_lines), coeffs.size, coeffs_types))
        log.debug("l\tm\tg_lm\th_lm")
        # for coeff in coeffs.contents():
        #     log.debug("%d\t%d\t%e\t%e" % (coeff[0][0],coeff[0][1],coeff[1][0],coeff[1][1]))

        full_coeffs.append(coeffs)

    log.debug("Finished reading magnetogram file \"%s\"." % fname)
    return shc.hstack(full_coeffs)


def write_magnetogram_file(coeffs, fname="test_field_wso.dat", degree_max=None):
    """

    :param coeffs:
    :param fname:
    :param degree_max:
    :return:
    """
    log.debug("Begin writing magnetogram file \"%s\"..." % fname)

    if degree_max is None:
        degree_max = coeffs.degree_max

    # TODO This should use .arrays() or .zdi() of the magnetogram.
    with open(fname, 'w') as f:
        f.write("Output of %s\n" % __file__)
        f.write("Order:%d\n" % degree_max)

        for degree in range(0, degree_max + 1):
            for order in range(0, degree + 1):
                coeff = coeffs.get(degree, order)
                f.write("%d %d %e %e\n" % (degree, order, np.real(coeff), np.imag(coeff)))

    log.info("Finished writing magnetogram file \"%s\"." % fname)


def forward_conversion_factor(degree_l, order_m):
    """
    Conversion from zdipy format to wso format
    :param degree_l:
    :param order_m:
    :return:
    """

    #
    # Calculate the complex-to-real rescaling factor $\sqrt{2-\delta_{m,0}}$
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
