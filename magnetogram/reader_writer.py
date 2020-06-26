import numpy as np
import cmath
import logging

log = logging.getLogger(__name__)

import stellarwinds.magnetogram.coefficients as shc


def read_magnetogram_file(fname, types=None):
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
    valid_types = (("radial", "poloidal", "toroidal"), ("radial",))
    if types is None:
        types = valid_types[0]

    valid = False
    for _vt in valid_types:
        if types == _vt:
            valid = True

    if not valid:
        raise KeyError(f"Argument \"types\" was {types}; must be one of {valid_types}.")

    log.debug("Begin reading magnetogram file \"%s\"..." % fname)
    with open(fname) as f:
        magnetogram_file_lines = f.readlines()
        # Remove whitespace characters like `\n` at the end of each line
        magnetogram_file_lines = [x.strip() for x in magnetogram_file_lines]


    full_coeffs = []
    line_offset = 0

    # Note this is hacky: Only the length of types is really used.
    # Also, the code should automatically determine whether there is one or three sets of coefficients.
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
                # log.debug("Read coefficient line %d: \"%s\"" % (line_no, line))
            except (ValueError, IndexError):
                if coeffs.size == 0:
                    # log.debug("Read header line: %d: \"%s\"" % (len(header_lines), line))
                    header_lines.append(line)
                else:
                    # log.debug("Read non-coefficient line \"%s\", finished reading %s." % (line, coeffs_types))
                    line_offset = line_no
                    break

        log.info("Read %d header lines and %d %s coefficient lines." % (len(header_lines), coeffs.size, coeffs_types))
        # log.debug("l\tm\tg_lm\th_lm")
        # for coeff in coeffs.contents():
        #     log.debug("%d\t%d\t%e\t%e" % (coeff[0][0],coeff[0][1],coeff[1][0],coeff[1][1]))

        full_coeffs.append(coeffs)

    log.debug("Finished reading magnetogram file \"%s\"." % fname)
    return shc.hstack(full_coeffs)


def write_magnetogram_file(coeffs, fname, degree_max=None):
    """

    :param coeffs:
    :param fname:
    :param degree_max:
    :return:
    """
    log.debug("Begin writing magnetogram file \"%s\"..." % fname)

    if degree_max is None:
        degree_max = coeffs.degree_max

    if coeffs.default_coefficients.shape != (1,):
        log.error("Can only write radial components")
        raise NotImplementedError("Can only write radial components")

    # TODO This should use .arrays() or .zdi() of the magnetogram.
    with open(fname, 'w') as f:
        f.write("Output of %s\n" % __file__)
        f.write("Order:%d\n" % degree_max)

        for degree in range(0, degree_max + 1):
            for order in range(0, degree + 1):
                coeff = coeffs.get(degree, order)
                f.write("%3d  %3d  %13e  %13e\n" % (degree, order, np.real(coeff), np.imag(coeff)))

    log.info("Finished writing magnetogram file \"%s\"." % fname)


