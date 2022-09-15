import numpy as np
import cmath
import logging

log = logging.getLogger(__name__)

import stellarwinds.magnetogram.coefficients as shc


def read_magnetogram_file(file_name):
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
    # valid_types = (("radial", "poloidal", "toroidal"), ("radial",))
    # if types is None:
    #     types = valid_types[0]
    #
    # valid = False
    # for _vt in valid_types:
    #     if types == _vt:
    #         valid = True
    #
    # if not valid:
    #     raise KeyError(f"Argument \"types\" was {types}; must be one of {valid_types}.")
    #
    log.debug("Begin reading magnetogram file \"%s\"..." % file_name)

    with open(file_name) as f:
        magnetogram_file_lines = [x.strip() for x in f.readlines()]  # Remove whitespace from line ends.

    coefficient_sets = [shc.Coefficients()]

    for line_no, line in enumerate(magnetogram_file_lines):

        try:
            line_tokens = line.split()
            degree_l = int(line_tokens[0])
            order_m = int(line_tokens[1])
            real_coeff = float(line_tokens[2])
            imag_coeff = float(line_tokens[3])
            log.debug(f"Line {line_no}: \"{line}\" accepted.")
        except IndexError:
            log.debug(f"Line {line_no}: \"{line}\" ignored; has too few tokens.")
            continue
        except ValueError:
            log.debug(f"Line {line_no}: \"{line}\" ignored; tokens do not convert to ints and floats.")
            continue

        if coefficient_sets[-1].has(degree_l, order_m):
            log.debug(f"New coefficient set starting on line {line_no}: \"{line}\".")
            coefficient_sets.append(shc.empty_like(coefficient_sets[-1]))

        coefficient_sets[-1].append(degree_l, order_m, real_coeff + 1j * imag_coeff)

    log.info(f"Read {len(coefficient_sets)} coefficent sets in {len(magnetogram_file_lines)} lines.")

    sizes = np.array([c.size for c in coefficient_sets])
    if np.all(sizes == sizes[0]):
        log.info(f"Each coefficient set has {sizes[0]} elements.")
    else:
        log.warning(f"Number of elements vary; file may not read correctly.")

    log.debug("Finished reading magnetogram file \"%s\"." % file_name)
    return shc.hstack(coefficient_sets)


def write_magnetogram_file(coefficient_sets, file_name, degree_max=None, order_min=0, write_swmf_header=False):
    """

    :param coefficient_sets:
    :param file_name:
    :param degree_max:
    :param order_min:
    :return:
    """
    if degree_max is None:
        degree_max = coefficient_sets.degree_max

    log.debug("Begin writing magnetogram file \"%s\"..." % file_name)

    coefficient_data_lines = []
    for coefficients in shc.hsplit(coefficient_sets):
        coefficient_data_lines.append(["%3d  %3d  %13e  %13e" % (d, m, np.real(data), np.imag(data)) for
                              (d, m, data) in zip(*coefficients.as_arrays(degree_l_max=degree_max,
                                                                          order_m_min=order_min))])

    if write_swmf_header:
        header_lines = create_swmf_header(file_name, len(coefficient_data_lines[0]), degree_max,
                                    carrington_rotation_number=0, dlon=0.0)
    else:
        header_lines = ["Output of %s" % __file__, "Order:%d" % degree_max]

    with open(file_name, 'w') as f:
        f.write("\n".join(header_lines))
        f.write("\n")
        for data_lines in coefficient_data_lines:
            f.write("\n".join(data_lines))
            f.write("\n")

    log.info("Finished writing magnetogram file \"%s\"." % file_name)


def create_swmf_header(file_name="Default-file-name", num_data_lines=496, max_degree=30,
                       carrington_rotation_number=2077, dlon=0.0):
    """Add new style SWMF file header. The header looks like this
        CR2077_GNG.dat
        0    0.0    -2  3  2
        496    1
        30    2077    0.0
        n m g h nOrder CR dLon
    where the parameters are timestep, time, and data dimensions (-2, 3, 2); see ModReadMagnetogram.f90
    The 496,1 is the number of data lines, and 30, 2077, 0.0 is max degree, carrington rotation number and
    delta longitude. For stellar models the only parameter to change is number of data lines."""
    log.debug('Adding SWMF header')
    header = [
        str(file_name),
        "  ".join([str(s) for s in (0, 0.0, -2, 3, 2)]),
        "  ".join([str(s) for s in (num_data_lines, 1)]),
        "  ".join([str(s) for s in (max_degree, carrington_rotation_number, dlon)]),
        "  ".join([str(s) for s in ("n", "m", "g", "h", "nOrder", "CR", "dLon")])
    ]
    return header
