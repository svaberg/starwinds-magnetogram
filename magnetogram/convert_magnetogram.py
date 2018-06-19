import numpy as np
import logging

log = logging.getLogger(__name__)


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
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]

    header_lines=[]
    degree_l = []
    order_m = []
    g_lm = []
    h_lm = []
    for line in content:
        try:
            line_tokens = line.split()
            l = int(line_tokens[0])
            m = int(line_tokens[1])
            g = float(line_tokens[2])
            h = float(line_tokens[3])
            log.debug("Read coefficient line %d: \"%s\"" % (len(g_lm)+1, line))
            degree_l.append(l)
            order_m.append(m)
            g_lm.append(g)
            h_lm.append(h)
        except:
            if len(h_lm) == 0:
                log.debug("Read header line: %d: \"%s\"" % (len(header_lines), line))
                header_lines.append(line)
            else:
                break

    log.info("Read %d header lines and %d radial coefficient lines." % (len(header_lines), len(g_lm)))
    log.debug("l\tm\tg_lm\th_lm")
    for data in zip(degree_l, order_m, g_lm, h_lm):
        log.debug("%d\t%d\t%e\t%e" % data)

    log.info("Finished reading magnetogram file \"%s\"." % fname)

    return degree_l, order_m, g_lm, h_lm


def write_magnetogram_file(degree_l, order_m, g_lm, h_lm, fname="test_field_wso.dat"):
    log.debug("Begin writing magnetogram file \"%s\"..." % fname)

    with open(fname, 'w') as f:
        f.write("Output of %s\n" % __file__)
        f.write("Order:%d\n" % np.max(degree_l))

        for data in zip(degree_l, order_m, g_lm, h_lm):
            f.write("%d %d %e %e\n" % data)

    log.info("Finished writing magnetogram file \"%s\"." % fname)


def convert(degree_l, order_m, g_lm, h_lm, power=1):
    """
    Convert Donati et al. (2006) normalized harmonic coefficients to magnetogram to
    The Wilcox Solar Observatory style
    :param degree_l: Degree of spherical harmonic coefficient
    :param order_m: Power of spherical harmonic coefficient
    :param g_lm: Real-like value of spherical harmonic coefficient
    :param h_lm: Imaginary-like value of spherical harmonic coefficient
    :param power: Power of conversion factor. Use 1 for zdipy to wso and -1 for wso to zdipy.
    :return:
    """
    log.debug("Begin conversion...")

    degree_l = np.array(degree_l)
    order_m = np.array(order_m)
    g_lm = np.array(g_lm)
    h_lm = np.array(h_lm)

    conversion_factor = forward_conversion_factor(degree_l, order_m)**power
    g_lm_out = conversion_factor * g_lm
    h_lm_out = conversion_factor * h_lm

    log.info("Finished conversion.")
    return g_lm_out, h_lm_out


def forward_conversion_factor(degree_l, order_m):

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

    log.info("l m \t     ctr\t      csp\t     usf\t      ss \t       cf")
    for data in zip(degree_l, order_m,
                    complex_to_real_rescaling,
                    corton_shortley_phase,
                    np.ones_like(degree_l) * unit_sphere_factor,
                    schmidt_scaling,
                    conversion_factor):
        log.info("%d %d\t%f\t%f\t%f\t%f\t%f" % data)

    return conversion_factor


#
# Tests
#
def test_forward_conversion_factor():
    assert(np.isclose(forward_conversion_factor(0),  (4.0 * np.pi)**(-0.5)))
    assert(np.isclose(forward_conversion_factor(1), -(8.0 * np.pi)**(-0.5)))
    assert(np.isclose(forward_conversion_factor(2),  (8.0 * np.pi)**(-0.5)))
    assert(np.isclose(forward_conversion_factor(3), -(8.0 * np.pi)**(-0.5)))

    assert(np.isclose(forward_conversion_factor(98),  (8.0 * np.pi)**(-0.5)))
    assert(np.isclose(forward_conversion_factor(99), -(8.0 * np.pi)**(-0.5)))

    m = np.arange(0, 4)
    expected = np.array([(4.0 * np.pi)**(-0.5),
                        -(8.0 * np.pi)**(-0.5),
                        (8.0 * np.pi)**(-0.5),
                        -(8.0 * np.pi)**(-0.5)])
    assert(np.allclose(forward_conversion_factor(m), expected))


def test_convert():

    assert(np.allclose(convert(0, 1, 1), (4.0 * np.pi)**(-0.5)))
    assert(np.allclose(convert(1, 1, 1), -(8.0 * np.pi)**(-0.5)))
    assert(np.allclose(convert(2, 1, 1), (8.0 * np.pi)**(-0.5)))

    g_lm = np.array([-2, -1, 0, 1, 2])
    m = np.arange(0, len(g_lm))
    g_lm_out, h_lm_out = convert(m, g_lm, g_lm)
    assert(np.allclose(g_lm_out, h_lm_out))


def test_back_and_forth(n=4):
    g_lm = np.linspace(-11, 11, n)
    h_lm = np.geomspace(1e-7, 4.3e12, n)
    m = np.arange(0, len(g_lm))

    g_lm_out, h_lm_out = convert_inv(m, *convert(m, g_lm, h_lm))
    assert(np.allclose(g_lm, g_lm_out))
    assert(np.allclose(h_lm, h_lm_out))

    g_lm_out, h_lm_out = convert(m, *convert_inv(m, g_lm, h_lm))
    assert (np.allclose(g_lm, g_lm_out))
    assert (np.allclose(h_lm, h_lm_out))


def test_read(fname='test_field_zdipy.dat'):
    content = r"""General poloidal plus toroidal field
4 3 -3
 1  0 1. 1.
 1  1 1. 1.
 2  0 1. 1.
 2  1 1. 1.
 2  2 1. 1.

 1  0 100. 101.
 1  1 110. 111.
 2  0 200. 201.
 2  1 210. 211.
 2  2 220. 221.

 1  0 1000. 1010.
 1  1 1100. 1110.
 2  0 2000. 2010.
 2  1 2100. 2110.
 2  2 2200. 2210.
 """
    with open(fname, 'w') as f:
        f.write(content)

    data = read_magnetogram_file(fname)
    result = convert(*data)

    write_magnetogram_file(degree_l, order_m, *result, fname='test_field_wso.dat')


def pad_magnetogram(degree_l, order_m, g_lm, h_lm, degree_l_max):
    """
    Pad magnetogram by adding zeros up to degree $\ell$.
    :return: Padded magnetogram.
    """
    missing_degrees = range(np.max(degree_l)+1, degree_l_max+1)
    for deg in missing_degrees:
        for ord in range(0, deg+1):
            degree_l.append(deg)
            order_m.append(ord)
            g_lm.append(0.0)
            h_lm.append(0.0)


def truncate_magnetogram(degree_l, order_m, g_lm, h_lm, degree_l_max):
    """
    Truncate magnetogram to degree l
    :return: Truncated magnetogram
    """
    # Assert that the array is sorted
    assert(np.all(degree_l[:-1] <= degree_l[1:]))

    max_id = np.searchsorted(degree_l, degree_l_max, side='right')

    del degree_l[max_id:]
    del order_m[max_id:]
    del g_lm[max_id:]
    del h_lm[max_id:]


def convert_magnetogram_file(input_file, output_name=None, power=1, degree_l_max=None):

    # Make an output file name if none was given
    if output_name is None:
        file_tokens = input_file.split(".")
        file_tokens[0] += "_wso"
        output_name = ".".join(file_tokens)

    # Read input file
    degree_l, order_m, g_lm, h_lm = read_magnetogram_file(input_file)

    # Pad or truncate to max degree $\ell$.
    if degree_l_max is not None:
        log.info("Pad/truncate to degree %d" % degree_l_max)
        if degree_l_max > np.max(degree_l):
            pad_magnetogram(degree_l, order_m, g_lm, h_lm, degree_l_max)
        elif degree_l_max < np.max(degree_l):
            truncate_magnetogram(degree_l, order_m, g_lm, h_lm, degree_l_max)

    result = convert(degree_l, order_m, g_lm, h_lm, power)
    write_magnetogram_file(degree_l, order_m, *result, fname=output_name)


if __name__ == "__main__":

    log.warning("Test 1")
    test_forward_conversion_factor()
    log.info("Test 1")
    test_convert()
    log.info("Test 1")
    test_back_and_forth()
    log.info("Test 1")
    test_read()
