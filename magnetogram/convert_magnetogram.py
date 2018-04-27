import numpy as np
import logging

log = logging.getLogger(__name__)


def read_zdipy(fname):
    r"""
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

    assert(content[0] == 'General poloidal plus toroidal field')

    line1_tokens = content[1].split()

    try:
        line1_values = list(map(int, line1_tokens))
        #n_lines = line1_values[0]
        #log.info("Read number of lines: %d." % n_lines)
        assert(line1_values[1:] == [3, -3])
    except:
        log.warning("Unexpected file format on line %d: %s" % (1, contents[1]))
        n_lines = 1000
        pass

    degree_l = []
    order_m = []
    g_lm = []
    h_lm = []
    for line in content[2:]:

        try:
            line_tokens = line.split()
            if len(line_tokens) < 4:
                break

            degree_l.append(int(line_tokens[0]))
            order_m.append(int(line_tokens[1]))

            g_lm.append(float(line_tokens[2]))
            h_lm.append(float(line_tokens[3]))

        except:
            print("Warning: Unexpected file format on line %s" % line)

    log.debug("l\tm\tg_lm\th_lm")
    for line_id,_ in enumerate(degree_l):
        log.debug("%d\t%d\t%e\t%e" %(degree_l[line_id], order_m[line_id], g_lm[line_id], h_lm[line_id]))

    log.info("Finished reading magnetogram file \"%s\"." % fname)

    return degree_l, order_m, g_lm, h_lm


def write_wso(degree_l, order_m, g_lm, h_lm, fname="test_field_wso.dat"):
    log.debug("Begin writing magnetogram file \"%s\"..." % fname)

    with open(fname, 'w') as f:
        f.write("Output of %s\n" % __file__)
        f.write("Order:%d\n" % np.max(degree_l))

        for l_id in range(len(degree_l)):
            f.write("%d %d %e %e\n" % (degree_l[l_id], order_m[l_id], g_lm[l_id], h_lm[l_id]))

    log.info("Finished writing magnetogram file \"%s\"." % fname)


def convert(degree_l, order_m, g_lm, h_lm):
    r"""
    Convert Donati et al. (2006) normalized harmonic coefficients to magnetogram to
    The Wilcox Solar Observatory style
    :param degree_l:
    :param order_m:
    :param g_lm:
    :param h_lm:
    :return:
    """
    log.debug("Begin converting zdipy coefficients to wso coefficients...")

    degree_l = np.array(degree_l)
    order_m = np.array(order_m)
    g_lm = np.array(g_lm)
    h_lm = np.array(h_lm)

    conversion_factor = forward_conversion_factor(degree_l, order_m)
    g_lm_out = conversion_factor * g_lm
    h_lm_out = conversion_factor * h_lm

    log.info("Finished converting zdipy coefficients to wso coefficients.")
    return g_lm_out, h_lm_out


def convert_inv(order_m, g_lm, h_lm):
    r"""
    Inverse of convert
    :param order_m:
    :param g_lm:
    :param h_lm:
    :return:
    """
    log.debug("Begin converting wso coefficients to zdipy coefficients...")

    conversion_factor = 1.0 / forward_conversion_factor(order_m)
    g_lm_out = conversion_factor * g_lm
    h_lm_out = conversion_factor * h_lm

    log.info("Finished converting wso coefficients to zdipy coefficients.")
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
    for line_id in range(len(degree_l)):
        log.info("%d %d\t%f\t%f\t%f\t%f\t%f" % (degree_l[line_id], order_m[line_id],
                                                complex_to_real_rescaling[line_id],
                                                corton_shortley_phase[line_id],
                                                unit_sphere_factor,
                                                schmidt_scaling[line_id],
                                                conversion_factor[line_id]))
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

    degree_l, order_m, g_lm, h_lm = read_zdipy(fname)
    result = convert(order_m, g_lm, h_lm)

    write_wso(degree_l, order_m, *result, fname='test_field_wso.dat')


def convert_magnetogram(input_file, output_name=None, inverse=False):

    if output_name is None:
        file_tokens = input_file.split(".")
        file_tokens[0] += "_wso"
        output_name = ".".join(file_tokens)

    degree_l, order_m, g_lm, h_lm = read_zdipy(input_file)

    if inverse:
        log.error("Not ready")
    else:
        result = convert(degree_l, order_m, g_lm, h_lm)
        write_wso(degree_l, order_m, *result, fname=output_name)


if __name__ == "__main__":

    log.warning("Test 1")
    test_forward_conversion_factor()
    log.info("Test 1")
    test_convert()
    log.info("Test 1")
    test_back_and_forth()
    log.info("Test 1")
    test_read()
