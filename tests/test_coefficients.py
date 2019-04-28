import numpy as np
import logging
import pytest

from tests import context  # Test context

from stellarwinds.magnetogram import coefficients

log = logging.getLogger(__name__)

_defaults = (np.zeros(2), np.zeros(3, dtype=np.complex), np.zeros(6), 0j,)
@pytest.mark.parametrize("default", _defaults)
def test_methods_not_failing(default):
    data = default + 1
    degree = 0
    order = 0

    c = coefficients.Coefficients(default)

    c.append(degree, order, data)
    c.set(degree, order, data)
    c.get(degree, order)
    c.degree_max
    c.order_min
    c.size
    c.__str__()
    # c.apply_scaling(scale_function, power=1)
    c.contents()
    c.as_arrays()

    # THis is dodgy
    # c.as_zdi(accept_negative_orders=False)


@pytest.mark.parametrize("default", _defaults)
def test_add(default):
    c = coefficients.Coefficients(default)
    c.append(1, 1, default + 1)
    log.info(c)

    c = c + c
    log.info(c)

    d = coefficients.Coefficients(default)
    d.append(1, -1, default - 2)
    d.append(1, 1, default + 1)
    c = c + d
    log.info(c)


@pytest.mark.parametrize("default", _defaults)
def test_subtract(default):
    c = coefficients.Coefficients(default)
    c.append(1, 1, default + 1)
    log.info(c)

    c = c - c
    log.info(c)

    d = coefficients.Coefficients(default)
    d.append(1, -1, default - 2)
    d.append(1, 1, default + 1)
    c = c - d
    log.info(c)


@pytest.mark.parametrize("default", _defaults)
def test_multiply(default):
    c = coefficients.Coefficients(default)
    c.append(1, 1, default+1)
    log.info(c)

    c = c * 2
    log.info(c)

    c = c * (default+1)
    log.info(c)

    c = 2 * c
    log.info(c)

    c *= 2
    log.info(c)

    c /= 2
    log.info(c)


@pytest.mark.parametrize("default", _defaults)
def test_hstack(default):
    c1 = coefficients.Coefficients(default)
    c1.append(1, 0, default-1)

    c2 = coefficients.Coefficients(default)
    c2.append(1, 1, default-1)

    c3 = coefficients.Coefficients(default)
    c3.append(2, 0, default-1)

    c = coefficients.hstack((c1, c2, c3))
    log.info(c)


@pytest.mark.parametrize("default", (0j,))
def test_hsplit(default):
    c1 = coefficients.Coefficients(default)
    c1.append(1, 0, default-1)

    c2 = coefficients.Coefficients(default)
    c2.append(1, 1, default-1)

    c3 = coefficients.Coefficients(default)
    c3.append(2, 0, default-1)

    c = coefficients.hstack((c1, c2, c3))
    log.info(c)

    splits = coefficients.hsplit(c)
    assert len(splits) == 3
    log.info("After splitting")
    for c in splits:
        log.info(c)


@pytest.mark.parametrize("default", _defaults)
def test_isclose(default):
    c1 = coefficients.Coefficients(default)
    c1.append(1, 0, default-1)
    c1.append(1, 1, default-1)

    assert coefficients.isclose(c1, c1)