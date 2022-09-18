import numpy as np
import logging
import pytest

from tests import context  # Test context

from starwinds_magnetogram import coefficients

log = logging.getLogger(__name__)

_defaults = (np.zeros(2), np.zeros(3, dtype=np.complex), np.zeros(6), 0j,
             np.ones(2), np.ones(3, dtype=np.complex), np.ones(6), 1j)
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
    c1 = coefficients.Coefficients(default)
    c1.append(1, 1, default+1)

    c2 = c1 * 2
    assert np.allclose(c2.get(1, 1), c1.get(1, 1) * 2)

    c2 = c1 * (default+1)
    log.info(c1)
    assert np.allclose(c2.get(1, 1), c1.get(1, 1) * (default + 1))

    c2 = 2 * c1
    assert np.allclose(c2.get(1, 1), c1.get(1, 1) * 2)

    c2 = c1 ** 3
    assert np.allclose(c2.get(1, 1), c1.get(1, 1) ** 3)

    # In-place
    c2 = c1.copy()
    c2 *= 2
    assert np.allclose(c2.get(1, 1), c1.get(1, 1) * 2)

    c2 = c1.copy()
    c2 /= 2
    assert np.allclose(c2.get(1, 1), c1.get(1, 1) / 2)

    # Multiply by other coefficients
    c3 = c1 * c2
    assert np.allclose(c3.get(1, 1), c1.get(1, 1) * c2.get(1, 1))

    c3 = c2 * c1
    assert np.allclose(c3.get(1, 1), c2.get(1, 1) * c1.get(1, 1))

    c3 = c1 / c2
    assert np.allclose(c3.get(1, 1), c1.get(1, 1) / c2.get(1, 1))


@pytest.mark.parametrize("default", _defaults)
def test_power(default):
    c1 = coefficients.Coefficients(default)
    c1.append(1, 1, (default + 1) * 2 + 1)

    c2 = c1**2
    assert np.allclose(c2.get(1, 1), c1.get(1, 1) ** 2)

    c2 = c1**-2
    assert np.allclose(c2.get(1, 1), c1.get(1, 1) ** -2)


@pytest.mark.parametrize("default", _defaults)
def test_hstack(default):
    c1 = coefficients.Coefficients(default)
    c1.append(1, 0, default + 1)

    c2 = coefficients.Coefficients(default)
    c2.append(1, 1, default + 1)

    c3 = coefficients.Coefficients(default)
    c3.append(2, 0, default + 1)

    c = coefficients.hstack((c1, c2, c3))
    log.info(c)


@pytest.mark.parametrize("default", (0j,))
def test_hsplit(default):
    c1 = coefficients.Coefficients(default)
    c1.append(1, 0, default + 1)

    c2 = coefficients.Coefficients(default)
    c2.append(1, 1, default + 1)

    c3 = coefficients.Coefficients(default)
    c3.append(2, 0, default + 1)

    c = coefficients.hstack((c1, c2, c3))
    log.info(c)

    splits = coefficients.hsplit(c)
    assert len(splits) == 3
    log.info("After splitting")
    for c in splits:
        log.info(c)


@pytest.mark.parametrize("default", _defaults)
def test_allclose(default):
    c1 = coefficients.Coefficients(default)
    c1.append(1, 0, default + 1)
    c1.append(1, 1, default + 1)

    assert coefficients.allclose(c1, c1)


@pytest.mark.parametrize("default", _defaults)
def test_allclose_rtol(default):
    c1 = coefficients.Coefficients(default)
    c1.append(1, 0, default + 1)
    c1.append(1, 1, default + 1)

    c2 = c1.copy()
    c2.set(1, 1, 1.005 * (default + 1))

    assert coefficients.allclose(c1, c2, rtol=1e-2)
    assert not coefficients.allclose(c1, c2, rtol=1e-3)


@pytest.mark.parametrize("default", _defaults)
def test_allclose_atol(default):
    c1 = coefficients.Coefficients(default)
    c1.append(1, 0, default + 1)
    c1.append(1, 1, 0 * default)

    c2 = c1.copy()
    c2.set(1, 1, 5e-3 + 0 * default)

    assert coefficients.allclose(c1, c2, atol=1e-2)
    assert not coefficients.allclose(c1, c2, atol=1e-3)
