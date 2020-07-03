import numpy as np
import numpy.random
import logging
log = logging.getLogger(__name__)


class Coefficients(object):
    """
    Spherical harmonics coefficient data container class.
    Can contain a numpy array for each degree and order.
    It is not meant to 'know' its own physical meaning, it's just a container.
    """
    def __init__(self, default_coefficients=0j):
        """Create an empty set of coefficients"""
        self.coefficients = {}
        self._default_coefficients = np.atleast_1d(default_coefficients)
        self._degree_max = 0
        self._order_min = 0

    @property
    def default_coefficients(self):
        """Return a copy of the default coefficients of this magnetogram. """
        return self._default_coefficients.copy()  # A copy is needed as the default coefficients are mutable.

    def append(self, degree, order, data):
        """Append coefficients for a given degree and order (cannot already exist)."""
        if (degree, order) in self.coefficients:
            raise IndexError(f"Already added coefficient with l={degree}, m={order}.")
        self.set(degree, order, data)

    def set(self, degree, order, data):
        """Set (overwrite) coefficients for a given degree and order."""
        data = np.atleast_1d(data)
        data = data.astype(self.default_coefficients.dtype)

        assert np.abs(order) <= degree, "Order must be between -l and l for degree l."
        assert data.shape == self._default_coefficients.shape, "Incompatible coefficient shapes."

        degree = int(degree)  # Integer please
        order = int(order)

        self.coefficients[(degree, order)] = data
        self._degree_max = max(self._degree_max, degree)
        self._order_min = min(self._order_min, order)

    def has(self, degree, order):
        """Return true if coefficients exist at given degree and order"""
        return (degree, order) in self.coefficients.keys()

    def get(self, degree, order):
        """Return coefficients for at given degree and order"""

        assert type(degree) == int, "Degree index must be integer"
        assert type(order) == int, "Order index must be integer"

        return self.coefficients.get((degree, order), self._default_coefficients)

    @property
    def degree_max(self):
        """Return highest degree in coefficient set"""
        return self._degree_max

    @property
    def order_min(self):
        """Return lowest order in coefficient set"""
        return self._order_min

    @property
    def size(self):
        """Return number of coefficients."""
        return len(self.coefficients.keys())

    def __str__(self):
        if len(self.coefficients) == 0:
            return "Empty spherical harmonics coefficients with element like %s." % str(self.default_coefficients)

        str_ = "Spherical harmonics coefficients:\n"
        for degree, order in sorted(self.coefficients):
            str_ += "%d, %d, %s\n" % (degree, order, self.get(degree, order))
        return str_

    def contents(self):
        def iterator():
            for key in sorted(self.coefficients):
                yield key, self.coefficients[key]
        return iterator()

    def as_arrays(self, include_unset=True,
                  degree_l_min=None, degree_l_max=None,
                  order_m_min=None, order_m_max=None):
        """
        Get full arrays of degrees, orders, and data.
        :param include_unset: Include coefficients that have never been set and have the default (zero) value.
        :param degree_l_min: Lowest degree to return.
        :param degree_l_max: Highest degree to return.
        :param order_m_min: Lowest order to return.
        :param order_m_max: Highest order to return.
        :return: degrees, orders, data
        """
        if degree_l_min is None:
            degree_l_min = 0
        if degree_l_max is None:
            degree_l_max = self.degree_max
        if order_m_min is None:
            order_m_min = -degree_l_max
        if order_m_max is None:
            order_m_max = +degree_l_max

        # Build full list of degrees and orders
        full_set_requested_degrees = []
        full_set_requested_orders = []
        for degree in range(degree_l_min, degree_l_max + 1):
            _order_min = np.maximum(order_m_min, -degree)
            _order_max = np.minimum(order_m_max, degree)
            for order in range(_order_min, _order_max + 1):
                full_set_requested_degrees.append(degree)
                full_set_requested_orders.append(order)

        # Initialize to right size with new dimension first.
        coefficients = np.stack((self.default_coefficients,)*len(full_set_requested_degrees))
        assert coefficients.size == len(full_set_requested_degrees) * self.default_coefficients.size, "Sizes must match."
        assert coefficients.dtype == self.default_coefficients.dtype, "Data type must match."

        returned_degrees = []
        returned_orders = []
        returned_coefficients = []
        for _deg, _ord in zip(full_set_requested_degrees, full_set_requested_orders):
            if include_unset or self.has(_deg, _ord):
                returned_degrees.append(_deg)
                returned_orders.append(_ord)
                returned_coefficients.append(self.get(_deg, _ord))

        return np.asarray(returned_degrees), np.asarray(returned_orders), np.asarray(returned_coefficients)

    def copy(self): return copy(self)

    def scale(self, scale_function, power=1): return scale(self, scale_function, power)

    def __add__(self, other): return add(self, other)

    def __sub__(self, other): return add(self, multiply(other, -1))

    def __mul__(self, value): return multiply(self, value)

    def __rmul__(self, value): return multiply(self, value)

    def __truediv__(self, value): return multiply(self, value**-1)

    def truncated(self, degree_max): return truncated(self, degree_max)


def from_arrays(degrees_l,
                      orders_m,
                      coeffs_lm):

    coeffs = Coefficients(0 * coeffs_lm[0])
    for deg_l, ord_m, coeff in zip(degrees_l, orders_m, coeffs_lm):
        coeffs.append(deg_l, ord_m, coeff)

    return coeffs


def noise(degree_max=15, noise_fn=numpy.random.normal, beta=0):
    coeffs = Coefficients()

    for deg_l in range(1, degree_max + 1):
        for order_m in range(0, deg_l + 1):
            _noise = noise_fn(size=2)
            _c = _noise[0] + 1j * _noise[1]
            _c /= (2 * deg_l + 1)  # Wikipeida
            _c /= deg_l**beta
            coeffs.append(deg_l, order_m, _c)

    return coeffs


def allclose(shc0,
             shc1,
             **kwargs):
    """
    Numerical comparison of two sets of spherical harmonics coefficients.
    :param shc0: First set
    :param shc1: Second set

    :return: True if the sets are close, otherwise False.

    """
    if shc0.default_coefficients.shape != shc1.default_coefficients.shape:
        return False

    bool_coeffs = isclose(shc0, shc1, **kwargs)
    bool_values = np.stack(v for k, v in bool_coeffs.contents())

    return np.all(bool_values)


def isclose(shc0, shc1, **kwargs):

    keys = set.union(*[set(a.coefficients) for a in (shc0, shc1)])

    c0 = np.vstack([shc0.get(degree, order) for degree, order in keys])
    c1 = np.vstack([shc1.get(degree, order) for degree, order in keys])

    coeffs = Coefficients(default_coefficients=False)  # The default coefficient is a boolean
    for (degree, order), bool_ in zip(keys, np.isclose(c0, c1, **kwargs)):
        coeffs.append(degree, order, bool_)
    return coeffs


def empty_like(shc):
    """
    Get empty set of spherical harmonics coefficients with same type of coefficients.
    """
    return Coefficients(shc.default_coefficients)


def zeros_like(shc):
    """
    Get zero-filled set of spherical harmonics coefficients with same type of coefficients.
    """
    output = empty_like(shc)
    for (degree, order), coeffs in shc.contents():
        output.append(degree, order, shc.default_coefficients)
    return output


def copy(shc):
    """
    Get a deep copy
    :param shc:
    :return:
    """
    output = empty_like(shc)
    for (degree, order), coeffs in shc.contents():
        output.append(degree, order, coeffs)
    return output


def scale(shc, scale_function, power=1):
    """Make a scaled copy of the coefficients. Scale by applying function to each element."""

    output = copy(shc)

    for (degree, order) in shc.coefficients:
        factor = scale_function(degree, order) ** power
        output.coefficients[(degree, order)] *= factor

    return output


def truncated(shc, degree_max):
    """
    Get coefficients truncated to degree
    :param shc: coefficients object
    :param degree_max: degree to which to truncate
    :return: truncated coefficients
    """
    output = empty_like(shc)
    for (degree, order), coeffs in shc.contents():
        if degree <= degree_max:
            output.append(degree, order, coeffs)

    return output


def add(shc0, shc1):
    """
    Add coefficients.

    Missing coefficients are assumed to be zero.
    :param shc0: First set of coefficients.
    :param shc1: Second set of coefficients.
    :return: Sum of coefficients.
    """
    assert shc0.default_coefficients.shape == shc1.default_coefficients.shape, "Incompatible coefficients."

    # Get all degree, order pairs from the objects
    keys = set.union(*[set(a.coefficients) for a in (shc0, shc1)])

    shc = empty_like(shc0)
    for degree, order in keys:
        v = shc0.get(degree, order) + shc1.get(degree, order)
        shc.append(degree, order, v)

    return shc


def multiply(shc, value):
    """
    Multiply coefficients by constant.
    :param shc: Spherical harmonics coefficients
    :param value: Constant (must be addable to each
    :return:
    """

    output = empty_like(shc)
    for (degree, order), coeffs in shc.contents():
        output.append(degree, order, coeffs * value)

    return output


def hstack(shcs):
    """
    Create a set of spherical harmonics coefficients by horizontally stacking (appending)
    each of the individual of coefficients together.
    :param shcs: tuple of magnetograms
    :return: stacked magnetogram
    """

    default_coefficients = np.hstack([a.default_coefficients for a in shcs])

    shc = Coefficients(default_coefficients)

    # Get all degree, order pairs from the objects
    keys = set.union(*[set(a.coefficients) for a in shcs])

    for degree, order in keys:
        values = [a.get(degree, order) for a in shcs]
        c = np.hstack(values)
        shc.append(degree, order, c)

    return shc


def hsplit(shc, indices_or_sections=None):
    """
    Split spherical harmonics coefficients by horizontally splitting each of the individual coefficients.
    :param shc: Spherical harmonics coefficients
    :param indices_or_sections:
    :return:
    """
    if indices_or_sections is None:
        indices_or_sections = len(shc.default_coefficients)

    default_coefficients = np.hsplit(shc.default_coefficients, indices_or_sections)

    shcs = [Coefficients(_dc) for _dc in default_coefficients]

    for (degree, order), value in shc.contents():
        split_values = np.hsplit(value, indices_or_sections)

        for _shc, _split_value in zip(shcs, split_values):
            _shc.append(degree, order, _split_value)

    return shcs
