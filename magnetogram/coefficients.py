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
        """
        Return a copy of the default coefficients of this magnetogram.
        :return:
        """
        return self._default_coefficients.copy()  # A copy is needed as the default coefficients are mutable.

    def append(self, degree, order, data):
        """Append coefficients for a given degree and order (cannot already exist)."""
        assert (degree, order) not in self.coefficients
        self.set(degree, order, data)

    def set(self, degree, order, data):
        """Set (overwrite) coefficients for a given degree and order."""
        data = np.atleast_1d(data)
        data = data.astype(self.default_coefficients.dtype)

        assert np.abs(order) <= degree, "Order must be between -l and l for degree l."
        assert data.shape == self._default_coefficients.shape, "Incompatible coefficient shapes."

        assert type(degree) == int, "Degree index must be integer"
        assert type(order) == int, "Order index must be integer"

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
        str = "Spherical harmonics coefficients:\n"
        for degree, order in sorted(self.coefficients):
            str += "%d, %d, %s\n" % (degree, order, self.get(degree, order))
        return str

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
    #
    # def as_zdi(self, accept_negative_orders=False):
    #     """ Feel free to improve"""
    #     if not accept_negative_orders and self.order_min < 0:
    #         raise ValueError("Will not convert negative orders to ZDI format. Use map_to_positive_orders first.")
    #
    #     degrees = []
    #     orders = []
    #
    #     for degree in range(0, self.degree_max + 1):
    #         for order in range(-degree, degree + 1):
    #             degrees.append(degree)
    #             orders.append(order)
    #
    #     # Create one coefficient row and use it to determine length of coefficient rows
    #     _c0 = self.default_coefficients
    #     coeffs = np.zeros((len(degrees), len(self.default_coefficients)))
    #     for row_id in range(len(degrees)):
    #         coeffs[row_id] = self.get(degrees[row_id], orders[row_id])
    #         assert np.imag(coeffs[row_id]) == 0
    #
    #     return np.asarray(degrees), np.asarray(orders), coeffs

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

    def __add__(self, other): return add(self, other)

    def __sub__(self, other): return add(self, multiply(other, -1))

    def __mul__(self, value): return multiply(self, value)

    def __rmul__(self, value): return multiply(self, value)

    def __truediv__(self, value): return multiply(self, value**-1)

    def truncated(self, degree_max): return truncated(self, degree_max)


def noise(degree_max=15, noisinator=numpy.random.normal, beta=0):
    coeffs = Coefficients()

    for deg_l in range(1, degree_max + 1):
        for order_m in range(0, deg_l + 1):
            _noise = noisinator(size=2)
            _c = _noise[0] + 1j * _noise[1]
            _c /= (2 * deg_l + 1)  # Wikipeida
            _c /= deg_l**beta
            coeffs.append(deg_l, order_m, _c)

    return coeffs


def isclose(shc0,
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

    # Get all degree, order pairs from the objects
    keys = set.union(*[set(a.coefficients) for a in (shc0, shc1)])

    c0 = []
    c1 = []

    for degree, order in keys:

        c0.append(shc0.get(degree, order))
        c1.append(shc1.get(degree, order))

    c0 = np.vstack(c0)
    c1 = np.vstack(c1)

    return np.allclose(c0, c1, **kwargs)


def empty_like(shc):
    """
    Get empty set of spherical harmonics coefficients with same type of coefficients.
    """
    return Coefficients(shc.default_coefficients)


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
