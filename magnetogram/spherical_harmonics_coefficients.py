import numpy as np
import logging
log = logging.getLogger(__name__)


class SphericalHarmonicsCoefficients(object):
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
        return self._default_coefficients.copy()

    def append(self, degree, order, data):
        """Append coefficients for a given degree and order (cannot already exist)."""
        assert (degree, order) not in self.coefficients
        self.set(degree, order, data)

    def set(self, degree, order, data):
        """Set (overwrite) coefficients for a given degree and order."""
        data = np.atleast_1d(data)

        assert np.abs(order) <= degree, "Order must be between -l and l for degree l."
        assert data.shape == self._default_coefficients.shape, "Incompatible coefficients."

        self.coefficients[(degree, order)] = data
        self._degree_max = max(self._degree_max, degree)
        self._order_min = min(self._order_min, order)

    def get(self, degree, order):
        """Return coefficients for at given degree and order"""
        return self.coefficients.get((degree, order), self._default_coefficients)

    # TODO use @property
    def degree_max(self):
        """Return highest degree in coefficient set"""
        return self._degree_max

    # TODO use @property
    def order_min(self):
        """Return lowest order in coefficient set"""
        return self._order_min

    # TODO use @property
    def size(self):
        """Return number of coefficients."""
        return len(self.coefficients.keys())

    def __str__(self):
        str = "Coefficients:\n"
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

    def as_zdi(self, accept_negative_orders=False):
        """ Feel free to improve"""
        if not accept_negative_orders and self.order_min() < 0:
            raise ValueError("Will not convert negative orders to ZDI format. Use map_to_positive_orders first.")

        degrees = []
        orders = []

        for degree in range(0, self.degree_max() + 1):
            for order in range(-degree, degree + 1):
                degrees.append(degree)
                orders.append(order)

        coeffs = np.zeros((len(degrees), len(self.default_coefficients)))
        for row_id in range(len(degrees)):
                coeffs[row_id] = self.get(degrees[row_id], orders[row_id])

        return np.asarray(degrees), np.asarray(orders), coeffs

    def as_arrays(self,
                  degree_l_min=None, degree_l_max=None,
                  order_m_min=None, order_m_max=None):
        """
        Get full arrays of degrees, orders, and data.
        :param degree_l_min: Lowest degree to return.
        :param degree_l_max: Highest degree to return.
        :param order_m_min: Lowest order to return.
        :param order_m_max: Highest order to return.
        :return: degrees, orders, data
        """
        if degree_l_min is None:
            degree_l_min = 0
        if degree_l_max is None:
            degree_l_max = self.degree_max()
        if order_m_min is None:
            order_m_min = -degree_l_max
        if order_m_max is None:
            order_m_max = +degree_l_max

        # Build full list of degrees and orders
        degrees = []
        orders = []
        for degree in range(degree_l_min, degree_l_max + 1):
            _order_min = np.maximum(order_m_min, -degree)
            _order_max = np.minimum(order_m_max, degree)
            for order in range(_order_min, _order_max + 1):
                degrees.append(degree)
                orders.append(order)

        # Initialize to right size with new dimension first.
        coefficients = np.stack((self.default_coefficients,)*len(degrees))
        assert coefficients.size == len(degrees) * self.default_coefficients.size, "Sizes must match."
        assert coefficients.dtype == self.default_coefficients.dtype, "Data type must match."

        for row_id in range(len(degrees)):
                coefficients[row_id] = self.get(degrees[row_id], orders[row_id])

        return np.asarray(degrees), np.asarray(orders), coefficients

    def __add__(self, other): return add(self, other)

    def __sub__(self, other): return add(self, multiply(other, -1))

    def __mul__(self, value): return multiply(self, value)

    def __rmul__(self, value): return multiply(self, value)

    def __truediv__(self, value): return multiply(self, value**-1)


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
    return SphericalHarmonicsCoefficients(shc.default_coefficients)


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
    Stack magnetograms by using hstack on each coefficient.
    :param shcs: tuple of magnetograms
    :return: stacked magnetogram
    """

    default_coefficients = np.hstack([a.default_coefficients for a in shcs])

    shc = SphericalHarmonicsCoefficients(default_coefficients)

    # Get all degree, order pairs from the objects
    keys = set.union(*[set(a.coefficients) for a in shcs])

    for degree, order in keys:
        values = [a.get(degree, order) for a in shcs]
        c = np.hstack(values)
        shc.append(degree, order, c)

    return shc