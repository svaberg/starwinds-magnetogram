import numpy as np
import logging
log = logging.getLogger(__name__)


class SphericalHarmonicsCoefficients(object):
    """
    Spherical harmonics coefficient data container class.
    Can contain a numpy array for each degree and order.
    It is not meant to 'know' its own physical meaning, it's just a container.
    """
    def __init__(self, default_coefficients):
        """Create an empty set of coefficients"""
        self.coefficients = {}
        self._default_coefficients = np.atleast_1d(default_coefficients)
        self._degree_max = 0
        self._order_min = 0

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

    def degree_max(self):
        """Return highest degree in coefficient set"""
        return self._degree_max

    def order_min(self):
        """Return lowest order in coefficient set"""
        return self._order_min

    def size(self):
        """Return number of coefficients."""
        return len(self.coefficients.keys())

    def __str__(self):
        str = ""
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

        coeffs = np.zeros((len(degrees), len(self._default_coefficients)))
        for row_id in range(len(degrees)):
                coeffs[row_id] = self.get(degrees[row_id], orders[row_id])

        return np.asarray(degrees), np.asarray(orders), coeffs

    def add(self, other):
        assert self._default_coefficients.shape == other._default_coefficients.shape, "Incompatible coefficients."

        for (degree, order), co in other.contents():
            cs = self.get(degree, order)
            self.set(degree, order, cs + co)

    def multiply(self, value):
        value = np.atleast_1d(value)
        assert self._default_coefficients.shape == value.shape, "Incompatible coefficients."

        for (degree, order), coeffs in self.contents():
            self.set(degree, order, coeffs * value)


