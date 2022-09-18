import numpy as np
import scipy.special


def calculate_lpmn(degrees_l, orders_m, points_polar):
    r"""
    Use scipy.lpmn to calculate the derivatives of the associated Legendre polynomial. The values returned
    are $P(\cos\theta)$ and $\partial \theta P(\cos \theta)$.
    :param points_polar: point polar angle values
    :return: tuple of values and derivative values. The last index corresponds to the order and degree.
    """
    max_degree = np.max(degrees_l)
    max_order = np.max(orders_m)
    Pmn_cos_theta_result = np.empty(points_polar.shape + (max_order + 1, max_degree + 1))
    Pmn_d_cos_theta_result = np.empty_like(Pmn_cos_theta_result)

    for ndindex in np.ndindex(points_polar.shape):
        a, b = scipy.special.lpmn(m=max_order,
                                  n=max_degree,
                                  z=np.cos(points_polar[ndindex]))
        Pmn_cos_theta_result[ndindex] = a
        Pmn_d_cos_theta_result[ndindex] = b * -np.sin(points_polar[ndindex])

    Pmn_cos_theta_result = Pmn_cos_theta_result[..., orders_m, degrees_l]
    Pmn_d_cos_theta_result = Pmn_d_cos_theta_result[..., orders_m, degrees_l]

    return Pmn_cos_theta_result, Pmn_d_cos_theta_result