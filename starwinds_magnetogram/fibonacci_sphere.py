import logging
log = logging.getLogger(__name__)
import math
import random
import numpy as np


def fibonacci_sphere(num_points, randomize=False):
    """

    :param num_points:
    :param randomize:
    :return:
    """
    log.info("Using Fibonacci sphere algorithm.")
    points = np.empty((num_points, 3))

    rnd = 1.
    if randomize:
        rnd = random.random() * num_points

    offset = 2. / num_points
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(num_points):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % num_points) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points[i, :] = np.array((x, y, z))

    return points

