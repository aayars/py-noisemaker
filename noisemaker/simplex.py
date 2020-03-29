import math
import random

from opensimplex import OpenSimplex

import numpy as np
import tensorflow as tf


_seed = None


def get_seed():
    """
    """

    global _seed
    return _seed or random.randint(1, 65536)


def simplex(shape, time=0.0, square=False, seed=None):
    """
    Return simplex noise values. Lives in its own module to avoid circular dependencies.
    """

    tensor = np.empty(shape, dtype=np.float32)

    if seed is None:
        seed = get_seed()

    # h/t Etienne Jacob
    # https://necessarydisorder.wordpress.com/2017/11/15/drawing-from-noise-and-then-making-animated-loopy-gifs-from-there/
    two_pi_times_time = math.pi * 2 * time
    z = math.cos(two_pi_times_time)
    w = math.sin(two_pi_times_time)

    for c in range(shape[2]):
        simplex = OpenSimplex(seed=seed + c * 65535)

        for y in range(shape[0]):
            for x in range(shape[1]):
                tensor[y][x][c] = simplex.noise4d(x, y, z, w)

    tensor = (tf.stack(tensor) + 1.0) * .5

    if square:
        tensor = tf.square(tf.square(tensor))

    return tensor
