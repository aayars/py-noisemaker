import math
import random as _random

from opensimplex import OpenSimplex

import numpy as np
import tensorflow as tf


_seed = None


def get_seed():
    """
    """

    global _seed
    return _seed or _random.randint(1, 65536)


def random(time, seed=None, speed=1.0):
    """Like random.random(), but returns a smooth periodic value over time."""

    two_pi_times_time = math.pi * 2 * time
    z = math.cos(two_pi_times_time) * speed
    w = math.sin(two_pi_times_time) * speed

    return (OpenSimplex(seed=seed or _random.randint(1, 65536)).noise2d(z, w) + 1.0) * .5


def simplex(shape, time=0.0, seed=None, speed=1.0, as_np=False):

    """
    Return simplex noise values. Lives in its own module to avoid circular dependencies.
    """

    tensor = np.empty(shape, dtype=np.float32)

    if seed is None:
        seed = get_seed()

    # h/t Etienne Jacob
    # https://necessarydisorder.wordpress.com/2017/11/15/drawing-from-noise-and-then-making-animated-loopy-gifs-from-there/
    two_pi_times_time = math.pi * 2 * time
    z = math.cos(two_pi_times_time) * speed
    w = math.sin(two_pi_times_time) * speed

    if len(shape) == 2:
        simplex = OpenSimplex(seed=seed)

        for y in range(shape[0]):
            for x in range(shape[1]):
                tensor[y][x] = simplex.noise4d(x, y, z, w)

    else:
        for c in range(shape[2]):
            simplex = OpenSimplex(seed=seed + c * 65535)

            for y in range(shape[0]):
                for x in range(shape[1]):
                    tensor[y][x][c] = simplex.noise4d(x, y, z, w)

    tensor = (tensor + 1.0) * .5

    if not as_np:
        tensor = tf.stack(tensor)

    return tensor
