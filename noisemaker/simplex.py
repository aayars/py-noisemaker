import math

import numpy as np
import tensorflow as tf
import noisemaker.rng as rng

from opensimplex import OpenSimplex


_seed = None


GRADIENTS_3D = [
    -11, 4, 4, -4, 11, 4, -4, 4, 11,
    11, 4, 4, 4, 11, 4, 4, 4, 11,
    -11, -4, 4, -4, -11, 4, -4, -4, 11,
    11, -4, 4, 4, -11, 4, 4, -4, 11,
    -11, 4, -4, -4, 11, -4, -4, 4, -11,
    11, 4, -4, 4, 11, -4, 4, 4, -11,
    -11, -4, -4, -4, -11, -4, -4, -4, -11,
    11, -4, -4, 4, -11, -4, 4, -4, -11,
]


def from_seed(seed):
    r = rng.Random(seed)
    perm = [0] * 256
    perm_grad = [0] * 256
    source = list(range(256))
    grad_len = len(GRADIENTS_3D) // 3
    for i in range(255, -1, -1):
        idx = r.random_int(0, i)
        perm[i] = source[idx]
        perm_grad[i] = (perm[i] % grad_len) * 3
        source[idx] = source[i]
    os = OpenSimplex(0)
    os._perm = perm
    os._perm_grad_index_3D = perm_grad
    return os, {"perm": perm, "perm_grad": perm_grad}


def get_seed():
    """
    """

    global _seed

    if _seed is not None:
        _seed = (_seed + 1) & 0xFFFFFFFF
    else:
        _seed = rng.random_int(1, 65536)

    return _seed


def random(time, seed=None, speed=1.0):
    """Like random.random(), but returns a smooth periodic value over time.

    RNG call order:
        1. If ``seed`` is ``None``, consumes one ``rng.random_int`` to obtain it.
    """

    two_pi_times_time = math.tau * time
    z = math.cos(two_pi_times_time) * speed
    w = math.sin(two_pi_times_time) * speed

    s = seed or rng.random_int(1, 65536)
    simplex, _ = from_seed(s)
    return (simplex.noise2d(z, w) + 1.0) * 0.5


def simplex(shape, time=0.0, seed=None, speed=1.0, as_np=False):
    """Return simplex noise values. Lives in its own module to avoid circular dependencies.

    RNG call order:
        1. If ``seed`` is ``None``, ``get_seed`` consumes one ``rng.random_int``.
        2. No further global RNG values are consumed; gradients and permutations are
           derived from per-call :class:`rng.Random` instances seeded with the
           provided ``seed`` and do not affect global RNG state.
    """

    height, width = shape[0], shape[1]
    channels = shape[2] if len(shape) > 2 else 1
    tensor = np.empty((height, width, channels), dtype=np.float32)

    base_seed = seed if seed is not None else get_seed()

    angle = math.tau * time
    z = math.cos(angle) * speed

    for c in range(channels):
        os, _ = from_seed(base_seed + c * 65535)
        for y in range(height):
            for x in range(width):
                val = os.noise3d(x, y, z)
                tensor[y][x][c] = (val + 1.0) * 0.5

    if channels == 1:
        tensor = tensor[:, :, 0]

    if not as_np:
        tensor = tf.stack(tensor)

    return tensor
