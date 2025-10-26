import math

import numpy as np
import tensorflow as tf

_seed = 0x12345678
_call_count = 0


class Random:
    """Deterministic mulberry32 RNG with independent state."""

    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF

    def random(self) -> float:
        global _call_count
        _call_count += 1
        t = (self.state + 0x6D2B79F5) & 0xFFFFFFFF
        t = (t ^ (t >> 15)) * (t | 1) & 0xFFFFFFFF
        t ^= t + ((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF
        self.state = t & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    def random_int(self, a: int, b: int) -> int:
        if b < a:
            a, b = b, a
        return int(self.random() * (b - a + 1)) + a

    def choice(self, seq):
        if not seq:
            raise IndexError("Cannot choose from an empty sequence")
        idx = self.random_int(0, len(seq) - 1)
        return seq[idx]


def set_seed(seed: int) -> None:
    """Set the global RNG seed."""
    global _seed
    _seed = seed & 0xFFFFFFFF


def get_seed() -> int:
    """Return the current RNG seed."""
    return _seed


def reset_call_count() -> None:
    """Reset the global RNG call counter."""
    global _call_count
    _call_count = 0


def get_call_count() -> int:
    """Return the number of RNG calls since last reset."""
    return _call_count


def _next() -> float:
    global _seed, _call_count
    _call_count += 1
    t = (_seed + 0x6D2B79F5) & 0xFFFFFFFF
    t = (t ^ (t >> 15)) * (t | 1)
    t &= 0xFFFFFFFF
    t ^= t + ((t ^ (t >> 7)) * (t | 61))
    t &= 0xFFFFFFFF
    _seed = t
    return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296


def random() -> float:
    """Return a float in [0,1)."""
    return _next()


def random_int(a: int, b: int) -> int:
    """Return a random integer N such that a <= N <= b."""
    if b < a:
        a, b = b, a
    return int(random() * (b - a + 1)) + a


# Compatibility alias
randint = random_int


def choice(seq):
    """Return a random element from *seq*."""
    if not seq:
        raise IndexError("Cannot choose from an empty sequence")
    idx = random_int(0, len(seq) - 1)
    return seq[idx]


def _normalize_shape(shape) -> tuple:
    """Normalize *shape* values into a concrete tuple of integers."""

    if shape is None:
        return ()

    if isinstance(shape, (int, np.integer)):
        return (int(shape),)

    if isinstance(shape, (list, tuple)):
        return tuple(int(dim) for dim in shape)

    if tf.is_tensor(shape):
        static_value = tf.get_static_value(shape)
        if static_value is not None:
            shape = static_value
        else:
            shape = shape.numpy()

    if isinstance(shape, np.ndarray):
        if shape.ndim == 0:
            return (int(shape.item()),)
        return tuple(int(dim) for dim in shape.tolist())

    return (int(shape),)


def _to_tensor(values: np.ndarray, shape: tuple, dtype) -> tf.Tensor:
    """Convert a flat numpy array to a TensorFlow tensor with the desired shape and dtype."""

    dtype = tf.dtypes.as_dtype(dtype)
    reshaped = values.reshape(shape if shape else ())
    tensor = tf.convert_to_tensor(reshaped.astype(dtype.as_numpy_dtype), dtype=dtype)
    if shape:
        return tf.reshape(tensor, shape)
    return tensor


def uniform(shape, minval=0.0, maxval=1.0, dtype=tf.float32):
    """Return a tensor of uniformly distributed random values from the custom RNG."""

    minval = float(minval)
    maxval = float(maxval)

    shape = _normalize_shape(shape)
    total = int(np.prod(shape, dtype=np.int64)) if shape else 1

    if total <= 0:
        return tf.zeros(shape, dtype=tf.dtypes.as_dtype(dtype))

    span = maxval - minval
    values = np.empty(total, dtype=np.float64)
    for i in range(total):
        values[i] = minval + span * random()

    return _to_tensor(values, shape, dtype)


def normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32):
    """Return a tensor of normally distributed random values from the custom RNG."""

    mean = float(mean)
    stddev = float(stddev)

    shape = _normalize_shape(shape)
    total = int(np.prod(shape, dtype=np.int64)) if shape else 1

    if total <= 0:
        return tf.zeros(shape, dtype=tf.dtypes.as_dtype(dtype))

    values = np.empty(total, dtype=np.float64)
    i = 0
    while i < total:
        u1 = random()
        u2 = random()

        if u1 <= 0.0:
            continue

        mag = math.sqrt(-2.0 * math.log(u1))
        z0 = mag * math.cos(2.0 * math.pi * u2)
        values[i] = mean + stddev * z0
        i += 1

        if i < total:
            z1 = mag * math.sin(2.0 * math.pi * u2)
            values[i] = mean + stddev * z1
            i += 1

    return _to_tensor(values, shape, dtype)
