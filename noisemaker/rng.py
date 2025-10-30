"""Deterministic random number generation for Noisemaker."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
import tensorflow as tf

_seed: int = 0x12345678
_call_count: int = 0

T = TypeVar("T")


class Random:
    """Deterministic mulberry32 RNG with independent state."""

    def __init__(self, seed: int):
        """
        Initialize a new random number generator.

        Args:
            seed: Random seed value
        """
        self.state = seed & 0xFFFFFFFF

    def random(self) -> float:
        """
        Return a random float in [0, 1).

        Returns:
            Random float value
        """
        global _call_count
        _call_count += 1
        t = (self.state + 0x6D2B79F5) & 0xFFFFFFFF
        t = (t ^ (t >> 15)) * (t | 1) & 0xFFFFFFFF
        t ^= t + ((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF
        self.state = t & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    def random_int(self, a: int, b: int) -> int:
        """
        Return a random integer N such that a <= N <= b.

        Args:
            a: Minimum value (inclusive)
            b: Maximum value (inclusive)

        Returns:
            Random integer in range [a, b]
        """
        if b < a:
            a, b = b, a
        return int(self.random() * (b - a + 1)) + a

    def choice(self, seq: Sequence[T]) -> T:
        """
        Return a random element from sequence.

        Args:
            seq: Sequence to choose from

        Returns:
            Random element from sequence

        Raises:
            IndexError: If sequence is empty
        """
        if not seq:
            raise IndexError("Cannot choose from an empty sequence")
        idx = self.random_int(0, len(seq) - 1)
        return seq[idx]


def set_seed(seed: int) -> None:
    """
    Set the global RNG seed.

    Args:
        seed: Random seed value
    """
    global _seed
    _seed = seed & 0xFFFFFFFF


def get_seed() -> int:
    """
    Return the current RNG seed.

    Returns:
        Current seed value
    """
    return _seed


def reset_call_count() -> None:
    """Reset the global RNG call counter."""
    global _call_count
    _call_count = 0


def get_call_count() -> int:
    """
    Return the number of RNG calls since last reset.

    Returns:
        Number of RNG calls
    """
    return _call_count


def _next() -> float:
    """Internal function to generate next random value."""
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
    """
    Return a random float in [0, 1).

    Returns:
        Random float value
    """
    return _next()


def random_int(a: int, b: int) -> int:
    """
    Return a random integer N such that a <= N <= b.

    Args:
        a: Minimum value (inclusive)
        b: Maximum value (inclusive)

    Returns:
        Random integer in range [a, b]
    """
    if b < a:
        a, b = b, a
    return int(random() * (b - a + 1)) + a


# Compatibility alias
randint = random_int


def choice(seq: Sequence[T]) -> T:
    """
    Return a random element from sequence.

    Args:
        seq: Sequence to choose from

    Returns:
        Random element from sequence

    Raises:
        IndexError: If sequence is empty
    """
    if not seq:
        raise IndexError("Cannot choose from an empty sequence")
    idx = random_int(0, len(seq) - 1)
    return seq[idx]


def _normalize_shape(shape: Any) -> tuple:
    """
    Normalize shape values into a concrete tuple of integers.

    Args:
        shape: Shape specification (None, int, list, or tuple)

    Returns:
        Normalized shape as tuple
    """

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


def uniform(shape, minval=0.0, maxval=1.0, dtype=tf.float32) -> tf.Tensor:
    """
    Return a tensor of uniformly distributed random values from the custom RNG.

    Args:
        shape: Shape specification for output tensor
        minval: Minimum value (inclusive), default 0.0
        maxval: Maximum value (exclusive), default 1.0
        dtype: TensorFlow data type, default tf.float32

    Returns:
        Tensor of uniformly distributed random values in [minval, maxval)
    """

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


def normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32) -> tf.Tensor:
    """
    Return a tensor of normally distributed random values from the custom RNG.

    Uses Box-Muller transform to generate normally distributed values from uniform random values.

    Args:
        shape: Shape specification for output tensor
        mean: Mean of the normal distribution, default 0.0
        stddev: Standard deviation of the normal distribution, default 1.0
        dtype: TensorFlow data type, default tf.float32

    Returns:
        Tensor of normally distributed random values with specified mean and stddev
    """

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
