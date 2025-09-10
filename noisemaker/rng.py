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


def set_seed(seed: int) -> None:
    """Set the global RNG seed."""
    global _seed
    _seed = seed & 0xFFFFFFFF
    tf.random.set_seed(_seed)


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


def uniform(shape, minval=0.0, maxval=1.0, dtype=tf.float32):
    """TensorFlow uniform random tensor with RNG-derived seed."""
    seed = random_int(0, 0xFFFFFFFF)
    return tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed)


def normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32):
    """TensorFlow normal random tensor with RNG-derived seed."""
    seed = random_int(0, 0xFFFFFFFF)
    return tf.random.normal(shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)
