"""Direct parity tests for RNG functions against the JavaScript implementation.

Each test invokes the corresponding JavaScript RNG in a subprocess and compares
its output directly with the Python version without relying on precomputed
fixtures.
"""

import numpy as np
import pytest

from noisemaker import rng
from .utils import js_rng

# 20 randomly chosen 32-bit seeds
SEEDS = [
    3626764237, 1654615998, 3255389356, 3823568514, 1806341205,
    173879092, 1112038970, 4146640122, 2195908194, 2087043557,
    1739178872, 3943786419, 3366389305, 3564191072, 1302718217,
    4156669319, 2046968324, 1537810351, 2505606783, 3829653368,
]

COUNT = 10


@pytest.mark.parametrize("seed", SEEDS)
def test_random(seed):
    rng.set_seed(seed)
    expected = js_rng("random", seed, COUNT)
    actual = [rng.random() for _ in range(COUNT)]
    assert np.allclose(actual, expected, atol=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_random_int(seed):
    rng.set_seed(seed)
    expected = js_rng("randomInt", seed, 0, 99, COUNT)
    actual = [rng.random_int(0, 99) for _ in range(COUNT)]
    assert actual == expected


@pytest.mark.parametrize("seed", SEEDS)
def test_random_int_swapped(seed):
    rng.set_seed(seed)
    expected = js_rng("randomInt", seed, 99, 0, COUNT)
    actual = [rng.random_int(99, 0) for _ in range(COUNT)]
    assert actual == expected


@pytest.mark.parametrize("seed", SEEDS)
def test_choice(seed):
    seq = list(range(10))
    rng.set_seed(seed)
    expected = js_rng("choice", seed, COUNT)
    actual = [rng.choice(seq) for _ in range(COUNT)]
    assert actual == expected
