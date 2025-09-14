"""Direct parity tests for RNG functions against the JavaScript implementation.

Each test invokes the corresponding JavaScript RNG in a subprocess and compares
its output directly with the Python version without relying on precomputed
fixtures. In addition to value parity, we also verify that RNG call counts and
internal seeds remain in sync between the Python and JavaScript versions for
both the module-level functions and the ``Random`` class.
"""

from __future__ import annotations

import numpy as np
import pytest

from noisemaker import rng
from .utils import js_rng

# 20 randomly chosen 32-bit seeds
SEEDS = [
    3626764237,
    1654615998,
    3255389356,
    3823568514,
    1806341205,
    173879092,
    1112038970,
    4146640122,
    2195908194,
    2087043557,
    1739178872,
    3943786419,
    3366389305,
    3564191072,
    1302718217,
    4156669319,
    2046968324,
    1537810351,
    2505606783,
    3829653368,
]

COUNT = 10


@pytest.mark.parametrize("seed", SEEDS)
def test_random(seed: int) -> None:
    rng.set_seed(seed)
    rng.reset_call_count()
    out = js_rng("random", seed, COUNT)
    expected = out["values"]
    actual = [rng.random() for _ in range(COUNT)]
    assert np.allclose(actual, expected, atol=1e-9)
    assert rng.get_call_count() == out["callCount"]
    assert rng.get_seed() == out["seed"]


@pytest.mark.parametrize("seed", SEEDS)
def test_random_int(seed: int) -> None:
    rng.set_seed(seed)
    rng.reset_call_count()
    out = js_rng("randomInt", seed, 0, 99, COUNT)
    expected = out["values"]
    actual = [rng.random_int(0, 99) for _ in range(COUNT)]
    assert actual == expected
    assert rng.get_call_count() == out["callCount"]
    assert rng.get_seed() == out["seed"]


@pytest.mark.parametrize("seed", SEEDS)
def test_random_int_swapped(seed: int) -> None:
    rng.set_seed(seed)
    rng.reset_call_count()
    out = js_rng("randomInt", seed, 99, 0, COUNT)
    expected = out["values"]
    actual = [rng.random_int(99, 0) for _ in range(COUNT)]
    assert actual == expected
    assert rng.get_call_count() == out["callCount"]
    assert rng.get_seed() == out["seed"]


@pytest.mark.parametrize("seed", SEEDS)
def test_choice(seed: int) -> None:
    seq = list(range(10))
    rng.set_seed(seed)
    rng.reset_call_count()
    out = js_rng("choice", seed, COUNT)
    expected = out["values"]
    actual = [rng.choice(seq) for _ in range(COUNT)]
    assert actual == expected
    assert rng.get_call_count() == out["callCount"]
    assert rng.get_seed() == out["seed"]


# -------- ``Random`` class parity --------


@pytest.mark.parametrize("seed", SEEDS)
def test_random_class(seed: int) -> None:
    rng.reset_call_count()
    py_rng = rng.Random(seed)
    out = js_rng("random", seed, COUNT, scope="class")
    expected = out["values"]
    actual = [py_rng.random() for _ in range(COUNT)]
    assert np.allclose(actual, expected, atol=1e-9)
    assert rng.get_call_count() == out["callCount"]
    assert py_rng.state == out["seed"]


@pytest.mark.parametrize("seed", SEEDS)
def test_random_int_class(seed: int) -> None:
    rng.reset_call_count()
    py_rng = rng.Random(seed)
    out = js_rng("randomInt", seed, 0, 99, COUNT, scope="class")
    expected = out["values"]
    actual = [py_rng.random_int(0, 99) for _ in range(COUNT)]
    assert actual == expected
    assert rng.get_call_count() == out["callCount"]
    assert py_rng.state == out["seed"]


@pytest.mark.parametrize("seed", SEEDS)
def test_choice_class(seed: int) -> None:
    seq = list(range(10))
    rng.reset_call_count()
    py_rng = rng.Random(seed)
    out = js_rng("choice", seed, COUNT, scope="class")
    expected = out["values"]
    actual = [py_rng.choice(seq) for _ in range(COUNT)]
    assert actual == expected
    assert rng.get_call_count() == out["callCount"]
    assert py_rng.state == out["seed"]

