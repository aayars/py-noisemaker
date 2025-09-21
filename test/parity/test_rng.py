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

from .seeds import LONG_SEQUENCE_SEEDS, PARITY_SEEDS
from .utils import js_rng

# Five randomly chosen 32-bit seeds shared across parity tests.
SEEDS = PARITY_SEEDS
LONG_SEEDS = LONG_SEQUENCE_SEEDS

COUNT = 10
LONG_COUNT = 1000


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


# -------- Long sequence tests --------


@pytest.mark.parametrize("seed", LONG_SEEDS)
def test_random_long_sequence(seed: int) -> None:
    rng.set_seed(seed)
    rng.reset_call_count()
    out = js_rng("random", seed, LONG_COUNT)
    expected = out["values"]
    actual = [rng.random() for _ in range(LONG_COUNT)]
    assert np.allclose(actual, expected, atol=1e-9)
    assert rng.get_call_count() == out["callCount"]
    assert rng.get_seed() == out["seed"]


@pytest.mark.parametrize("seed", LONG_SEEDS)
def test_random_class_long_sequence(seed: int) -> None:
    rng.reset_call_count()
    py_rng = rng.Random(seed)
    out = js_rng("random", seed, LONG_COUNT, scope="class")
    expected = out["values"]
    actual = [py_rng.random() for _ in range(LONG_COUNT)]
    assert np.allclose(actual, expected, atol=1e-9)
    assert rng.get_call_count() == out["callCount"]
    assert py_rng.state == out["seed"]

