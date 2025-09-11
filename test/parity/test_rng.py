"""Direct parity tests between Python and JavaScript RNG implementations."""

import pytest

from noisemaker import rng
from .utils import generate_hashes

DATA = generate_hashes()["rng"]
SEEDS = list(DATA.keys())


@pytest.mark.parametrize("seed", SEEDS)
def test_random(seed):
    rng.set_seed(seed)
    expected = DATA[seed]["random"]
    for i, val in enumerate(expected):
        assert abs(rng.random() - val) < 1e-9, f"seed {seed} index {i}"


@pytest.mark.parametrize("seed", SEEDS)
def test_random_int(seed):
    rng.set_seed(seed)
    expected = DATA[seed]["randomInt"]
    for i, val in enumerate(expected):
        assert rng.random_int(0, 99) == val, f"seed {seed} index {i}"


@pytest.mark.parametrize("seed", SEEDS)
def test_random_int_swapped(seed):
    rng.set_seed(seed)
    expected = DATA[seed]["randomIntSwap"]
    for i, val in enumerate(expected):
        assert rng.random_int(99, 0) == val, f"seed {seed} index {i}"


@pytest.mark.parametrize("seed", SEEDS)
def test_choice(seed):
    seq = list(range(10))
    rng.set_seed(seed)
    expected = DATA[seed]["choice"]
    for i, val in enumerate(expected):
        assert rng.choice(seq) == val, f"seed {seed} index {i}"
