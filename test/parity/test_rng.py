"""Direct parity tests between Python and JavaScript RNG implementations."""

import json
import subprocess
from pathlib import Path

import pytest

from noisemaker import rng

# Use three seeds for coverage
SEEDS = [3626764237, 1654615998, 3255389356]
COUNT = 10


def _js_sequences(seed: int):
    script = Path(__file__).with_name("rng_sequence.js")
    result = subprocess.run(
        ["node", str(script), str(seed), str(COUNT)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


JS_DATA = {seed: _js_sequences(seed) for seed in SEEDS}


@pytest.mark.parametrize("seed", SEEDS)
def test_random(seed):
    rng.set_seed(seed)
    expected = JS_DATA[seed]["random"]
    for i, val in enumerate(expected):
        assert abs(rng.random() - val) < 1e-9, f"seed {seed} index {i}"


@pytest.mark.parametrize("seed", SEEDS)
def test_random_int(seed):
    rng.set_seed(seed)
    expected = JS_DATA[seed]["randomInt"]
    for i, val in enumerate(expected):
        assert rng.random_int(0, 99) == val, f"seed {seed} index {i}"


@pytest.mark.parametrize("seed", SEEDS)
def test_random_int_swapped(seed):
    rng.set_seed(seed)
    expected = JS_DATA[seed]["randomIntSwap"]
    for i, val in enumerate(expected):
        assert rng.random_int(99, 0) == val, f"seed {seed} index {i}"


@pytest.mark.parametrize("seed", SEEDS)
def test_choice(seed):
    seq = list(range(10))
    rng.set_seed(seed)
    expected = JS_DATA[seed]["choice"]
    for i, val in enumerate(expected):
        assert rng.choice(seq) == val, f"seed {seed} index {i}"
