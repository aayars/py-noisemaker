"""Pixel-to-pixel parity tests against the JavaScript implementation.

Rather than comparing against canned fixtures, these tests invoke the
JavaScript reference implementation in a subprocess for each run.  The JS
script hashes its output tensors and returns the digests, which we then compare
against the Python results.  This ensures we're always testing parity with the
current JS code base.
"""

import hashlib

import pytest

from noisemaker import generators, rng, value
from .utils import generate_hashes

# 20 randomly chosen 32-bit seeds
SEEDS = [
    3626764237, 1654615998, 3255389356, 3823568514, 1806341205,
    173879092, 1112038970, 4146640122, 2195908194, 2087043557,
    1739178872, 3943786419, 3366389305, 3564191072, 1302718217,
    4156669319, 2046968324, 1537810351, 2505606783, 3829653368,
]

HASHES = generate_hashes()["generators"]

def _hash(tensor):
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

@pytest.mark.parametrize("seed", SEEDS)
def test_basic(seed):
    rng.set_seed(seed)
    value.set_seed(seed)
    tensor = generators.basic(2, [128, 128, 3], hue_rotation=0)
    assert tensor.shape == (128, 128, 3)
    assert _hash(tensor) == HASHES["basic"][seed]

@pytest.mark.parametrize("seed", SEEDS)
def test_multires(seed):
    rng.set_seed(seed)
    value.set_seed(seed)
    tensor = generators.multires(
        None,
        seed,
        freq=2,
        shape=[128, 128, 3],
        octaves=2,
        hue_rotation=0,
        post_effects=[],
        final_effects=[],
    )
    assert tensor.shape == (128, 128, 3)
    assert _hash(tensor) == HASHES["multires"][seed]
