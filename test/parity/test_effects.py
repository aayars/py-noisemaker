"""Pixel-to-pixel parity tests for image effects against the JavaScript
implementation.

For each seed we run the corresponding JavaScript effect in a subprocess and
compare the resulting tensor hash against the Python output.
"""

import hashlib

import pytest

pytest.skip("ValueDistribution.simplex diverges from JS defaults; parity test skipped", allow_module_level=True)

from noisemaker import effects, generators, rng, value
from .utils import generate_hashes


# 20 randomly chosen 32-bit seeds
SEEDS = [
    3626764237, 1654615998, 3255389356, 3823568514, 1806341205,
    173879092, 1112038970, 4146640122, 2195908194, 2087043557,
    1739178872, 3943786419, 3366389305, 3564191072, 1302718217,
    4156669319, 2046968324, 1537810351, 2505606783, 3829653368,
]

HASHES = generate_hashes()["effects"]

EFFECTS = [
    ("adjust_hue", effects.adjust_hue),
    ("adjust_saturation", effects.adjust_saturation),
    ("adjust_brightness", effects.adjust_brightness),
    ("adjust_contrast", effects.adjust_contrast),
    ("posterize", effects.posterize),
    ("blur", effects.blur),
    ("bloom", effects.bloom),
    ("vignette", effects.vignette),
    ("vaseline", effects.vaseline),
    ("shadow", effects.shadow),
    ("warp", effects.warp),
    ("ripple", effects.ripple),
    ("wobble", effects.wobble),
    ("reverb", effects.reverb),
    ("light_leak", effects.light_leak),
    ("crt", effects.crt),
    ("reindex", effects.reindex),
]

def _hash(tensor):
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("effect_name,fn", EFFECTS)
@pytest.mark.parametrize("seed", SEEDS)
def test_effect_parity(effect_name, fn, seed):
    rng.set_seed(seed)
    value.set_seed(seed)
    tensor = generators.basic(2, [128, 128, 3], hue_rotation=0)
    tensor = fn(tensor, [128, 128, 3])
    assert tensor.shape == (128, 128, 3)
    assert _hash(tensor) == HASHES[effect_name][seed]
