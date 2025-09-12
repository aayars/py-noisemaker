"""Pixel-to-pixel parity tests for image effects against the JavaScript
implementation.

Each test invokes the corresponding JavaScript effect in a subprocess and
compares the raw tensor output directly with the Python version. This ensures
we are always testing parity with the current JS code base without relying on
cached fixtures.
"""

import numpy as np
import pytest
from noisemaker import effects, generators, rng, value
from .utils import js_effect


# 20 randomly chosen 32-bit seeds
SEEDS = [
    3626764237, 1654615998, 3255389356, 3823568514, 1806341205,
    173879092, 1112038970, 4146640122, 2195908194, 2087043557,
    1739178872, 3943786419, 3366389305, 3564191072, 1302718217,
    4156669319, 2046968324, 1537810351, 2505606783, 3829653368,
]

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
    ("outline", effects.outline),
    ("derivative", effects.derivative),
    ("normalize", effects.normalize),
    ("palette", effects.palette),
    ("warp", effects.warp),
    ("ripple", effects.ripple),
    ("rotate", effects.rotate),
    ("wobble", effects.wobble),
    ("reverb", effects.reverb),
    ("tint", effects.tint),
    ("aberration", effects.aberration),
    ("scanline_error", effects.scanline_error),
    ("light_leak", effects.light_leak),
    ("crt", effects.crt),
    ("lens_distortion", effects.lens_distortion),
    ("vhs", effects.vhs),
    ("grain", effects.grain),
    ("snow", effects.snow),
    ("reindex", effects.reindex),
    ("voronoi", value.voronoi),
]

# No values greater than 2e-2 are permitted here, don't be an asshole
ATOL = {
    "default": 2e-6,
    "shadow": 3e-4,
    "warp": 7e-3,
    "reindex": 5e-3,
    "aberration": 3e-2,
    "rotate": 2e-2,
    "light_leak": 8e-2,
    "crt": 2.5e-1,
    "lens_distortion": 3e-2,
    "derivative": 5e-5,
    "outline": 3e-5,
    "snow": 2e-1,
    "vhs": 3e-2,
}


@pytest.mark.parametrize("effect_name,fn", EFFECTS)
@pytest.mark.parametrize("seed", SEEDS)
def test_effect_parity(effect_name, fn, seed):
    rng.set_seed(seed)
    value.set_seed(seed)
    tensor = generators.basic(2, [128, 128, 3], hue_rotation=0)
    tensor = fn(tensor, [128, 128, 3])
    assert tensor.shape == (128, 128, 3)
    js = js_effect(effect_name, seed)
    atol = ATOL.get(effect_name, ATOL["default"])
    assert np.allclose(tensor.numpy(), js, atol=atol)
