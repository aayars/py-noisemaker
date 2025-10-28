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

from .seeds import PARITY_SEEDS
from .utils import js_effect


# Five randomly chosen 32-bit seeds shared across parity tests.
SEEDS = PARITY_SEEDS

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
    ("glowing_edges", effects.glowing_edges),
    ("derivative", effects.derivative),
    ("sobel", effects.sobel_operator),
    ("normalize", effects.normalize),
    ("ridge", effects.ridge),
    ("palette", effects.palette),
    ("color_map", effects.color_map),
    ("false_color", effects.false_color),
    ("warp", effects.warp),
    ("ripple", effects.ripple),
    ("sine", effects.sine),
    ("rotate", effects.rotate),
    ("wobble", effects.wobble),
    ("reverb", effects.reverb),
    ("tint", effects.tint),
    ("aberration", effects.aberration),
    ("scanline_error", effects.scanline_error),
    ("light_leak", effects.light_leak),
    ("crt", effects.crt),
    ("lens_warp", effects.lens_warp),
    ("lens_distortion", effects.lens_distortion),
    ("vhs", effects.vhs),
    ("grain", effects.grain),
    ("fxaa", value.fxaa),
    ("snow", effects.snow),
    ("smoothstep", value.smoothstep),
    ("reindex", effects.reindex),
    ("voronoi", value.voronoi),
    ("clouds", effects.clouds),
    ("conv_feedback", effects.conv_feedback),
    ("convolve", value.convolve),
    ("degauss", effects.degauss),
    ("density_map", effects.density_map),
    ("dla", effects.dla),
    ("erosion_worms", effects.erosion_worms),
    ("fibers", effects.fibers),
    ("frame", effects.frame),
    ("glyph_map", effects.glyph_map),
    ("grime", effects.grime),
    ("jpeg_decimate", effects.jpeg_decimate),
    ("kaleido", effects.kaleido),
    ("lowpoly", effects.lowpoly),
    ("nebula", effects.nebula),
    ("normal_map", effects.normal_map),
    ("on_screen_display", effects.on_screen_display),
    ("pixel_sort", effects.pixel_sort),
    ("refract", value.refract),
    ("scratches", effects.scratches),
    ("simple_frame", effects.simple_frame),
    ("sketch", effects.sketch),
    ("spatter", effects.spatter),
    ("spooky_ticker", effects.spooky_ticker),
    ("stray_hair", effects.stray_hair),
    ("texture", effects.texture),
    ("value_refract", effects.value_refract),
    ("vortex", effects.vortex),
    ("wormhole", effects.wormhole),
    ("worms", effects.worms),
]

# Keep tolerances as tight as observed parity gaps allow; large values should be
# treated as temporary until the implementations converge.
# Note: After fixing the time==0 animation bug in value.py, input tensors to effects
# changed slightly, requiring tolerance adjustments for effects with cascading calculations.
ATOL = {
    "default": 1e-4,  # Increased from 2e-6 due to cascading precision in color space conversions
    "aberration": 2.3e-2,
    "crt": 2.2e-1,
    "derivative": 5.5e-5,
    "glowing_edges": 1e-5,
    "lens_distortion": 2.25e-2,
    "light_leak": 7.6e-2,
    "lens_warp": 7e-3,
    "outline": 2.5e-5,
    "sobel": 4e-5,
    "reindex": 4.1e-3,
    "ripple": 4e-6,
    "rotate": 1.2e-2,
    "shadow": 2e-4,
    "snow": 1.73e-1,
    "vaseline": 5e-3,
    "vhs": 2.7e-2,
    "vignette": 9e-3,
    "warp": 6.9e-3,
}


@pytest.mark.parametrize("effect_name,fn", EFFECTS)
@pytest.mark.parametrize("seed", SEEDS)
def test_effect_parity(effect_name, fn, seed):
    rng.set_seed(seed)
    value.set_seed(seed)
    tensor = generators.basic(2, [128, 128, 3], hue_rotation=0)
    shape = [128, 128, 3]
    if effect_name == "color_map":
        clut = value.values(freq=[4, 4], shape=shape)
        tensor = fn(tensor, shape, clut=clut)
    else:
        tensor = fn(tensor, shape)
    assert tensor.shape == (128, 128, 3)
    js = js_effect(effect_name, seed)
    atol = ATOL.get(effect_name, ATOL["default"])
    assert np.allclose(tensor.numpy(), js, atol=atol)
