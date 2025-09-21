"""Pixel-to-pixel parity tests against the JavaScript implementation.

These tests invoke the JavaScript reference implementation in a subprocess for
each run and compare the raw tensor outputs directly with the Python version.
This ensures we're always testing parity with the current JS code base without
relying on canned fixtures.
"""

import numpy as np
import pytest

from noisemaker import generators, rng, value
from noisemaker.constants import (
    ColorSpace,
    InterpolationType,
    ValueDistribution,
    ValueMask,
)

from .seeds import PARITY_SEEDS
from .utils import js_generator

# Five randomly chosen 32-bit seeds shared across parity tests.
SEEDS = PARITY_SEEDS

@pytest.mark.parametrize("seed", SEEDS)
def test_basic(seed):
    rng.set_seed(seed)
    value.set_seed(seed)
    rng.reset_call_count()
    tensor = generators.basic(2, [128, 128, 3])
    assert tensor.shape == (128, 128, 3)
    js, js_calls = js_generator("basic", seed)
    assert np.allclose(tensor.numpy(), js, atol=1e-6)
    assert rng.get_call_count() == js_calls


@pytest.mark.parametrize("seed", SEEDS)
def test_basic_grayscale_sin(seed):
    rng.set_seed(seed)
    value.set_seed(seed)
    rng.reset_call_count()
    tensor = generators.basic(
        2,
        [128, 128, 1],
        color_space=ColorSpace.grayscale,
        sin=1.2,
    )
    assert tensor.shape == (128, 128, 1)
    js, js_calls = js_generator(
        "basic",
        seed,
        shape=[128, 128, 1],
        color_space=ColorSpace.grayscale.value,
        sin=1.2,
    )
    assert np.allclose(tensor.numpy(), js, atol=1e-6)
    assert rng.get_call_count() == js_calls


@pytest.mark.parametrize("seed", SEEDS)
def test_basic_oklab(seed):
    rng.set_seed(seed)
    value.set_seed(seed)
    rng.reset_call_count()
    tensor = generators.basic(2, [128, 128, 3], color_space=ColorSpace.oklab)
    assert tensor.shape == (128, 128, 3)
    js, js_calls = js_generator(
        "basic",
        seed,
        color_space=ColorSpace.oklab.value,
    )
    assert np.allclose(tensor.numpy(), js, atol=1e-5)
    assert rng.get_call_count() == js_calls


@pytest.mark.parametrize("seed", SEEDS)
def test_basic_constant_ridges(seed):
    rng.set_seed(seed)
    value.set_seed(seed)
    rng.reset_call_count()
    tensor = generators.basic(
        2,
        [128, 128, 3],
        ridges=True,
        spline_order=InterpolationType.constant,
        color_space=ColorSpace.rgb,
        distrib=ValueDistribution.simplex,
    )
    assert tensor.shape == (128, 128, 3)
    js, js_calls = js_generator(
        "basic",
        seed,
        ridges=True,
        splineOrder=InterpolationType.constant.value,
        color_space=ColorSpace.rgb.value,
        distrib=ValueDistribution.simplex.value,
    )
    assert np.allclose(tensor.numpy(), js, atol=1e-6)
    assert rng.get_call_count() == js_calls


@pytest.mark.parametrize("seed", SEEDS)
def test_basic_center_brightness_mask_constant(seed):
    rng.set_seed(seed)
    value.set_seed(seed)
    rng.reset_call_count()
    tensor = generators.basic(
        2,
        [128, 128, 3],
        distrib=ValueDistribution.center_hexagon,
        spline_order=InterpolationType.constant,
        brightness_distrib=ValueDistribution.simplex,
        brightness_freq=[3, 4],
        mask=ValueMask.truchet_tile_03,
        color_space=ColorSpace.rgb,
        sin=1.0,
    )
    assert tensor.shape == (128, 128, 3)
    js, js_calls = js_generator(
        "basic",
        seed,
        distrib=ValueDistribution.center_hexagon.value,
        splineOrder=InterpolationType.constant.value,
        brightnessDistrib=ValueDistribution.simplex.value,
        brightnessFreq=[3, 4],
        mask=ValueMask.truchet_tile_03.value,
        color_space=ColorSpace.rgb.value,
        sin=1.0,
    )
    assert np.allclose(tensor.numpy(), js, atol=1e-6)
    assert rng.get_call_count() == js_calls


@pytest.mark.parametrize("seed", SEEDS)
def test_multires(seed):
    rng.set_seed(seed)
    value.set_seed(seed)
    rng.reset_call_count()
    tensor = generators.multires(
        None,
        seed,
        freq=2,
        shape=[128, 128, 3],
        octaves=2,
        post_effects=[],
        final_effects=[],
    )
    assert tensor.shape == (128, 128, 3)
    js, js_calls = js_generator("multires", seed)
    assert np.allclose(tensor.numpy(), js, atol=1e-6)
    assert rng.get_call_count() == js_calls


@pytest.mark.parametrize("seed", SEEDS)
def test_multires_single_octave(seed):
    rng.set_seed(seed)
    value.set_seed(seed)
    rng.reset_call_count()
    tensor = generators.multires(
        None,
        seed,
        freq=2,
        shape=[128, 128, 3],
        octaves=1,
        post_effects=[],
        final_effects=[],
    )
    assert tensor.shape == (128, 128, 3)
    js, js_calls = js_generator("multires", seed, octaves=1)
    assert np.allclose(tensor.numpy(), js, atol=1e-6)
    assert rng.get_call_count() == js_calls


@pytest.mark.parametrize("seed", SEEDS)
def test_multires_single_octave_hue_rotation(seed):
    rng.set_seed(seed)
    value.set_seed(seed)
    rng.reset_call_count()
    tensor = generators.multires(
        None,
        seed,
        freq=2,
        shape=[128, 128, 3],
        octaves=1,
        hue_rotation=0,
        post_effects=[],
        final_effects=[],
    )
    assert tensor.shape == (128, 128, 3)
    js, js_calls = js_generator(
        "multires",
        seed,
        octaves=1,
        hue_rotation=0,
    )
    assert np.allclose(tensor.numpy(), js, atol=1e-6)
    assert rng.get_call_count() == js_calls


@pytest.mark.parametrize("seed", SEEDS)
def test_multires_frequency_list_single_octave(seed):
    rng.set_seed(seed)
    value.set_seed(seed)
    rng.reset_call_count()
    shape = [128, 96, 3]
    freq = [2, 3]
    tensor = generators.multires(
        None,
        seed,
        freq=freq,
        shape=shape,
        octaves=1,
        post_effects=[],
        final_effects=[],
    )
    assert tensor.shape == (128, 96, 3)
    js, js_calls = js_generator(
        "multires",
        seed,
        freq=freq,
        shape=shape,
        octaves=1,
    )
    assert js.shape == (128, 96, 3)
    assert np.allclose(tensor.numpy(), js, atol=1e-6)
    assert rng.get_call_count() == js_calls


@pytest.mark.parametrize("seed", SEEDS)
def test_multires_hue_distrib(seed):
    rng.set_seed(seed)
    value.set_seed(seed)
    rng.reset_call_count()
    tensor = generators.multires(
        None,
        seed,
        freq=2,
        shape=[128, 128, 3],
        octaves=1,
        hue_distrib=ValueDistribution.simplex,
        post_effects=[],
        final_effects=[],
    )
    assert tensor.shape == (128, 128, 3)
    js, js_calls = js_generator(
        "multires",
        seed,
        octaves=1,
        hueDistrib=ValueDistribution.simplex.value,
    )
    assert np.allclose(tensor.numpy(), js, atol=1e-6)
    assert rng.get_call_count() == js_calls


@pytest.mark.parametrize("seed", SEEDS)
def test_multires_supersample(seed):
    rng.set_seed(seed)
    value.set_seed(seed)
    rng.reset_call_count()
    tensor = generators.multires(
        None,
        seed,
        freq=2,
        shape=[128, 128, 3],
        octaves=2,
        hue_rotation=0,
        with_supersample=True,
        post_effects=[],
        final_effects=[],
    )
    assert tensor.shape == (128, 128, 3)
    js, js_calls = js_generator(
        "multires",
        seed,
        withSupersample=True,
    )
    assert np.allclose(tensor.numpy(), js, atol=1e-6)
    assert rng.get_call_count() == js_calls


@pytest.mark.parametrize("seed", SEEDS)
def test_multires_rectangular_shape(seed):
    rng.set_seed(seed)
    value.set_seed(seed)
    rng.reset_call_count()
    shape = [64, 128, 3]
    tensor = generators.multires(
        None,
        seed,
        freq=2,
        shape=shape,
        octaves=2,
        post_effects=[],
        final_effects=[],
    )
    assert tensor.shape == (64, 128, 3)
    js, js_calls = js_generator("multires", seed, shape=shape)
    assert js.shape == (64, 128, 3)
    assert np.allclose(tensor.numpy(), js, atol=1e-6)
    assert rng.get_call_count() == js_calls


def test_multires_seed_zero_preserves_rng_state():
    base_seed = SEEDS[0]
    rng.set_seed(base_seed)
    value.set_seed(base_seed)
    rng.reset_call_count()
    tensor = generators.multires(
        None,
        0,
        freq=2,
        shape=[128, 128, 3],
        octaves=1,
        post_effects=[],
        final_effects=[],
    )
    assert tensor.shape == (128, 128, 3)
    js, js_calls = js_generator(
        "multires",
        0,
        octaves=1,
        skipSeedInit=True,
        initialSeed=base_seed,
    )
    assert np.allclose(tensor.numpy(), js, atol=1e-6)
    assert rng.get_call_count() == js_calls
