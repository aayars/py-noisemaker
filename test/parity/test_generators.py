"""Pixel-to-pixel parity tests against the JavaScript implementation.

These tests invoke the JavaScript reference implementation in a subprocess for
each run and compare the raw tensor outputs directly with the Python version.
This ensures we're always testing parity with the current JS code base without
relying on canned fixtures.
"""

import numpy as np
import pytest

from noisemaker import generators, rng, value
from noisemaker.constants import ColorSpace, ValueDistribution

from .utils import js_generator

# 20 randomly chosen 32-bit seeds
SEEDS = [
    3626764237, 1654615998, 3255389356, 3823568514, 1806341205,
    173879092, 1112038970, 4146640122, 2195908194, 2087043557,
    1739178872, 3943786419, 3366389305, 3564191072, 1302718217,
    4156669319, 2046968324, 1537810351, 2505606783, 3829653368,
]

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
