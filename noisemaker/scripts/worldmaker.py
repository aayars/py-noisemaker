import random

import click
import tensorflow as tf

from noisemaker.util import save, load

import noisemaker.effects as effects
import noisemaker.generators as generators
import noisemaker.recipes as recipes

SMALL_X = 512
SMALL_Y = 512

LARGE_X = 1024
LARGE_Y = 1024

FREQ = 4
SATURATION = .333
OCTAVES = 8

CONTROL_FILENAME = "worldmaker-control.png"
LOW_FILENAME = "worldmaker-lowland.png"
MID_FILENAME = "worldmaker-midland.png"
HIGH_FILENAME = "worldmaker-highland.png"

BLENDED_FILENAME = "worldmaker-blended.png"

FINAL_FILENAME = "worldmaker.png"


@click.group()
def main():
    pass


@main.command()
def lowland():
    shape = [LARGE_Y, LARGE_X, 3]

    kwargs = {
        "deriv": 1,
        "deriv_alpha": .5,
        "freq": FREQ,
        "hue_range": .125 + random.random() * .25,
        "hue_rotation": .875 + random.random() * .125,
        "lattice_drift": 1,
        "octaves": OCTAVES,
        "point_freq": 5,
        "saturation": SATURATION * 2,
        "voronoi_alpha": .333,
        "voronoi_inverse": True,
        "voronoi_nth": 0,
        "with_voronoi": 2,
    }

    tensor = generators.multires(shape=shape, **kwargs)

    tensor = tf.image.adjust_brightness(tensor, .1)

    with tf.Session().as_default():
        save(tensor, LOW_FILENAME)


@main.command()
def midland():
    shape = [LARGE_Y, LARGE_X, 3]

    kwargs = {
        "deriv": 1,
        "deriv_alpha": .25,
        "freq": FREQ * 2,
        "hue_range": .25 + random.random() * .125,
        "hue_rotation": .875 + random.random() * .1,
        "octaves": OCTAVES,
        "point_freq": 5,
        "saturation": SATURATION * 2,
        "voronoi_refract": .5,
        "voronoi_alpha": .5,
        "voronoi_nth": 1,
        "with_voronoi": 6,
    }

    tensor = generators.multires(shape=shape, **kwargs)

    with tf.Session().as_default():
        save(tensor, MID_FILENAME)


@main.command()
def highland():
    shape = [LARGE_Y, LARGE_X, 3]

    kwargs = {
        "deriv": 1,
        "deriv_alpha": 0.25 + random.random() * .125,
        "freq": FREQ * 4,
        "hue_range": .125 + random.random() * .125,
        "hue_rotation": .925 + random.random() * .05,
        "octaves": OCTAVES,
        "point_freq": 8,
        "ridges": True,
        "saturation": SATURATION,
        "voronoi_alpha": .5,
        "voronoi_nth": 3,
        "with_voronoi": 2,
    }

    tensor = generators.multires(shape=shape, **kwargs)

    tensor = tf.image.adjust_brightness(tensor, .1)

    with tf.Session().as_default():
        save(tensor, HIGH_FILENAME)


@main.command("control")
def _control():
    shape = [SMALL_Y, SMALL_X, 1]

    control = generators.multires(shape=shape, freq=3, octaves=OCTAVES, post_refract_range=1.0)

    erode_kwargs = {
        "alpha": .02,
        "density": 50,
        "iterations": 25,
        "inverse": True,
    }

    iterations = 5
    for i in range(iterations):
       control = effects.erode(control, shape, **erode_kwargs)
       control = effects.convolve(effects.ConvKernel.blur, control, shape)

    post_shape = [LARGE_Y, LARGE_X, 1]
    control = effects.resample(control, post_shape)

    iterations = 2
    for i in range(iterations):
       control = effects.erode(control, post_shape, **erode_kwargs)
       control = effects.convolve(effects.ConvKernel.blur, control, post_shape)

    control = effects.convolve(effects.ConvKernel.sharpen, control, post_shape)
    control = effects.normalize(control)

    with tf.Session().as_default():
        save(control, CONTROL_FILENAME)


@main.command()
def blended():
    shape = [LARGE_Y, LARGE_X, 3]

    erode_kwargs = {
        "alpha": .025,
        "density": 250,
        "iterations": 50,
        "inverse": True,
    }

    control = tf.image.convert_image_dtype(load(CONTROL_FILENAME), tf.float32)

    water = tf.ones(shape) * tf.stack([.05, .2, .333])
    water = effects.blend(water, control * 4.0, .125)

    low = tf.image.convert_image_dtype(load(LOW_FILENAME), tf.float32)
    mid = tf.image.convert_image_dtype(load(MID_FILENAME), tf.float32)
    high = tf.image.convert_image_dtype(load(HIGH_FILENAME), tf.float32)

    blend_control = generators.multires(shape=shape, freq=FREQ * 2, ridges=True, octaves=2, post_refract_range=2.5)
    blend_control = 1.0 - effects.value_map(blend_control, shape, keep_dims=True) * .5

    combined_land = effects.blend_layers(control, shape, blend_control, control * 2, low, mid, high)
    combined_land = effects.erode(combined_land, shape, xy_blend=.25, **erode_kwargs)
    combined_land = effects.erode(combined_land, shape, **erode_kwargs)

    combined_land = effects.shadow(combined_land, shape, alpha=1.0, reference=control)

    combined = effects.blend_layers(control, shape, .01, water, combined_land, combined_land, combined_land)
    combined = effects.blend(combined_land, combined, .625)

    combined = tf.image.adjust_brightness(combined, .05)
    combined = tf.image.adjust_contrast(combined, .875)
    combined = tf.image.adjust_saturation(combined, .625)

    with tf.Session().as_default():
        save(combined, BLENDED_FILENAME)


@main.command()
@click.argument('input_filename', default=BLENDED_FILENAME)
def clouds(input_filename):
    tensor = tf.image.convert_image_dtype(load(input_filename), tf.float32)

    pre_shape = [SMALL_Y, SMALL_X, 1]
    post_shape = [LARGE_Y, LARGE_X, 1]

    control_kwargs = {
        "freq": FREQ * 2,
        "lattice_drift": 1,
        "octaves": OCTAVES,
        "ridges": True,
        "shape": pre_shape,
        "warp_freq": 3,
        "warp_range": .25,
        "warp_octaves": 2,
    }

    control = generators.multires(**control_kwargs)

    layer_0 = tf.ones(pre_shape)
    layer_1 = tf.zeros(pre_shape)

    combined = effects.blend_layers(control, pre_shape, 1.0, layer_0, layer_1)

    shadow = effects.offset(combined, pre_shape, random.randint(-15, 15), random.randint(-15, 15))
    shadow = tf.minimum(shadow * 2.5, 1.0)
    shadow = effects.convolve(effects.ConvKernel.blur, shadow, pre_shape)
    shadow = effects.convolve(effects.ConvKernel.blur, shadow, pre_shape)
    shadow = effects.convolve(effects.ConvKernel.blur, shadow, pre_shape)

    shadow = effects.resample(shadow, post_shape)
    combined = effects.resample(combined, post_shape)

    tensor = effects.blend(tensor, tf.zeros(post_shape), shadow * .5)
    tensor = effects.blend(tensor, tf.ones(post_shape), combined)

    post_shape = [LARGE_Y, LARGE_X, 3]

    tensor = effects.shadow(tensor, post_shape, alpha=.25)

    tensor = effects.bloom(tensor, post_shape, .333)
    tensor = recipes.dither(tensor, post_shape, .075)

    with tf.Session().as_default():
        save(tensor, FINAL_FILENAME)
