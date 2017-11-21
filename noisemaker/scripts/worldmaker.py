import random

import click
import tensorflow as tf

from noisemaker.util import save, load

import noisemaker.effects as effects
import noisemaker.generators as generators
import noisemaker.recipes as recipes


@click.group()
def main():
    pass


@main.command()
def land():
    pre_shape = [512, 512, 3]
    post_shape = [1024, 1024, 3]

    freq = 3
    saturation = .333
    octaves = 8

    water_kwargs = {
        "distrib": "uniform",
        "freq": 256,
        "hue_range": .05 + random.random() * .05,
        "hue_rotation": .5125 + random.random() * .025,
        "reflect_range": .333 + random.random() * .333,
    }

    mid_kwargs = {
        "deriv": 1,
        "deriv_alpha": .25,
        "freq": freq,
        "hue_range": .125 + random.random() * .125,
        "hue_rotation": .925 + random.random() * .05,
        "lattice_drift": 1,
        "octaves": octaves,
        "point_freq": 10,
        "saturation": saturation,
        "voronoi_alpha": .333,
        "voronoi_nth": 0,
        "with_voronoi": 2,
    }

    mid2_kwargs = {
        "deriv": 1,
        "deriv_alpha": .25,
        "freq": freq * 2,
        "hue_range": .125 + random.random() * .125,
        "hue_rotation": .925 + random.random() * .05,
        "octaves": octaves,
        "point_freq": 10,
        "saturation": saturation,
        "voronoi_refract": .5,
        "voronoi_alpha": .333,
        "voronoi_nth": 1,
        "with_voronoi": 6,
    }

    high_kwargs = {
        "deriv": 1,
        "deriv_alpha": 0.25 + random.random() * .125,
        "freq": freq * 3,
        "hue_range": .125 + random.random() * .125,
        "hue_rotation": .925 + random.random() * .05,
        "octaves": octaves,
        "point_freq": 10,
        "ridges": True,
        "saturation": saturation,
        "voronoi_alpha": .333,
        "voronoi_nth": 1,
        "with_voronoi": 2,
    }

    erode_kwargs = {
        "alpha": .05,
        "density": 250,
        "iterations": 250,
        "inverse": True,
        "xy_blend": .25,
    }

    control = generators.multires(shape=pre_shape, freq=4, octaves=octaves)

    water = generators.multires(shape=pre_shape, **water_kwargs)
    water = effects.blend(water, tf.ones(pre_shape) * tf.stack([.05, .2, .333]), .875)
    water = effects.blend(water, control * 4.0, .125)

    mid = generators.multires(shape=pre_shape, **mid_kwargs)
    mid2 = generators.multires(shape=pre_shape, **mid2_kwargs)
    high = generators.multires(shape=pre_shape, **high_kwargs)

    values = effects.value_map(control, pre_shape, keep_dims=True)

    blend_control = generators.multires(shape=pre_shape, freq=freq, ridges=True, octaves=1, post_refract_range=2.5)
    blend_control = 1.0 - effects.value_map(blend_control, pre_shape, keep_dims=True) * .5

    combined_land = effects.blend_layers(values, pre_shape, blend_control, control * 2, mid, mid2, high)
    combined_land = effects.erode(combined_land, pre_shape, **erode_kwargs)
    combined_land = effects.shadow(combined_land, pre_shape, 1.0)

    combined = effects.blend_layers(effects.resample(values, pre_shape), pre_shape, .01, water, combined_land, combined_land, combined_land)
    combined = effects.blend(combined_land, combined, .625)

    combined = effects.resample(combined, post_shape)

    combined = effects.bloom(combined, post_shape, .333)
    combined = recipes.dither(combined, post_shape, .1)

    combined = tf.image.adjust_saturation(combined, .75)

    with tf.Session().as_default():
        save(combined, 'world.png')


@main.command()
@click.argument('input_filename', default="world.png")
def clouds(input_filename):
    tensor = tf.image.convert_image_dtype(load(input_filename), tf.float32)

    pre_shape = [512, 512, 1]
    post_shape = [1024, 1024, 1]

    control_kwargs = {
        "freq": 8,
        "lattice_drift": 1,
        "octaves": 8,
        "ridges": True,
        "shape": pre_shape,
        "warp_freq": 3,
        "warp_range": .25,
        "warp_octaves": 2,
    }

    control = generators.multires(**control_kwargs) * 2.0

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

    post_shape = [1024, 1024, 3]

    tensor = effects.bloom(tensor, post_shape, .333)
    tensor = recipes.dither(tensor, post_shape, .1)

    with tf.Session().as_default():
        save(tensor, 'clouds.png')
