import os
import random

import click
import tensorflow as tf

from noisemaker.util import load, save, shape_from_file

import noisemaker.cli as cli
import noisemaker.effects as effects
import noisemaker.generators as generators
import noisemaker.value as value


@click.group(help="""
        Collagemaker - Image collage tool

        https://github.com/aayars/py-noisemaker
        """, context_settings=cli.CLICK_CONTEXT_SETTINGS)
def main():
    pass


@main.command()
@cli.width_option()
@cli.height_option()
@cli.input_dir_option(required=True)
@cli.name_option(default="collage.png")
@click.option("--control-filename", help="Control image filename (optional)")
@click.option("--retro-upscale", is_flag=True, help="Nearest-neighbor upsample (for small images)")
@click.pass_context
def basic(ctx, width, height, input_dir, name, control_filename, retro_upscale):
    shape = [height, width, 3]  # Any shape you want, as long as it's [1024, 1024, 3]

    filenames = []

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(('.png', '.jpg')):
                filenames.append(os.path.join(root, filename))

    collage_count = min(random.randint(4, 6), len(filenames))
    collage_images = []

    for i in range(collage_count + 1):
        index = random.randint(0, len(filenames) - 1)

        input_filename = os.path.join(input_dir, filenames[index])

        collage_input = tf.image.convert_image_dtype(load(input_filename, channels=3), dtype=tf.float32)

        input_shape = shape_from_file(input_filename)

        if retro_upscale:
            input_shape = [input_shape[0] * 2, input_shape[1] * 2, input_shape[2]]

            collage_input = value.resample(collage_input, input_shape, spline_order=0)

        collage_input = effects.square_crop_and_resize(collage_input, input_shape, 1024)

        collage_images.append(collage_input)

    base = generators.basic(freq=random.randint(2, 5), shape=shape, lattice_drift=random.randint(0, 1), hue_range=random.random())

    if control_filename:
        control = tf.image.convert_image_dtype(load(control_filename, channels=1), dtype=tf.float32)

        control = effects.square_crop_and_resize(control, shape_from_file(control_filename), 1024)

        control = effects.value_map(control, shape, keepdims=True)

    else:
        control = effects.value_map(collage_images.pop(), shape, keepdims=True)

    control = effects.convolve(effects.ValueMask.conv2d_blur, control, [height, width, 1])

    with tf.compat.v1.Session().as_default():
        # sort collage images by brightness
        collage_images = [j[1] for j in sorted([(tf.reduce_sum(i).numpy(), i) for i in collage_images])]

        tensor = effects.blend_layers(control, shape, random.random() * .5, *collage_images)

        tensor = value.blend(tensor, base, .125 + random.random() * .125)

        tensor = effects.bloom(tensor, shape, alpha=.25 + random.random() * .125)
        tensor = effects.shadow(tensor, shape, alpha=.25 + random.random() * .125, reference=control)

        tensor = tf.image.adjust_brightness(tensor, .05)
        tensor = tf.image.adjust_contrast(tensor, 1.25)

        save(tensor, name)

    print('mashup')
