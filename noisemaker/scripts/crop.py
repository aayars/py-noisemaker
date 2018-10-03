import click
import tensorflow as tf

from noisemaker.util import save, load

import noisemaker.cli as cli
import noisemaker.effects as effects


@click.command(help="""
        crop - Quick hack to crop/resize an image into a 1024x1024 JPEG.

        https://github.com/aayars/py-noisemaker
        """, context_settings=cli.CLICK_CONTEXT_SETTINGS)
@cli.name_option(default="cropped.jpg")
@click.argument('input_filename')
@click.pass_context
def main(ctx, name, input_filename):
    tensor = tf.image.convert_image_dtype(load(input_filename), tf.float32)

    max_height = 1024
    max_width = 1024

    with tf.Session().as_default():
        height, width, channels = tf.shape(tensor).eval()

        need_resample = False

        if height != width:
            length = min(height, width)
            height = length
            width = length

            tensor = tf.image.resize_image_with_crop_or_pad(tensor, length, length)

        if height > max_height:
            need_resample = True
            width = int(width * (max_height / height))
            height = max_height

        if width > max_width:
            need_resample = True
            height = int(height * (max_width / width))
            width = max_width

        shape = [height, width, channels]

        if need_resample:
            tensor = effects.resample(tensor, shape)

        save(tensor, name)