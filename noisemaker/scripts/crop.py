import click
import tensorflow as tf

from noisemaker.util import load, save

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
    tensor = effects.square_crop_and_resize(tf.image.convert_image_dtype(load(input_filename), tf.float32), effects.shape_from_file(input_filename), 1024)

    with tf.Session().as_default():
        save(tensor, name)
