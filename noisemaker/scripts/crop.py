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
@click.option('--retro-upscale', is_flag=True, help="Nearest-neighbor upsample (for small images)")
@click.argument('input_filename')
@click.pass_context
def main(ctx, name, retro_upscale, input_filename):
    shape = effects.shape_from_file(input_filename)

    tensor = tf.image.convert_image_dtype(load(input_filename, channels=3), tf.float32)

    if retro_upscale:
        shape = [shape[0] * 2, shape[1] * 2, shape[2]]

        tensor = effects.resample(tensor, shape, spline_order=0)

    tensor = effects.square_crop_and_resize(tensor, shape, 1024)

    with tf.compat.v1.Session().as_default():
        save(tensor, name)
