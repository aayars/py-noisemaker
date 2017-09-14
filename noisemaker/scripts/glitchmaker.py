import random

import click
import tensorflow as tf

from noisemaker.util import save, load

import noisemaker.cli as cli
import noisemaker.effects as effects
import noisemaker.recipes as recipes


@click.group(help="""
        Glitchmaker - Glitch art tool

        https://github.com/aayars/py-noisemaker
        """, context_settings=cli.CLICK_CONTEXT_SETTINGS)
def main():
    pass


@main.command()
@cli.name_option(default="glitch.png")
@click.argument('input_filename')
@click.pass_context
def auto(ctx, name, input_filename):
    glitch = (random.random() > .333)
    vhs = (random.random() > .666)
    crt = (random.random() > .333)
    scan_error = (random.random() > .5)
    snow = random.random() * .5 if (random.random() > .5) else 0
    dither = random.random() * .25 if (random.random() > .5) else 0
    aberration = random.random() * .02 if (random.random() > .333) else 0
    bloom = .5 + random.random() * .5 if (random.random() > .5) else 0

    render(ctx, glitch, vhs, crt, scan_error, snow, dither, aberration, bloom, name, input_filename)


@main.command()
@cli.glitch_option(default=True)
@cli.vhs_option()
@cli.crt_option(default=True)
@cli.scan_error_option(default=True)
@cli.snow_option()
@cli.dither_option()
@cli.aberration_option(default=.01)
@cli.bloom_option()
@cli.name_option(default="glitch.png")
@click.argument('input_filename')
@click.pass_context
def advanced(*args):
    render(*args)


def render(ctx, glitch, vhs, crt, scan_error, snow, dither, aberration, bloom, name, input_filename):
    tensor = tf.image.convert_image_dtype(load(input_filename), tf.float32)

    freq = [3, 3]

    max_height = 1024
    max_width = 1024

    with tf.Session().as_default():
        height, width, channels = tf.shape(tensor).eval()

        need_resample = False

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

        tensor = effects.post_process(tensor, shape=shape, freq=freq, with_bloom=bloom, with_aberration=aberration)

        tensor = recipes.post_process(tensor, shape=shape, freq=freq, with_glitch=glitch, with_vhs=vhs, with_crt=crt,
                                      with_scan_error=scan_error, with_snow=snow, with_dither=dither)

        save(tensor, name)

    print(name)