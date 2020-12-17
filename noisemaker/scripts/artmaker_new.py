import random

import click
import tensorflow as tf

from noisemaker.composer import Preset
from noisemaker.util import load, logger, shape_from_file
from noisemaker.presets import PRESETS

import noisemaker.cli as cli
import noisemaker.effects as effects
import noisemaker.value as value

GENERATORS = {}
EFFECTS = {}

for preset_name in PRESETS:
    preset = Preset(preset_name, PRESETS)

    if preset.is_generator():
        GENERATORS[preset_name] = preset

    if preset.is_effect():
        EFFECTS[preset_name] = preset


@click.group(help="""
        Artmaker - Presets for Noisemaker

        https://github.com/aayars/py-noisemaker
        """, context_settings=cli.CLICK_CONTEXT_SETTINGS)
def main():
    pass


@main.command()
@cli.width_option(default=2048)
@cli.height_option()
@cli.channels_option()
@cli.time_option()
@cli.clut_option()
@cli.seed_option()
@cli.filename_option(default='art.png')
@click.argument('preset_name', type=click.Choice(["random"] + sorted(GENERATORS)))
@click.pass_context
def generator(ctx, width, height, channels, time, clut, seed, filename, preset_name):
    if not seed:
        seed = random.randint(1, 2**32)

    value.set_seed(seed)

    if preset_name == "random":
        preset_name = list(GENERATORS)[random.randint(0, len(GENERATORS) - 1)]

    print(f"{preset_name} (seed: {seed})")

    preset = GENERATORS[preset_name]

    shape = [height, width, channels]

    try:
        preset.render(shape=shape, name=filename)

    except Exception as e:
        logger.error(f"preset.render() failed: {e}\nSeed: {seed}\nArgs: {preset.__dict__}")
        raise


@main.command()
@cli.seed_option()
@cli.filename_option(default='mangled.png')
@cli.option('--no-resize', is_flag=True, help="Don't resize image. May break some presets.")
@cli.time_option()
@click.argument('preset_name', type=click.Choice(['random'] + sorted(EFFECTS)))
@click.argument('input_filename')
@click.pass_context
def effect(ctx, seed, filename, no_resize, time, preset_name, input_filename):
    if not seed:
        seed = random.randint(1, 2**32)

    value.set_seed(seed)

    input_shape = shape_from_file(input_filename)

    input_shape[2] = min(input_shape[2], 3)

    tensor = tf.image.convert_image_dtype(load(input_filename, channels=input_shape[2]), dtype=tf.float32)

    if preset_name == "random":
        preset_name = list(EFFECTS)[random.randint(0, len(EFFECTS) - 1)]

    print(f"{preset_name} (seed: {seed})")

    preset = EFFECTS[preset_name]
    preset.settings['time'] = time

    if no_resize:
        shape = input_shape

    else:
        shape = [1024, 1024, input_shape[2]]

        tensor = effects.square_crop_and_resize(tensor, input_shape, shape[0])

    try:
        preset.render(tensor=tensor, shape=shape, name=filename)

    except Exception as e:
        logger.error(f"preset.render() failed: {e}\nSeed: {seed}\nArgs: {preset.__dict__}")
        raise
