import json

import click
import tensorflow as tf

from noisemaker.util import logger, dumps, load, save

import noisemaker.cli as cli
import noisemaker.effects as effects
import noisemaker.presets as presets
import noisemaker.recipes as recipes


@click.command(help="""
        artmangler - Do something arty with an input image

        https://github.com/aayars/py-noisemaker
        """, context_settings=cli.CLICK_CONTEXT_SETTINGS)
@cli.seed_option()
@cli.name_option(default='mangled.png')
@cli.option('--no-resize', is_flag=True, help="Don't resize image. May break some presets.")
@cli.option('--overrides', type=str, help='A JSON dictionary containing preset overrides')
@cli.time_option()
@click.argument('preset_name', type=click.Choice(['random'] + sorted(presets.EFFECTS_PRESETS)))
@click.argument('input_filename')
@click.pass_context
def main(ctx, seed, name, no_resize, overrides, time, preset_name, input_filename):
    generators.set_seed(seed)
    presets.bake_presets()

    input_shape = effects.shape_from_file(input_filename)

    input_shape[2] = min(input_shape[2], 3)

    tensor = tf.image.convert_image_dtype(load(input_filename, channels=input_shape[2]), dtype=tf.float32)

    if preset_name == 'random':
        preset_name = 'random-effect'

    kwargs = presets.preset(preset_name)

    print(kwargs['name'])

    kwargs['time'] = time

    if 'freq' not in kwargs:
        kwargs['freq'] = [3, 3]

    if 'octaves' not in kwargs:
        kwargs['octaves'] = 1

    if 'ridges' not in kwargs:
        kwargs['ridges'] = False

    if no_resize:
        kwargs['shape'] = input_shape

    else:
        kwargs['shape'] = [1024, 1024, input_shape[2]]

        tensor = effects.square_crop_and_resize(tensor, input_shape, kwargs['shape'][0])

    if overrides:
        kwargs.update(json.loads(overrides))

    try:
        tensor = effects.post_process(tensor, **kwargs)

    except Exception as e:
        logger.error(f"effects.post_process() failed: {e}\nSeed: {seed}\nArgs: {dumps(kwargs)}")
        raise

    try:
        tensor = recipes.post_process(tensor, **kwargs)

    except Exception as e:
        logger.error(f"recipes.post_process() failed: {e}\nSeed: {seed}\nArgs: {dumps(kwargs)}")
        raise

    with tf.compat.v1.Session().as_default():
        save(tensor, name)
