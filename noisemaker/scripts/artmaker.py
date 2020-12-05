from enum import Enum

import json

import click
import tensorflow as tf

from noisemaker.util import logger, dumps, save

import noisemaker.cli as cli
import noisemaker.generators as generators
import noisemaker.presets as presets
import noisemaker.recipes as recipes
import noisemaker.value as value


@click.command(help="""
        Artmaker - Presets for Noisemaker

        https://github.com/aayars/py-noisemaker
        """, context_settings=cli.CLICK_CONTEXT_SETTINGS)
@cli.width_option()
@cli.height_option()
@cli.channels_option()
@cli.time_option()
@cli.clut_option()
@cli.seed_option()
@cli.option('--overrides', type=str, help='A JSON dictionary containing preset overrides')
@cli.option('--settings', is_flag=True, help='Just print the preset settings and exit')
@cli.name_option(default='art.png')
@click.argument('preset_name', type=click.Choice(['random'] + sorted(presets.PRESETS)))
@click.pass_context
def main(ctx, width, height, channels, time, clut, seed, overrides, settings, name, preset_name):
    value.set_seed(seed)
    presets.bake_presets()

    if preset_name == 'random':
        preset_name = 'random-preset'

    kwargs = presets.preset(preset_name)

    kwargs['shape'] = [height, width, channels]
    kwargs['time'] = time

    if 'freq' not in kwargs:
        kwargs['freq'] = 3

    if 'octaves' not in kwargs:
        kwargs['octaves'] = 1

    if 'ridges' not in kwargs:
        kwargs['ridges'] = False

    if clut:
        kwargs['clut'] = clut
        kwargs['clut_horizontal'] = True

    if overrides:
        kwargs.update(json.loads(overrides))

    if settings:
        for k, v in sorted(kwargs.items()):
            if k in {'name', 'shape', 'time'}:
                continue

            if k in {'ridges', 'with_convolve'} and not v:
                continue

            if k == 'octaves' and v == 1:
                continue

            if isinstance(v, float):
                v = f'{v:.02f}'

            if isinstance(v, Enum):
                v = v.name

            print(f'{k}: {v}')

        import sys
        sys.exit()

    else:
        print(kwargs['name'])

    try:
        tensor = generators.multires(**kwargs)

    except Exception as e:
        logger.error(f"generators.multires() failed: {e}\nSeed: {seed}\nArgs: {dumps(kwargs)}")
        raise

    try:
        tensor = recipes.post_process(tensor, **kwargs)

    except Exception as e:
        logger.error(f"recipes.post_process() failed: {e}\nSeed: {seed}\nArgs: {dumps(kwargs)}")
        raise

    with tf.compat.v1.Session().as_default():
        save(tensor, name)
