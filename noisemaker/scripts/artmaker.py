from enum import Enum

import json

import click
import tensorflow as tf

from noisemaker.util import save

import noisemaker.cli as cli
import noisemaker.generators as generators
import noisemaker.presets as presets
import noisemaker.recipes as recipes


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
    presets.bake_presets(seed)

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
        for key, value in sorted(kwargs.items()):
            if key in {'name', 'shape', 'time'}:
                continue

            if key in {'ridges', 'with_convolve'} and not value:
                continue

            if key == 'octaves' and value == 1:
                continue

            if isinstance(value, float):
                value = f'{value:.02f}'

            if isinstance(value, Enum):
                value = value.name

            print(f'{key}: {value}')

        import sys
        sys.exit()

    else:
        print(kwargs['name'])

    tensor = generators.multires(**kwargs)

    tensor = recipes.post_process(tensor, **kwargs)

    with tf.compat.v1.Session().as_default():
        save(tensor, name)
