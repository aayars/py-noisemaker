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
@cli.clut_option()
@cli.seed_option()
@cli.name_option(default='art.png')
@click.argument('preset_name', type=click.Choice(['random'] + sorted(presets.PRESETS)))
@click.pass_context
def main(ctx, width, height, channels, clut, seed, name, preset_name):
    presets.bake_presets(seed)

    if preset_name == 'random':
        preset_name = 'random-preset'

    kwargs = presets.preset(preset_name)

    extends = kwargs.get('extends')

    if extends:
        print('{} ({})'.format(kwargs['name'], ', '.join(sorted(extends))))

    else:
        print(kwargs['name'])

    kwargs['shape'] = [height, width, channels]

    if 'freq' not in kwargs:
        kwargs['freq'] = 3

    if 'octaves' not in kwargs:
        kwargs['octaves'] = 1

    if 'ridges' not in kwargs:
        kwargs['ridges'] = False

    if clut:
        kwargs['clut'] = clut
        kwargs['clut_horizontal'] = True

    tensor = generators.multires(**kwargs)

    tensor = recipes.post_process(tensor, **kwargs)

    with tf.Session().as_default():
        save(tensor, name)
