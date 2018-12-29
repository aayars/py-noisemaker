import click
import tensorflow as tf

from PIL import Image

from noisemaker.util import save, load

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
@click.argument('preset_name', type=click.Choice(['random'] + sorted(presets.EFFECTS_PRESETS)))
@click.argument('input_filename')
@click.pass_context
def main(ctx, seed, name, preset_name, input_filename):
    presets.bake_presets(seed)

    tensor = tf.image.convert_image_dtype(load(input_filename), dtype=tf.float32)

    if preset_name == 'random':
        preset_name = 'random-effect'

    kwargs = presets.preset(preset_name)

    print(kwargs['name'])

    if 'freq' not in kwargs:
        kwargs['freq'] = [3, 3]

    if 'octaves' not in kwargs:
        kwargs['octaves'] = 1

    if 'ridges' not in kwargs:
        kwargs['ridges'] = False

    input_shape = effects.shape_from_file(input_filename)

    kwargs['shape'] = [1024, 1024, input_shape[2]]

    tensor = effects.square_crop_and_resize(tensor, input_shape, kwargs['shape'][0])

    tensor = effects.post_process(tensor, **kwargs)

    tensor = recipes.post_process(tensor, **kwargs)

    with tf.Session().as_default():
        save(tensor, name)
