import click
import tensorflow as tf

from noisemaker.util import save, load

import noisemaker.cli as cli
import noisemaker.effects as effects
import noisemaker.generators as generators
import noisemaker.presets as presets
import noisemaker.recipes as recipes


@click.command(help="""
        artmangler - Do something arty with an input image

        https://github.com/aayars/py-noisemaker
        """, context_settings=cli.CLICK_CONTEXT_SETTINGS)
@cli.seed_option()
@cli.name_option(default="mangled.png")
@click.argument('preset_name', type=click.Choice(["random"] + sorted(presets.EFFECTS_PRESETS())))
@click.argument('input_filename')
@click.pass_context
def main(ctx, seed, name, preset_name, input_filename):
    generators.set_seed(seed)

    tensor = tf.image.convert_image_dtype(load(input_filename), dtype=tf.float32)

    tensor = effects.square_crop_and_resize(tensor, 1024)

    with tf.Session().as_default():
        height, width, channels = tf.shape(tensor).eval()

        kwargs, preset_name = presets.load(preset_name, presets.EFFECTS_PRESETS())

        kwargs["shape"] = [1024, 1024, channels]

        if "freq" not in kwargs:
            kwargs["freq"] = [3, 3]

        if "octaves" not in kwargs:
            kwargs["octaves"] = 1

        if "ridges" not in kwargs:
            kwargs["ridges"] = False

        tensor = effects.post_process(tensor, **kwargs)
        tensor = recipes.post_process(tensor, **kwargs)

        print(preset_name)
        save(tensor, name)
