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
@cli.name_option(default="art.png")
@click.argument('preset_name', type=click.Choice(["random"] + sorted(presets.PRESETS)))
@click.pass_context
def main(ctx, width, height, channels, name, preset_name):
    kwargs, post_kwargs, preset_name = presets.load(preset_name)

    kwargs["shape"] = [height, width, channels]

    # print(kwargs)

    if "freq" not in kwargs:
        kwargs["freq"] = 3

    if "octaves" not in kwargs:
        kwargs["octaves"] = 1

    if "ridges" not in kwargs:
        kwargs["ridges"] = False

    post_kwargs["shape"] = kwargs["shape"]
    post_kwargs["freq"] = kwargs["freq"]

    tensor = generators.multires(**kwargs)

    tensor = recipes.post_process(tensor, **post_kwargs)

    with tf.Session().as_default():
        save(tensor, name)

    print(preset_name)
