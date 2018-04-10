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

    tensor = tf.image.convert_image_dtype(load(input_filename), tf.float32)

    max_height = 1024
    max_width = 1024

    with tf.Session().as_default():
        height, width, channels = tf.shape(tensor).eval()

        need_resample = False

        # Some presets only like square images. Work around for now by cropping.
        if height != width:
            length = min(height, width)
            height = length
            width = length

            tensor = tf.image.resize_image_with_crop_or_pad(tensor, length, length)

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

        kwargs, post_kwargs, preset_name = presets.load(preset_name, presets.EFFECTS_PRESETS())

        kwargs["shape"] = [height, width, channels]

        if "freq" not in kwargs:
            kwargs["freq"] = [3, 3]

        if "octaves" not in kwargs:
            kwargs["octaves"] = 1

        if "ridges" not in kwargs:
            kwargs["ridges"] = False

        post_kwargs["shape"] = kwargs["shape"]
        post_kwargs["freq"] = kwargs["freq"]

        tensor = effects.post_process(tensor, **kwargs)
        tensor = recipes.post_process(tensor, **post_kwargs)

        print(preset_name)
        save(tensor, name)
