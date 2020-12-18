import os
import random
import shutil
import tempfile

import click
import tensorflow as tf

from noisemaker.composer import Preset
from noisemaker.constants import ValueDistribution
from noisemaker.presets import PRESETS

import noisemaker.cli as cli
import noisemaker.generators as generators
import noisemaker.effects as effects
import noisemaker.util as util
import noisemaker.value as value

GENERATORS = {}
EFFECTS = {}

MAX_SEED_VALUE = 2 ** 32


def _reload_presets():
    """Re-evaluate presets after changing the interpreter's random seed."""

    presets = PRESETS()

    for preset_name in presets:
        preset = Preset(preset_name, presets)

        if preset.is_generator():
            GENERATORS[preset_name] = preset

        if preset.is_effect():
            EFFECTS[preset_name] = preset


_reload_presets()


@click.group(help="""
        Noisemaker - Let's make generative art with noise

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
@click.option('--speed', help="Animation speed", type=float, default=0.25)
@cli.distrib_option()
@cli.filename_option(default='art.png')
@click.argument('preset_name', type=click.Choice(["random"] + sorted(GENERATORS)))
@click.pass_context
def generator(ctx, width, height, channels, time, clut, seed, speed, distrib, filename, preset_name):
    if not seed:
        seed = random.randint(1, MAX_SEED_VALUE)

    value.set_seed(seed)
    _reload_presets()

    if preset_name == "random":
        preset_name = list(GENERATORS)[random.randint(0, len(GENERATORS) - 1)]

    print(f"{preset_name} (seed: {seed})")

    preset = GENERATORS[preset_name]

    if distrib is not None:
        preset.generator_kwargs["distrib"] = distrib

    try:
        preset.render(shape=[height, width, channels], time=time, speed=speed, filename=filename)

    except Exception as e:
        util.logger.error(f"preset.render() failed: {e}\nSeed: {seed}\nArgs: {preset.__dict__}")
        raise


@main.command()
@cli.seed_option()
@cli.filename_option(default='mangled.png')
@cli.option('--no-resize', is_flag=True, help="Don't resize image. May break some presets.")
@cli.time_option()
@click.option('--speed', help="Animation speed", type=float, default=0.25)
@click.argument('preset_name', type=click.Choice(['random'] + sorted(EFFECTS)))
@click.argument('input_filename')
@click.pass_context
def effect(ctx, seed, filename, no_resize, time, speed, preset_name, input_filename):
    if not seed:
        seed = random.randint(1, MAX_SEED_VALUE)

    value.set_seed(seed)
    _reload_presets()

    input_shape = util.shape_from_file(input_filename)

    input_shape[2] = min(input_shape[2], 3)

    tensor = tf.image.convert_image_dtype(util.load(input_filename, channels=input_shape[2]), dtype=tf.float32)

    if preset_name == "random":
        preset_name = list(EFFECTS)[random.randint(0, len(EFFECTS) - 1)]

    print(f"{preset_name} (seed: {seed})")

    preset = EFFECTS[preset_name]

    if no_resize:
        shape = input_shape

    else:
        shape = [1024, 1024, input_shape[2]]

        tensor = effects.square_crop_and_resize(tensor, input_shape, shape[0])

    try:
        preset.render(tensor=tensor, shape=shape, time=time, speed=speed, filename=filename)

    except Exception as e:
        util.logger.error(f"preset.render() failed: {e}\nSeed: {seed}\nArgs: {preset.__dict__}")
        raise


@main.command(help="Create a .gif or .mp4. Requires ImageMagick and ffmpeg, respectively.")
@cli.width_option()
@cli.height_option()
@cli.channels_option()
@cli.seed_option()
@cli.option('--effect-preset', type=click.Choice(["random"] + sorted(EFFECTS)))
@cli.filename_option(default='ani.gif')
@cli.option('--save-frames', default=None, type=click.Path(exists=True, dir_okay=True))
@cli.option('--frame-count', type=int, default=30, help="How many frames total")
@cli.option('--watermark', type=str)
@click.argument('preset_name', type=click.Choice(['random'] + sorted(GENERATORS)))
@click.pass_context
def animation(ctx, width, height, channels, seed, effect_preset, filename, save_frames, frame_count, watermark, preset_name):
    if seed is None:
        seed = random.randint(1, MAX_SEED_VALUE)

    value.set_seed(seed)
    _reload_presets()

    if preset_name == 'random':
        preset_name = list(GENERATORS)[random.randint(0, len(GENERATORS) - 1)]

    if effect_preset == 'random':
        effect_preset = list(EFFECTS)[random.randint(0, len(EFFECTS) - 1)]

    if effect_preset:
        print(f"{preset_name} vs. {effect_preset} (seed: {seed})")
    else:
        print(f"{preset_name} (seed: {seed})")

    preset = GENERATORS[preset_name]

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(frame_count):
            frame_filename = f'{tmp}/{i:04d}.png'

            common_params = ['--seed', str(seed),
                             '--time', f'{i/frame_count:0.4f}',
                             '--filename', frame_filename]

            util.check_call(['noisemaker', 'generator', preset_name,
                             '--distrib', _use_periodic_distrib(preset.generator_kwargs.get("distrib")),
                             '--speed', str(_use_reasonable_speed(preset, frame_count)),
                             '--height', str(height),
                             '--width', str(width)] + common_params)

            if effect_preset:
                util.check_call(['noisemaker', 'effect', effect_preset, frame_filename,
                                 '--no-resize',
                                 '--speed', str(_use_reasonable_speed(EFFECTS[effect_preset], frame_count))] + common_params)

            if save_frames:
                shutil.copy(frame_filename, save_frames)

            if watermark:
                util.watermark(watermark, frame_filename)

        if filename.endswith(".mp4"):
            # when you want something done right
            util.check_call(['ffmpeg',
                             '-y',  # overwrite existing
                             '-framerate', '30',
                             '-i', f'{tmp}/%04d.png',
                             '-c:v', 'libx264',  # because this is what twitter wants
                             '-pix_fmt', 'yuv420p',  # because this is what twitter wants
                             '-b:v', '1700000',  # maximum allowed bitrate for 720x720 (2048k), minus some encoder overhead
                             '-s', '720x720',  # a twitter-recommended size
                             filename])

        else:
            util.magick(f'{tmp}/*png', filename)


@main.command()
@cli.input_dir_option(required=True)
@cli.filename_option(default="collage.png")
@click.option("--control-filename", help="Control image filename (optional)")
@click.pass_context
def mashup(ctx, input_dir, filename, control_filename):
    filenames = []

    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith(('.png', '.jpg')):
                filenames.append(os.path.join(root, f))

    collage_count = min(random.randint(4, 6), len(filenames))
    collage_images = []

    for i in range(collage_count + 1):
        index = random.randint(0, len(filenames) - 1)

        input_filename = os.path.join(input_dir, filenames[index])

        collage_input = tf.image.convert_image_dtype(util.load(input_filename, channels=3), dtype=tf.float32)

        collage_images.append(collage_input)

    if control_filename:
        control = tf.image.convert_image_dtype(load(control_filename, channels=1), dtype=tf.float32)

    else:
        control = collage_images.pop()

    shape = tf.shape(control)  # All images need to be the same size!

    control = effects.value_map(control, shape, keepdims=True)

    base = generators.basic(freq=random.randint(2, 5), shape=shape, lattice_drift=random.randint(0, 1), hue_range=random.random())

    value_shape = [shape[0], shape[1], 1]

    control = effects.convolve(kernel=effects.ValueMask.conv2d_blur, tensor=control, shape=value_shape)

    with tf.compat.v1.Session().as_default():
        tensor = effects.blend_layers(control, shape, random.random() * .5, *collage_images)

        tensor = value.blend(tensor, base, .125 + random.random() * .125)

        tensor = effects.bloom(tensor, shape, alpha=.25 + random.random() * .125)
        tensor = effects.shadow(tensor, shape, alpha=.25 + random.random() * .125, reference=control)

        tensor = tf.image.adjust_brightness(tensor, .05)
        tensor = tf.image.adjust_contrast(tensor, 1.25)

        util.save(tensor, filename)

    print('mashup')


def _use_periodic_distrib(distrib):
    """Make sure the given ValueDistribution is animation-friendly."""

    if distrib and isinstance(distrib, int):
        distrib = ValueDistribution(distrib)

    elif distrib and isinstance(distrib, str):
        distrib = ValueDistribution[distrib]

    if distrib == ValueDistribution.exp:
        distrib = ValueDistribution.periodic_exp

    elif distrib == ValueDistribution.lognormal:
        distrib = ValueDistribution.simplex_pow_inv_1

    elif distrib not in (
        ValueDistribution.ones,
        ValueDistribution.column_index,
        ValueDistribution.row_index,
    ) and not ValueDistribution.is_simplex(distrib) and not ValueDistribution.is_fastnoise(distrib) and not ValueDistribution.is_periodic(distrib):
        distrib = ValueDistribution.periodic_uniform

    return distrib.name


def _use_reasonable_speed(preset, frame_count):
    """Return a reasonable speed parameter for the given animation length."""

    return preset.settings.get("speed", 0.25) * (frame_count / 30.0)
