from enum import Enum

import os
import random
import shutil
import subprocess
import tempfile

import click
import tensorflow as tf

from noisemaker.composer import EFFECT_PRESETS, GENERATOR_PRESETS, reload_presets
from noisemaker.constants import ColorSpace, ValueDistribution
from noisemaker.presets import PRESETS

import noisemaker.ai as ai
import noisemaker.cli as cli
import noisemaker.generators as generators
import noisemaker.effects as effects
import noisemaker.util as util
import noisemaker.value as value

MAX_SEED_VALUE = 2 ** 32 - 1


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


reload_presets(PRESETS)


@click.group(help="""
        Noisemaker - Let's make generative art with noise

        https://github.com/aayars/py-noisemaker
        """, context_settings=cli.CLICK_CONTEXT_SETTINGS)
def main():
    pass


@main.command(help="Generate a .png or .jpg from preset")
@cli.width_option()
@cli.height_option()
@cli.channels_option()
@cli.time_option()
@click.option('--speed', help="Animation speed", type=float, default=0.25)
@cli.seed_option()
@cli.filename_option(default='art.png')
@click.option('--with-ai', help="Apply image-to-image AI (requires stability.ai key)", is_flag=True, default=False)
@click.option('--with-alt-text', help="Generate alt text (requires OpenAI key)", is_flag=True, default=False)
@click.argument('preset_name', type=click.Choice(["random"] + sorted(GENERATOR_PRESETS)))
@click.pass_context
def generator(ctx, width, height, channels, time, speed, seed, filename, with_ai, with_alt_text, preset_name):
    if not seed:
        seed = random.randint(1, MAX_SEED_VALUE)

    value.set_seed(seed)
    reload_presets(PRESETS)

    if preset_name == "random":
        preset_name = list(GENERATOR_PRESETS)[random.randint(0, len(GENERATOR_PRESETS) - 1)]

    print(f"{preset_name.replace('-', ' ')} (seed: {seed})")

    preset = GENERATOR_PRESETS[preset_name]

    # print_preset(preset, with_ai)

    try:
        preset.render(seed, shape=[height, width, channels], time=time, speed=speed, filename=filename, with_ai=with_ai)

    except Exception as e:
        util.logger.error(f"preset.render() failed: {e}\nSeed: {seed}\nArgs: {preset.__dict__}")
        raise

    if with_alt_text:
        print(ai.describe(preset.name.replace('-', ' '), preset.ai_settings.get("prompt"), filename))


def print_preset(preset, with_ai):
    if with_ai:
        print("")
        print(preset.ai_settings["prompt"])

    if preset.layers:
        print("")
        print("Extends: " + ", ".join(str(l).replace('-', ' ') for l in preset.flattened_layers))

    print("")
    print("Layers:")

    if preset.final_effects:
        print("  - Final:")

        for effect in reversed(preset.final_effects):
            if callable(effect):
                print(f"    - {effect.func.__name__.replace('_', ' ')}")
            else:
                print(f"    - {effect.name.replace('_', ' ').replace('-', ' ')}")

    if preset.post_effects or with_ai:
        print("  - Post:")

        if with_ai:
            print("    - stable diffusion")

        for effect in reversed(preset.post_effects):
            if callable(effect):
                print(f"    - {effect.func.__name__.replace('_', ' ')}")
            else:
                print(f"    - {effect.name.replace('_', ' ').replace('-', ' ')}")

    if preset.octave_effects:
        print("  - Per-Octave:")

        for effect in reversed(preset.octave_effects):
            if callable(effect):
                print(f"    - {effect.func.__name__.replace('_', ' ')}")
            else:
                print(f"    - {effect.name.replace('_', ' ').replace('-', ' ')}")

    print("")
    print("  - Settings:")
    for (k, v) in sorted(preset.settings.items()):
        if isinstance(v, Enum):
            print(f"    - {k.replace('_', ' ')}: {v.name.replace('_', ' ')}")
        elif isinstance(v, float):
            print(f"    - {k.replace('_', ' ')}: {round(v, 3)}")
        else:
            print(f"    - {k.replace('_', ' ')}: {v}")

    print("")


@main.command(help="Apply an effect to a .png or .jpg image")
@cli.seed_option()
@cli.filename_option(default='mangled.png')
@cli.option('--no-resize', is_flag=True, help="Don't resize image. May break some presets.")
@cli.time_option()
@click.option('--speed', help="Animation speed", type=float, default=0.25)
@click.argument('preset_name', type=click.Choice(['random'] + sorted(EFFECT_PRESETS)))
@click.argument('input_filename')
@click.pass_context
def effect(ctx, seed, filename, no_resize, time, speed, preset_name, input_filename):
    if not seed:
        seed = random.randint(1, MAX_SEED_VALUE)

    value.set_seed(seed)
    reload_presets(PRESETS)

    input_shape = util.shape_from_file(input_filename)

    input_shape[2] = min(input_shape[2], 3)

    tensor = tf.image.convert_image_dtype(util.load(input_filename, channels=input_shape[2]), dtype=tf.float32)

    if preset_name == "random":
        preset_name = list(EFFECT_PRESETS)[random.randint(0, len(EFFECT_PRESETS) - 1)]

    print(f"{preset_name} (seed: {seed})")

    preset = EFFECT_PRESETS[preset_name]

    if no_resize:
        shape = input_shape

    else:
        shape = [1024, 1024, input_shape[2]]

        tensor = effects.square_crop_and_resize(tensor, input_shape, shape[0])

    try:
        preset.render(seed=seed, tensor=tensor, shape=shape, time=time, speed=speed, filename=filename)

    except Exception as e:
        util.logger.error(f"preset.render() failed: {e}\nSeed: {seed}\nArgs: {preset.__dict__}")
        raise


@main.command(help="Generate a .gif or .mp4 from preset")
@cli.width_option(default=512)
@cli.height_option(default=512)
@cli.channels_option()
@cli.seed_option()
@cli.option('--effect-preset', type=click.Choice(["random"] + sorted(EFFECT_PRESETS)))
@cli.filename_option(default='ani.gif')
@cli.option('--save-frames', default=None, type=click.Path(exists=True, dir_okay=True))
@cli.option('--frame-count', type=int, default=50, help="How many frames total")
@cli.option('--watermark', type=str)
@cli.option('--preview-filename', type=click.Path(exists=False))
@click.option('--with-alt-text', help="Generate alt text (requires OpenAI key)", is_flag=True, default=False)
@click.argument('preset_name', type=click.Choice(['random'] + sorted(GENERATOR_PRESETS)))
@click.pass_context
def animation(ctx, width, height, channels, seed, effect_preset, filename, save_frames, frame_count, watermark, preview_filename, with_alt_text, preset_name):
    if seed is None:
        seed = random.randint(1, MAX_SEED_VALUE)

    value.set_seed(seed)
    reload_presets(PRESETS)

    if preset_name == 'random':
        preset_name = list(GENERATOR_PRESETS)[random.randint(0, len(GENERATOR_PRESETS) - 1)]

    if effect_preset == 'random':
        effect_preset = list(EFFECT_PRESETS)[random.randint(0, len(EFFECT_PRESETS) - 1)]

    if effect_preset:
        print(f"{preset_name} vs. {effect_preset} (seed: {seed})")
    else:
        print(f"{preset_name} (seed: {seed})")

    preset = GENERATOR_PRESETS[preset_name]

    caption = None

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(frame_count):
            frame_filename = f'{tmp}/{i:04d}.png'

            common_params = ['--seed', str(seed),
                             '--time', f'{i/frame_count:0.4f}',
                             '--filename', frame_filename]

            extra_params = []
            if with_alt_text and i == 0:
                extra_params = ['--with-alt-text']

            output = subprocess.check_output(['noisemaker', 'generator', preset_name,
                                              '--speed', str(_use_reasonable_speed(preset, frame_count)),
                                              '--height', str(height),
                                              '--width', str(width)] + common_params + extra_params,
                                              universal_newlines=True).strip().split("\n")

            if with_alt_text and i == 0:
                if len(output) == 6:  # Useless extra crap that Tensorflow on Apple Silicon spews to stdout
                    print(output[2])
                else:
                    print(output[1])

            if effect_preset:
                util.check_call(['noisemaker', 'effect', effect_preset, frame_filename,
                                 '--no-resize',
                                 '--speed', str(_use_reasonable_speed(EFFECT_PRESETS[effect_preset], frame_count))] + common_params)

            if save_frames:
                shutil.copy(frame_filename, save_frames)

            if watermark:
                util.watermark(watermark, frame_filename)

            if preview_filename and i == 0:
                shutil.copy(frame_filename, preview_filename)

        if filename.endswith(".mp4"):
            # these settings are bad and they should feel bad. TODO: higher quality output
            util.check_call(['ffmpeg',
                             '-y',  # overwrite existing
                             '-framerate', '50',
                             '-i', f'{tmp}/%04d.png',
                             '-c:v', 'libx264',  # because this is what twitter wants
                             '-pix_fmt', 'yuv420p',  # because this is what twitter wants
                             '-b:v', '1700000',  # maximum allowed bitrate for 720x720 (2048k), minus some encoder overhead
                             '-s', '720x720',  # a twitter-recommended size
                             filename])

        else:
            util.magick(f'{tmp}/*png', filename)


@main.command(help="Blend a directory of .png or .jpg images")
@cli.input_dir_option(required=True)
@cli.filename_option(default="collage.png")
@click.option("--control-filename", help="Control image filename (optional)")
@cli.time_option()
@click.option('--speed', help="Animation speed", type=float, default=0.25)
@cli.seed_option()
@click.pass_context
def mashup(ctx, input_dir, filename, control_filename, time, speed, seed):
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
        control_shape = util.shape_from_file(control_filename)
        control = tf.image.convert_image_dtype(util.load(control_filename, channels=control_shape[2]), dtype=tf.float32)

    else:
        control = collage_images.pop()

    shape = tf.shape(control)  # All images need to be the same size!

    control = value.value_map(control, shape, keepdims=True)

    base = generators.basic(freq=random.randint(2, 5), shape=shape, lattice_drift=random.randint(0, 1), hue_range=random.random(),
                            seed=seed, time=time, speed=speed)

    value_shape = value.value_shape(shape)

    control = value.convolve(kernel=effects.ValueMask.conv2d_blur, tensor=control, shape=value_shape)

    tensor = effects.blend_layers(control, shape, random.random() * .5, *collage_images)

    tensor = value.blend(tensor, base, .125 + random.random() * .125)

    tensor = effects.bloom(tensor, shape, alpha=.25 + random.random() * .125)
    tensor = effects.shadow(tensor, shape, alpha=.25 + random.random() * .125, reference=control)

    tensor = tf.image.adjust_brightness(tensor, .1)
    tensor = tf.image.adjust_contrast(tensor, 1.5)

    util.save(tensor, filename)

    print('mashup')


def _use_reasonable_speed(preset, frame_count):
    """Return a reasonable speed parameter for the given animation length."""

    return preset.settings.get("speed", 0.25) * (frame_count / 50.0)
