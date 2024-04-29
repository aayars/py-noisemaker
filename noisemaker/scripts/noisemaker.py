from enum import Enum

import os
import random
import shutil
import subprocess
import tempfile
import textwrap

import click
import tensorflow as tf

from noisemaker.composer import EFFECT_PRESETS, GENERATOR_PRESETS, reload_presets
from noisemaker.constants import ColorSpace, ValueDistribution
from noisemaker.presets import PRESETS

import noisemaker.ai as ai
import noisemaker.dreamer as dreamer
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
@cli.time_option()
@click.option('--speed', help="Animation speed", type=float, default=0.25)
@cli.seed_option()
@cli.filename_option(default='art.png')
@click.option('--with-alpha', help="Include alpha channel", is_flag=True, default=False)
@click.option('--with-supersample', help="Use x2 supersample for anti-aliasing", is_flag=True, default=False)
@click.option('--with-ai', help="AI: Apply image-to-image (requires stability.ai key)", is_flag=True, default=False)
@click.option('--with-upscale', help="AI: Apply x2 upscale (requires stability.ai key)", is_flag=True, default=False)
@click.option('--with-alt-text', help="AI: Generate alt text (requires OpenAI key)", is_flag=True, default=False)
@click.option('--stability-model', help="AI: Override default stability.ai model", type=str, default=None)
@click.option('--debug-print', help="Debug: Print ancestors and settings to STDOUT", is_flag=True, default=False)
@click.option('--debug-out', help="Debug: Log ancestors and settings to file", type=click.Path(dir_okay=False), default=None)
@click.argument('preset_name', type=click.Choice(["random"] + sorted(GENERATOR_PRESETS)))
@click.pass_context
def generate(ctx, width, height, time, speed, seed, filename, with_alpha, with_supersample, with_ai, with_upscale,
             with_alt_text, stability_model, debug_print, debug_out, preset_name):
    if not seed:
        seed = random.randint(1, MAX_SEED_VALUE)

    value.set_seed(seed)
    reload_presets(PRESETS)

    if preset_name == "random":
        preset_name = list(GENERATOR_PRESETS)[random.randint(0, len(GENERATOR_PRESETS) - 1)]

    preset = GENERATOR_PRESETS[preset_name]

    if debug_print or debug_out:
        debug_text = _debug_print(seed, preset, with_alpha, with_supersample, with_ai, with_upscale, stability_model)

        if debug_print:
            for line in debug_text:
                print(line)

        if debug_out is not None:
            with open(debug_out, 'w') as fh:
                for line in debug_text:
                    fh.write(line + "\n")

    try:
        preset.render(seed, shape=[height, width, None], time=time, speed=speed, filename=filename,
                      with_alpha=with_alpha, with_supersample=with_supersample, with_ai=with_ai,
                      with_upscale=with_upscale, stability_model=stability_model)

    except Exception as e:
        util.logger.error(f"preset.render() failed: {e}\nSeed: {seed}\nArgs: {preset.__dict__}")
        raise

    if preset.ai_success:
        print(f"{preset_name} vs. {preset.ai_settings['model']} (seed: {seed})")

    else:
        print(f"{preset_name} (seed: {seed})")

    if with_alt_text:
        print(ai.describe(preset.name.replace('-', ' '), preset.ai_settings.get("prompt"), filename))


def _debug_print(seed, preset, with_alpha, with_supersample, with_ai, with_upscale, stability_model):
    first_column = ["Layers:"]

    if preset.flattened_layers:
        first_column.append("  - Lineage (by newest):")

        if not preset.flattened_layers:
            first_column.append("    - None")

        for parent in reversed(preset.flattened_layers):
            first_column.append(f"    - {parent}")

    first_column.append("")
    first_column.append("  - Effects (by newest):")

    if not preset.final_effects and not with_ai and not preset.post_effects:
        first_column.append("    - None")
        first_column.append("")

    if preset.final_effects:
        first_column.append("    - Final Pass:")

        for effect in reversed(preset.final_effects):
            if callable(effect):
                first_column.append(f"      - {effect.func.__name__.replace('_', ' ')}")
            else:
                first_column.append(f"      - {effect.name.replace('_', ' ').replace('-', ' ')}")

        first_column.append("")

    if with_ai:
        first_column.append(f"    - AI Settings:")

        for (k, v) in sorted(preset.ai_settings.items()):
            if stability_model and k == 'model':
                v = stability_model

            for i, line in enumerate(textwrap.wrap(f"{k.replace('_', ' ')}: {v}", 42)):
                if i == 0:
                    first_column.append(f"      - {line}")
                else:
                    first_column.append(f"        {line}")

        first_column.append("")

    if preset.post_effects or with_ai:
        first_column.append("    - Post Pass:")

        if with_ai:
            first_column.append("      - stable diffusion")

        for effect in reversed(preset.post_effects):
            if callable(effect):
                first_column.append(f"      - {effect.func.__name__.replace('_', ' ')}")
            else:
                first_column.append(f"      - {effect.name.replace('_', ' ').replace('-', ' ')}")

        first_column.append("")

    if preset.octave_effects:
        first_column.append("    - Per-Octave Pass:")

        for effect in reversed(preset.octave_effects):
            if callable(effect):
                first_column.append(f"      - {effect.func.__name__.replace('_', ' ')}")
            else:
                first_column.append(f"      - {effect.name.replace('_', ' ').replace('-', ' ')}")

        first_column.append("")

    first_column.append("Canvas:")
    first_column.append(f"  - seed: {seed}")
    first_column.append(f"  - with alpha: {with_alpha}")
    first_column.append(f"  - with supersample: {with_supersample}")
    first_column.append(f"  - with upscale: {with_upscale}")
    first_column.append("")

    second_column = ["Settings:"]
    for (k, v) in sorted(preset.settings.items()):
        if isinstance(v, Enum):
            second_column.append(f"  - {k.replace('_', ' ')}: {v.name.replace('_', ' ')}")
        elif isinstance(v, float):
            second_column.append(f"  - {k.replace('_', ' ')}: {round(v, 3)}")
        else:
            second_column.append(f"  - {k.replace('_', ' ')}: {v}")

    second_column.append("")

    out = []

    for i in range(max(len(first_column), len(second_column))):
        if i < len(first_column):
            first = first_column[i]
        else:
            first = ""

        if i < len(second_column):
            second = second_column[i]
        else:
            second = ""

        out.append(f"{first:50} {second}")

    return out


@main.command(help="Apply an effect to a .png or .jpg image")
@cli.seed_option()
@cli.filename_option(default='mangled.png')
@cli.option('--no-resize', is_flag=True, help="Don't resize image. May break some presets.")
@cli.time_option()
@click.option('--speed', help="Animation speed", type=float, default=0.25)
@click.argument('preset_name', type=click.Choice(['random'] + sorted(EFFECT_PRESETS)))
@click.argument('input_filename')
@click.pass_context
def apply(ctx, seed, filename, no_resize, time, speed, preset_name, input_filename):
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
@cli.seed_option()
@cli.option('--effect-preset', type=click.Choice(["random"] + sorted(EFFECT_PRESETS)))
@cli.filename_option(default='ani.gif')
@cli.option('--save-frames', default=None, type=click.Path(exists=True, dir_okay=True))
@cli.option('--frame-count', type=int, default=50, help="How many frames total")
@cli.option('--watermark', type=str)
@cli.option('--preview-filename', type=click.Path(exists=False))
@click.option('--with-alt-text', help="Generate alt text (requires OpenAI key)", is_flag=True, default=False)
@click.option('--with-supersample', help="Use x2 supersample for anti-aliasing", is_flag=True, default=False)
@click.argument('preset_name', type=click.Choice(['random'] + sorted(GENERATOR_PRESETS)))
@click.pass_context
def animate(ctx, width, height,  seed, effect_preset, filename, save_frames, frame_count, watermark, preview_filename, with_alt_text, with_supersample, preset_name):
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

            if with_supersample:
                extra_params += ['--with-supersample']

            output = subprocess.check_output(['noisemaker', 'generate', preset_name,
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
                util.check_call(['noisemaker', 'apply', effect_preset, frame_filename,
                                 '--no-resize',
                                 '--speed', str(_use_reasonable_speed(EFFECT_PRESETS[effect_preset], frame_count))] + common_params)

            if save_frames:
                shutil.copy(frame_filename, save_frames)

            if watermark:
                util.watermark(watermark, frame_filename)

            if preview_filename and i == 0:
                shutil.copy(frame_filename, preview_filename)

        if filename.endswith(".mp4"):
            util.check_call(["ffmpeg",
                             "-framerate", "30",
                             "-i", f"{tmp}/%04d.png",
                             "-s", "1024x1024",
                             "-c:v", "libx264",
                             "-preset", "veryslow",
                             "-crf", "15",
                             "-pix_fmt", "yuv420p",
                             "-b:v", "8000k",
                             "-bufsize", "16000k",
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


@main.command(help="Let the machine dream whatever it wants")
@cli.width_option()
@cli.height_option()
@cli.filename_option(default='dream.png')
def dream(width, height, filename):
    name, prompt, description = dreamer.dream(width, height, filename=filename)

    print(name)
    print(prompt)
    print(description)
