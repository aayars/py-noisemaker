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
@click.option('--with-supersample', help="Apply x2 supersample anti-aliasing", is_flag=True, default=False)
@click.option('--with-fxaa', help="Apply FXAA anti-aliasing", is_flag=True, default=False)
@click.option('--with-ai', help="AI: Apply image-to-image (requires stability.ai key)", is_flag=True, default=False)
@click.option('--with-upscale', help="AI: Apply x4 upscale (requires stability.ai key)", is_flag=True, default=False)
@click.option('--with-alt-text', help="AI: Generate alt text (requires OpenAI key)", is_flag=True, default=False)
@click.option('--stability-model', help="AI: Override default stability.ai model", type=str, default=None)
@click.option('--debug-print', help="Debug: Print ancestors and settings to STDOUT", is_flag=True, default=False)
@click.option('--debug-out', help="Debug: Log ancestors and settings to file", type=click.Path(dir_okay=False), default=None)
@click.argument('preset_name', type=click.Choice(["random"] + sorted(GENERATOR_PRESETS)))
@click.pass_context
def generate(ctx, width, height, time, speed, seed, filename, with_alpha, with_supersample, with_fxaa, with_ai, with_upscale,
             with_alt_text, stability_model, debug_print, debug_out, preset_name):
    if not seed:
        seed = random.randint(1, MAX_SEED_VALUE)

    value.set_seed(seed)
    reload_presets(PRESETS)

    if preset_name == "random":
        preset_name = list(GENERATOR_PRESETS)[random.randint(0, len(GENERATOR_PRESETS) - 1)]

    preset = GENERATOR_PRESETS[preset_name]

    if debug_print or debug_out:
        debug_text = _debug_print(seed, preset, with_alpha, with_supersample, with_fxaa, with_ai, with_upscale, stability_model)

        if debug_print:
            for line in debug_text:
                print(line)

        if debug_out is not None:
            with open(debug_out, 'w') as fh:
                for line in debug_text:
                    fh.write(line + "\n")

    try:
        preset.render(seed, shape=[height, width, None], time=time, speed=speed, filename=filename,
                      with_alpha=with_alpha, with_supersample=with_supersample, with_fxaa=with_fxaa,
                      with_ai=with_ai, with_upscale=with_upscale, stability_model=stability_model)

    except Exception as e:
        util.logger.error(f"preset.render() failed: {e}\nSeed: {seed}\nArgs: {preset.__dict__}")
        raise

    if preset.ai_success:
        if preset.ai_settings['model'] in ('sd3', 'core', 'ultra'):
            ai_label = 'stable-image ai'
        else:
            ai_label = 'ai'

        print(f"{preset_name} (procedural) vs. {preset.ai_settings['model']} ({ai_label})")

    else:
        print(preset_name)

    if with_alt_text:
        print(ai.describe(preset.name.replace('-', ' '), preset.ai_settings.get("prompt"), filename))


def _debug_print(seed, preset, with_alpha, with_supersample, with_fxaa, with_ai, with_upscale, stability_model):
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
    first_column.append(f"  - with fxaa: {with_fxaa}")
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
@click.option('--with-fxaa', help="Apply FXAA anti-aliasing", is_flag=True, default=False)
@cli.time_option()
@click.option('--speed', help="Animation speed", type=float, default=0.25)
@click.argument('preset_name', type=click.Choice(['random'] + sorted(EFFECT_PRESETS)))
@click.argument('input_filename')
@click.pass_context
def apply(ctx, seed, filename, no_resize, with_fxaa, time, speed, preset_name, input_filename):
    if not seed:
        seed = random.randint(1, MAX_SEED_VALUE)

    value.set_seed(seed)
    reload_presets(PRESETS)

    input_shape = util.shape_from_file(input_filename)

    input_shape[2] = min(input_shape[2], 3)

    tensor = tf.image.convert_image_dtype(util.load(input_filename, channels=input_shape[2]), dtype=tf.float32)

    if preset_name == "random":
        preset_name = list(EFFECT_PRESETS)[random.randint(0, len(EFFECT_PRESETS) - 1)]

    print(preset_name)

    preset = EFFECT_PRESETS[preset_name]

    if no_resize:
        shape = input_shape

    else:
        shape = [1024, 1024, input_shape[2]]

        tensor = effects.square_crop_and_resize(tensor, input_shape, shape[0])

    try:
        preset.render(seed=seed, tensor=tensor, shape=shape, with_fxaa=with_fxaa, time=time, speed=speed, filename=filename)

    except Exception as e:
        util.logger.error(f"preset.render() failed: {e}\nSeed: {seed}\nArgs: {preset.__dict__}")
        raise


@main.command(help="Generate an animation from preset")
@cli.width_option(default=512)
@cli.height_option(default=512)
@cli.seed_option()
@cli.option("--effect-preset", type=click.Choice(["random"] + sorted(EFFECT_PRESETS)))
@cli.filename_option(default="animation.mp4")
@cli.option("--save-frames", default=None, type=click.Path(exists=True, dir_okay=True))
@cli.option("--frame-count", type=int, default=50, help="How many frames total")
@cli.option("--watermark", type=str)
@cli.option("--preview-filename", type=click.Path(exists=False))
@click.option("--with-alt-text", help="Generate alt text (requires OpenAI key)", is_flag=True, default=False)
@click.option("--with-supersample", help="Apply x2 supersample anti-aliasing", is_flag=True, default=False)
@click.option("--with-fxaa", help="Apply FXAA anti-aliasing", is_flag=True, default=False)
@click.option(
    "--target-duration",
    type=float,
    default=None,
    help="Stretch output to this duration (seconds) using motion-compensated interpolation"
)
@click.argument("preset_name", type=click.Choice(["random"] + sorted(GENERATOR_PRESETS)))
@click.pass_context
def animate(
    ctx,
    width,
    height,
    seed,
    effect_preset,
    filename,
    save_frames,
    frame_count,
    watermark,
    preview_filename,
    with_alt_text,
    with_supersample,
    with_fxaa,
    preset_name,
    target_duration,
):
    if seed is None:
        seed = random.randint(1, MAX_SEED_VALUE)

    value.set_seed(seed)
    reload_presets(PRESETS)

    if preset_name == "random":
        preset_name = random.choice(list(GENERATOR_PRESETS))

    if effect_preset == "random":
        effect_preset = random.choice(list(EFFECT_PRESETS))

    if effect_preset:
        print(f"{preset_name} vs. {effect_preset}")
    else:
        print(preset_name)

    generator = GENERATOR_PRESETS[preset_name]
    effect = EFFECT_PRESETS.get(effect_preset) if effect_preset else None

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(frame_count):
            frame_path = os.path.join(tmp, f"{i:04d}.png")
            time_frac = i / frame_count
            gen_speed = _use_reasonable_speed(generator, frame_count)

            try:
                generator.render(
                    seed=seed,
                    shape=[height, width, None],
                    time=time_frac,
                    speed=gen_speed,
                    filename=frame_path,
                    with_alpha=False,
                    with_supersample=with_supersample,
                    with_fxaa=with_fxaa,
                )
            except Exception as e:
                util.logger.error(f"Generator render failed: {e}\nSeed: {seed}\nArgs: {generator.__dict__}")
                raise

            if with_alt_text and i == 0:
                print(ai.describe(generator.name.replace("-", " "), generator.ai_settings.get("prompt"), frame_path))

            if effect:
                input_shape = util.shape_from_file(frame_path)
                input_shape[2] = min(input_shape[2], 3)
                tensor = tf.image.convert_image_dtype(
                    util.load(frame_path, channels=input_shape[2]), dtype=tf.float32
                )

                shape = input_shape

                try:
                    effect.render(
                        seed=seed,
                        tensor=tensor,
                        shape=shape,
                        with_fxaa=with_fxaa,
                        time=time_frac,
                        speed=_use_reasonable_speed(effect, frame_count),
                        filename=frame_path,
                    )
                except Exception as e:
                    util.logger.error(f"Effect render failed: {e}\nSeed: {seed}\nArgs: {effect.__dict__}")
                    raise

            if save_frames:
                shutil.copy(frame_path, save_frames)

            if watermark:
                util.watermark(watermark, frame_path)

            if preview_filename and i == 0:
                shutil.copy(frame_path, preview_filename)

        if filename.endswith(".mp4"):
            if target_duration is not None:
                first = os.path.join(tmp, "0000.png")
                last = os.path.join(tmp, f"{frame_count:04d}.png")
                shutil.copy(first, last)

                factor = 30 * target_duration / frame_count

                util.check_call([
                    "ffmpeg", "-y", "-framerate", "30",
                    "-i", os.path.join(tmp, "%04d.png"),
                    "-s", f"{width}x{height}",
                    "-vf", f"setpts={factor}*PTS,minterpolate=mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps=30",
                    "-c:v", "libx264", "-preset", "veryslow",
                    "-crf", "15", "-pix_fmt", "yuv420p",
                    "-b:v", "8000k", "-bufsize", "16000k",
                    filename,
                ])
            else:
                util.check_call([
                    "ffmpeg", "-y", "-framerate", "30",
                    "-i", os.path.join(tmp, "%04d.png"),
                    "-s", f"{width}x{height}",
                    "-c:v", "libx264", "-preset", "veryslow",
                    "-crf", "15", "-pix_fmt", "yuv420p",
                    "-b:v", "8000k", "-bufsize", "16000k",
                    filename,
                ])
        else:
            util.magick(f"{tmp}/*png", filename)


@main.command("magic-mashup", help="Animated collage from a directory of directories of frames")
@cli.input_dir_option(required=True)
@cli.width_option(default=512)
@cli.height_option(default=512)
@cli.seed_option()
@cli.option("--effect-preset", type=click.Choice(["random"] + sorted(EFFECT_PRESETS)))
@cli.filename_option(default="mashup.mp4")
@cli.option("--save-frames", default=None, type=click.Path(exists=True, dir_okay=True))
@cli.option("--frame-count", type=int, default=50, help="How many frames total")
@cli.option("--watermark", type=str)
@cli.option("--preview-filename", type=click.Path(exists=False))
@cli.option(
    "--target-duration",
    type=float,
    default=None,
    help="Stretch output to this duration (seconds) using motion-compensated interpolation"
)
@click.pass_context
def magic_mashup(
    ctx,
    input_dir,
    width,
    height,
    seed,
    effect_preset,
    filename,
    save_frames,
    frame_count,
    watermark,
    preview_filename,
    target_duration
):
    if seed is None:
        seed = random.randint(1, MAX_SEED_VALUE)

    value.set_seed(seed)

    if effect_preset == "random":
        effect_preset = random.choice(list(EFFECT_PRESETS))

    if effect_preset:
        print(f"magic-mashup vs. {effect_preset}")
    else:
        print(f"magic-mashup")

    effect = EFFECT_PRESETS.get(effect_preset) if effect_preset else None

    dirnames = [
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ]
    if not dirnames:
        click.echo(f"No subdirectories found in input dir {input_dir}")
        sys.exit(1)

    collage_count = min(random.randint(4, 6), len(dirnames))
    selected_dirs = random.sample(dirnames, collage_count)

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(frame_count):
            frame_path = os.path.join(tmp, f"{i:04d}.png")

            collage_images = []
            for dirname in selected_dirs:
                src_dir = os.path.join(input_dir, dirname)
                files = sorted(f for f in os.listdir(src_dir) if f.endswith(".png"))
                if i >= len(files):
                    continue
                src = os.path.join(src_dir, files[i])
                img = tf.image.convert_image_dtype(
                    util.load(src, channels=3),
                    dtype=tf.float32
                )
                collage_images.append(img)

            value.set_seed(seed)
            shape = [height, width, 3]

            base = generators.basic(
                freq=random.randint(2, 4),
                shape=shape,
                hue_range=random.random(),
                time=i / frame_count,
                speed=0.125
            )

            control_img = collage_images.pop() if collage_images else tf.zeros(shape)
            control = value.value_map(control_img, shape, keepdims=True)
            control = value.convolve(
                kernel=effects.ValueMask.conv2d_blur,
                tensor=control,
                shape=[shape[0], shape[1], 1]
            )

            tensor = effects.blend_layers(
                control, shape, random.random() * 0.5, *collage_images
            )
            tensor = value.blend(tensor, base, 0.125 + random.random() * 0.125)
            tensor = effects.bloom(tensor, shape, alpha=0.25 + random.random() * 0.125)
            tensor = effects.shadow(
                tensor, shape, alpha=0.25 + random.random() * 0.125, reference=control
            )
            tensor = tf.image.adjust_brightness(tensor, 0.1)
            tensor = tf.image.adjust_contrast(tensor, 1.5)

            if effect:
                try:
                    effect.render(
                        seed=seed,
                        tensor=tensor,
                        shape=shape,
                        time=i / frame_count,
                        speed=_use_reasonable_speed(effect, frame_count),
                        filename=frame_path,
                    )
                except Exception as e:
                    util.logger.error(f"Effect render failed: {e}\nSeed: {seed}\nArgs: {effect.__dict__}")
                    raise

            else:
                util.save(tensor, frame_path)

            if save_frames:
                shutil.copy(frame_path, save_frames)

            if watermark:
                util.watermark(watermark, frame_path)

            if preview_filename and i == 0:
                shutil.copy(frame_path, preview_filename)

        if filename.endswith(".mp4"):
            if target_duration is not None:
                first = os.path.join(tmp, "0000.png")
                last = os.path.join(tmp, f"{frame_count:04d}.png")
                shutil.copy(first, last)

                factor = 30 * target_duration / frame_count

                util.check_call([
                    "ffmpeg", "-y",
                    "-framerate", "30",
                    "-i", os.path.join(tmp, "%04d.png"),
                    "-s", f"{width}x{height}",
                    "-vf", f"setpts={factor}*PTS,minterpolate=mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps=30",
                    "-c:v", "libx264", "-preset", "veryslow",
                    "-crf", "15", "-pix_fmt", "yuv420p",
                    "-b:v", "8000k", "-bufsize", "16000k",
                    filename
                ])
            else:
                util.check_call([
                    "ffmpeg", "-y", "-framerate", "30",
                    "-i", os.path.join(tmp, "%04d.png"),
                    "-s", f"{width}x{height}",
                    "-c:v", "libx264", "-preset", "veryslow",
                    "-crf", "15", "-pix_fmt", "yuv420p",
                    "-b:v", "8000k", "-bufsize", "16000k",
                    filename
                ])
        else:
            util.check_call([
                "ffmpeg", "-y", "-framerate", "30",
                "-i", os.path.join(tmp, "%04d.png"),
                "-s", f"{width}x{height}",
                "-c:v", "libx264", "-preset", "veryslow",
                "-crf", "15", "-pix_fmt", "yuv420p",
                "-b:v", "8000k", "-bufsize", "16000k",
                filename
            ])


@main.command(help="Blend a directory of .png or .jpg images")
@cli.input_dir_option(required=True)
@cli.filename_option(default="mashup.png")
@click.option("--control-filename", help="Control image filename (optional)")
@cli.time_option()
@click.option('--speed', help="Animation speed", type=float, default=0.25)
@cli.seed_option()
@click.pass_context
def mashup(ctx, input_dir, filename, control_filename, time, speed, seed):
    filenames_list = []

    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith(('.png', '.jpg')):
                filenames_list.append(os.path.join(root, f))

    collage_count = min(random.randint(4, 6), len(filenames_list))
    collage_images = []

    for _ in range(collage_count + 1):
        src = random.choice(filenames_list)
        img = tf.image.convert_image_dtype(util.load(src, channels=3), dtype=tf.float32)
        collage_images.append(img)

    if control_filename:
        shape_ctrl = util.shape_from_file(control_filename)
        control_img = tf.image.convert_image_dtype(
            util.load(control_filename, channels=shape_ctrl[2]), dtype=tf.float32)
    else:
        control_img = collage_images.pop()

    shape = tf.shape(control_img)
    control = value.value_map(control_img, shape, keepdims=True)

    value.set_seed(seed)

    base = generators.basic(
        freq=random.randint(2, 5),
        shape=shape,
        lattice_drift=random.randint(0, 1),
        hue_range=random.random(),
        time=time,
        speed=speed
    )

    val_shape = value.value_shape(shape)
    control = value.convolve(
        kernel=effects.ValueMask.conv2d_blur,
        tensor=control,
        shape=val_shape
    )

    tensor = effects.blend_layers(control, shape, random.random() * 0.5, *collage_images)
    tensor = value.blend(tensor, base, 0.125 + random.random() * 0.125)
    tensor = effects.bloom(tensor, shape, alpha=0.25 + random.random() * 0.125)
    tensor = effects.shadow(
        tensor, shape, alpha=0.25 + random.random() * 0.125, reference=control
    )
    tensor = tf.image.adjust_brightness(tensor, 0.1)
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
