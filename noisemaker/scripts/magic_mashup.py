import os
import random
import shutil
import subprocess
import sys
import tempfile

import click
import tensorflow as tf

import noisemaker.cli as cli
import noisemaker.effects as effects
import noisemaker.generators as generators
import noisemaker.util as util


@click.group(help="""
        Magic Mashup - Animated collage tool

        https://github.com/aayars/py-noisemaker
        """, context_settings=cli.CLICK_CONTEXT_SETTINGS)
def main():
    pass


@main.command()
@cli.input_dir_option(required=True)
@cli.int_option('--seed', required=True)
@cli.name_option(default="mashup.png")
@cli.option('--save-frames', default=None, type=click.Path(exists=True, dir_okay=True))
@click.pass_context
def frames(ctx, input_dir, seed, name, save_frames):
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(30):
            filename = f'{tmp}/{i:04d}.png'

            subprocess.check_call(['magic-mashup', 'frame',
                '--input-dir', input_dir,
                '--frame', str(i),
                '--seed', str(seed),
                '--name', filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            subprocess.check_call(['artmangler', 'crt', filename,
                '--no-resize',
                '--seed', str(seed),
                '--overrides', '{"distrib": "simplex", "speed": 0.25}',
                '--time', str(i / 30.0),
                '--name', filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if save_frames:
                shutil.copy(filename, save_frames)

        subprocess.check_call(['convert', '-delay', '5', f'{tmp}/*png', name])

    print('magic-mashup')


@main.command()
@cli.input_dir_option(required=True)
@cli.int_option('--frame', required=True)
@cli.int_option('--seed', required=True)
@cli.name_option(default="mashup.png")
@click.pass_context
def frame(ctx, input_dir, frame, seed, name):
    generators.set_seed(seed)

    shape = [512, 512, 3]

    dirnames = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    if not dirnames:
        click.echo("Couldn't determine directory names inside of input dir " + input_dir)
        sys.exit(1)

    collage_count = min(random.randint(4, 6), len(dirnames))
    collage_images = []

    for i in range(collage_count + 1):
        index = random.randint(0, len(dirnames) - 1)

        dirname = dirnames[index]

        filenames = [f for f in sorted(os.listdir(os.path.join(input_dir, dirnames[index]))) if f.endswith('.png')]

        if not filenames:
            continue

        input_filename = os.path.join(input_dir, dirnames[index], filenames[frame])

        collage_images.append(tf.image.convert_image_dtype(util.load(input_filename, channels=3), dtype=tf.float32))

    base = generators.basic(freq=random.randint(2, 4), shape=shape, hue_range=random.random(), time=frame/30.0, speed=0.125, distrib="simplex")

    control = effects.value_map(collage_images.pop(), shape, keepdims=True)

    control = effects.convolve(effects.ValueMask.conv2d_blur, control, [512, 512, 1])

    with tf.compat.v1.Session().as_default():
        tensor = effects.blend_layers(control, shape, random.random() * .5, *collage_images)

        tensor = effects.blend(tensor, base, .125 + random.random() * .125)

        tensor = effects.bloom(tensor, shape, alpha=.25 + random.random() * .125)
        tensor = effects.shadow(tensor, shape, alpha=.25 + random.random() * .125, reference=control)

        tensor = tf.image.adjust_brightness(tensor, .05)
        tensor = tf.image.adjust_contrast(tensor, 1.25)

        util.save(tensor, name)
