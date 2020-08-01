import os
import random

import click
import tensorflow as tf

from noisemaker.util import save

import noisemaker.cli as cli
import noisemaker.effects as effects
import noisemaker.generators as generators
import noisemaker.points as points
import noisemaker.util as util


@click.group(help="""
        Collagemaker - Image collage tool

        https://github.com/aayars/py-noisemaker
        """, context_settings=cli.CLICK_CONTEXT_SETTINGS)
def main():
    pass


@main.command()
@cli.width_option()
@cli.height_option()
@cli.input_dir_option()
@cli.point_freq_option(default=None, help="(Optional) Max image count")
@cli.name_option(default="collage.png")
@click.pass_context
def auto(ctx, width, height, input_dir, point_freq, name):
    filenames = [f for f in os.listdir(input_dir) if f.endswith(".png") or f.endswith(".jpg")]

    if not point_freq:
        point_freq = min(random.randint(3, 5), len(filenames))

    if point_freq == 0:
        return

    point_freq = min(10, point_freq)

    voronoi_func = random.randint(1, 3)
    voronoi_nth = random.randint(0, point_freq - 1)

    distribs = [d for d in points.PointDistribution]
    point_distrib = distribs[random.randint(1, len(distribs) - 1)]

    point_drift = random.random() if random.random() < .125 else 0

    render(ctx, width, height, input_dir, voronoi_func, voronoi_nth, point_freq, point_distrib, point_drift, name)

    print('mashup')


@main.command()
@cli.width_option()
@cli.height_option()
@cli.input_dir_option()
@cli.voronoi_func_option()
@cli.voronoi_nth_option()
@cli.point_freq_option(default=5)
@cli.point_distrib_option()
@cli.point_drift_option()
@cli.name_option(default="collage.png")
@click.pass_context
def advanced(*args):
    render(*args)


def render(ctx, width, height, input_dir, voronoi_func, voronoi_nth, point_freq, point_distrib, point_drift, name):
    shape = [height, width, 3]

    x, y = points.point_cloud(point_freq, distrib=point_distrib, shape=shape, drift=point_drift)

    base = generators.basic(freq=random.randint(2, 4), shape=shape, lattice_drift=random.randint(0, 1), hue_range=random.random())

    tensor = effects.voronoi(base, shape, diagram_type=effects.VoronoiDiagramType.collage, xy=(x, y, len(x)), nth=voronoi_nth,
                             input_dir=input_dir, alpha=.333 + random.random() * .333)

    tensor = effects.bloom(tensor, shape, alpha=.333 + random.random() * .333)

    with tf.compat.v1.Session().as_default():
        save(tensor, name)

    print('mashup')


@main.command()
@cli.width_option()
@cli.height_option()
@cli.input_dir_option(required=True)
@cli.name_option(default="collage.png")
@click.option("--control-filename", help="Control image filename (optional)")
@click.option("--retro-upscale", is_flag=True, help="Nearest-neighbor upsample (for small images)")
@click.pass_context
def basic(ctx, width, height, input_dir, name, control_filename, retro_upscale):
    shape = [height, width, 3]  # Any shape you want, as long as it's [1024, 1024, 3]

    filenames = []

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(('.png', '.jpg')):
                filenames.append(os.path.join(root, filename))

    collage_count = min(random.randint(4, 6), len(filenames))
    collage_images = []

    for i in range(collage_count + 1):
        index = random.randint(0, len(filenames) - 1)

        input_filename = os.path.join(input_dir, filenames[index])

        collage_input = tf.image.convert_image_dtype(util.load(input_filename, channels=3), dtype=tf.float32)

        input_shape = effects.shape_from_file(input_filename)

        if retro_upscale:
            input_shape = [input_shape[0] * 2, input_shape[1] * 2, input_shape[2]]

            collage_input = effects.resample(collage_input, input_shape, spline_order=0)

        collage_input = effects.square_crop_and_resize(collage_input, input_shape, 1024)

        collage_images.append(collage_input)

    base = generators.basic(freq=random.randint(2, 5), shape=shape, lattice_drift=random.randint(0, 1), hue_range=random.random())

    if control_filename:
        control = tf.image.convert_image_dtype(util.load(control_filename, channels=1), dtype=tf.float32)

        control = effects.square_crop_and_resize(control, effects.shape_from_file(control_filename), 1024)

        control = effects.value_map(control, shape, keep_dims=True)

    else:
        control = effects.value_map(collage_images.pop(), shape, keep_dims=True)

    control = effects.convolve(effects.ValueMask.conv2d_blur, control, [height, width, 1])

    with tf.compat.v1.Session().as_default():
        # sort collage images by brightness
        collage_images = [j[1] for j in sorted([(tf.reduce_sum(i).eval(), i) for i in collage_images])]

        tensor = effects.blend_layers(control, shape, random.random() * .5, *collage_images)

        tensor = effects.blend(tensor, base, .125 + random.random() * .125)

        tensor = effects.bloom(tensor, shape, alpha=.25 + random.random() * .125)
        tensor = effects.shadow(tensor, shape, alpha=.25 + random.random() * .125, reference=control)

        tensor = tf.image.adjust_brightness(tensor, .05)
        tensor = tf.image.adjust_contrast(tensor, 1.25)

        save(tensor, name)

    print('mashup')
