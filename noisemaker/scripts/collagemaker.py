import os
import random

import click
import tensorflow as tf

from noisemaker.util import save

import noisemaker.cli as cli
import noisemaker.effects as effects
import noisemaker.points as points


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

    tensor = effects.voronoi(None, shape, diagram_type=effects.VoronoiDiagramType.collage, xy=(x, y, len(x)), nth=voronoi_nth,
                             input_dir=input_dir, image_count=point_freq)

    with tf.Session().as_default():
        save(tensor, name)

    print(name)