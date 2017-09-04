import click
import tensorflow as tf

from noisemaker.util import save

import noisemaker.cli as cli
import noisemaker.effects as effects
import noisemaker.points as points


@click.command(help="""
        Collagemaker - Image collage tool

        https://github.com/aayars/py-noisemaker
        """, context_settings=cli.CLICK_CONTEXT_SETTINGS)
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
def main(ctx, width, height, input_dir, voronoi_func, voronoi_nth, point_freq, point_distrib, point_drift, name):
    shape = [height, width, 3]

    x, y = points.point_cloud(point_freq, distrib=point_distrib, shape=shape, drift=point_drift)

    tensor = effects.voronoi(None, shape, diagram_type=effects.VoronoiDiagramType.collage, xy=(x, y, len(x)), input_dir=input_dir)

    with tf.Session().as_default():
        save(tensor, name)

    print(name)