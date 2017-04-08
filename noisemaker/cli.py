import click
import tensorflow as tf

import noisemaker.effects as effects
import noisemaker.generators as generators


def _save(tensor, name="out"):
    """
    Save as PNG. Prints a message to stdout.

    TODO: Probably put this in a real library somewhere. Support other image formats.

    :param Tensor tensor:
    :param str name:
    :return: None
    """

    tensor = effects.normalize(tensor)
    tensor = tf.image.convert_image_dtype(tensor, tf.uint8, saturate=True)

    png = tf.image.encode_png(tensor).eval()

    with open("{0}.png".format(name), "wb") as fh:
        fh.write(png)

    print("Saved noise to {0}.png".format(name))


def _post_process(tensor, args):
    """
    Apply post-processing functions from given CLI options.

    :param Tensor tensor:
    :param dict args:
    :return: Tensor
    """

    for kernel in effects.ConvKernel:
        if args.get(kernel.name):
            tensor =  effects.convolve(kernel, tensor)

    return tensor


@click.group(help="""
        Noisemaker - Visual noise generator

        https://github.com/aayars/py-noisemaker

        Effect options should be specified before the command name.

        --help is available for each command.
        """)
@click.option("--emboss", is_flag=True, default=False, help="Emboss")
@click.option("--shadow", is_flag=True, default=False, help="Shadow")
@click.option("--edges", is_flag=True, default=False, help="Edges")
@click.option("--sharpen", is_flag=True, default=False, help="Sharpen")
@click.option("--unsharp-mask", is_flag=True, default=False, help="Unsharp Mask")
@click.option("--invert", is_flag=True, default=False, help="Invert")
@click.pass_context
def main(ctx, **kwargs):
    ctx.obj = kwargs


@main.command(help="Scaled random values with a normal distribution")
@click.option("--freq", type=int, default=4, help="Heightwise noise frequency")
@click.option("--width", type=int, default=1024, help="Image output width")
@click.option("--height", type=int, default=1024, help="Image output height")
@click.option("--channels", type=int, default=3, help="Channel count. 1=Gray, 3=RGB, others may not work.")
@click.option("--ridged/--no-ridged", is_flag=True, default=False, help="\"Crease\" in the middle. (1 - unsigned((n-.5)*2))")
@click.option("--wavelet/--no-wavelet", is_flag=True, default=False, help="Maybe not wavelets this time?")
@click.option("--refract", type=float, default=0.0, help="Self-distortion gradient.")
@click.option("--reindex", type=float, default=0.0, help="Self-reindexing gradient.")
@click.option("--clut", type=str, default=0.0, help="Color lookup table (PNG or JPG)")
@click.option("--horizontal", is_flag=True, default=False, help="Preserve clut Y axis")
@click.option("--clut-range", type=float, default=.5, help="Gather distance for clut.")
@click.option("--worms", is_flag=True, default=False, help="Do worms.")
@click.option("--spline-order", type=int, default=3, help="Spline point count. 0=Constant, 1=Linear, 3=Bicubic, others may not work.")
@click.option("--seed", type=int, required=False, help="Random seed for reproducible output.")
@click.option("--name", default="gaussian", help="Base filename for image output")
@click.pass_context
def gaussian(ctx, freq, width, height, channels, ridged, wavelet, refract, reindex, clut, horizontal, clut_range, worms, spline_order, seed, name):
    with tf.Session().as_default():
        tensor = generators.gaussian(freq, width, height, channels, ridged=ridged, wavelet=wavelet, refract=refract, reindex=reindex,
                                     clut=clut, horizontal=horizontal, clut_range=clut_range, worms=worms, spline_order=spline_order, seed=seed)

        tensor = _post_process(tensor, ctx.obj)

        _save(tensor, name)


@main.command(help="Multiple gaussian layers (octaves). For each octave: freq increases, amplitude decreases.")
@click.option("--freq", type=int, default=4, help="Heightwise bottom layer frequency")
@click.option("--width", type=int, default=1024, help="Image output width")
@click.option("--height", type=int, default=1024, help="Image output height")
@click.option("--channels", type=int, default=3, help="Channel count. 1=Gray, 3=RGB, others may not work.")
@click.option("--ridged/--no-ridged", is_flag=True, default=True, help="\"Crease\" in the middle. (1 - unsigned((n-.5)*2))")
@click.option("--wavelet/--no-wavelet", is_flag=True, default=False, help="Maybe not wavelets this time?")
@click.option("--refract", type=float, default=0.0, help="Self-distortion gradient.")
@click.option("--layer-refract", type=float, default=0.0, help="Per-octave self-distortion gradient.")
@click.option("--reindex", type=float, default=0.0, help="Self-reindexing gradient.")
@click.option("--layer-reindex", type=float, default=0.0, help="Per-octave self-reindexing gradient.")
@click.option("--clut", type=str, default=0.0, help="Color lookup table (PNG or JPG)")
@click.option("--horizontal", is_flag=True, default=False, help="Preserve clut Y axis")
@click.option("--clut-range", type=float, default=.5, help="Gather distance for clut.")
@click.option("--worms", is_flag=True, default=False, help="Do worms.")
@click.option("--spline-order", type=int, default=3, help="Spline point count. 0=Constant, 1=Linear, 3=Bicubic, others may not work.")
@click.option("--seed", type=int, required=False, help="Random seed for reproducible output.")
@click.option("--octaves", type=int, default=3, help="Octave count. Number of multi-res layers. Typically 1-8")
@click.option("--name", default="multires", help="Base filename for image output")
@click.pass_context
def multires(ctx, freq, width, height, channels, octaves, ridged, wavelet, refract, layer_refract, reindex, layer_reindex,
             clut, horizontal, clut_range, worms, spline_order, seed, name):

    with tf.Session().as_default():
        tensor = generators.multires(freq, width, height, channels, octaves, ridged=ridged, wavelet=wavelet,
                                     refract=refract, layer_refract=layer_refract,
                                     reindex=reindex, layer_reindex=layer_reindex, clut=clut, horizontal=horizontal,
                                     clut_range=clut_range, worms=worms, spline_order=spline_order, seed=seed,
                                     )

        tensor = _post_process(tensor, ctx.obj)

        _save(tensor, name)
