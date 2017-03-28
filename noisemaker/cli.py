import click
import tensorflow as tf

import noisemaker.generators as generators
import noisemaker.effects as effects


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
@click.option("--emboss", is_flag=True, default=False, help="Emboss effect (convolution kernel)")
@click.option("--shadow", is_flag=True, default=False, help="Shadow effect (convolution kernel)")
@click.option("--edges", is_flag=True, default=False, help="Edges effect (convolution kernel)")
@click.option("--sharpen", is_flag=True, default=False, help="Sharpen effect (convolution kernel)")
@click.option("--unsharp-mask", is_flag=True, default=False, help="Unsharp Mask effect (convolution kernel)")
@click.pass_context
def main(ctx, **kwargs):
    ctx.obj = kwargs


@main.command(help="Scaled random values with a normal distribution")
@click.option("--freq", type=int, default=4, help="Noise frequency per image height")
@click.option("--width", type=int, default=1024, help="Image output width")
@click.option("--height", type=int, default=1024, help="Image output height")
@click.option("--channels", type=int, default=3, help="Channel count. 1=Gray, 3=RGB, others may not work.")
@click.option("--ridged/--no-ridged", is_flag=True, default=False, help="\"Crease\" in the middle. (1 - unsigned((n-.5)*2))")
@click.option("--wavelet/--no-wavelet", is_flag=True, default=False, help="Maybe not wavelets this time?")
@click.option("--displacement", type=float, default=0.0, help="Self-displacement gradient. Current implementation is slow :(")
@click.option("--spline-order", type=int, default=3, help="Spline point count. 0=Constant, 1=Linear, 3=Bicubic, others may not work.")
@click.option("--name", default="gaussian", help="Base filename for image output")
@click.pass_context
def gaussian(ctx, freq, width, height, channels, ridged, wavelet, displacement, name):
    with tf.Session().as_default():
        tensor = generators.gaussian(freq, width, height, channels, ridged=ridged, wavelet=wavelet, displacement=displacement,
                                     spline_order=spline_order)

        tensor = _post_process(tensor, ctx.obj)

        _save(tensor, name)


@main.command(help="Multiple gaussian layers (octaves). For each octave: freq increases, amplitude decreases.")
@click.option("--freq", type=int, default=4, help="Bottom layer frequency per image height")
@click.option("--width", type=int, default=1024, help="Image output width")
@click.option("--height", type=int, default=1024, help="Image output height")
@click.option("--channels", type=int, default=3, help="Channel count. 1=Gray, 3=RGB, others may not work.")
@click.option("--ridged/--no-ridged", is_flag=True, default=True, help="\"Crease\" in the middle. (1 - unsigned((n-.5)*2))")
@click.option("--wavelet/--no-wavelet", is_flag=True, default=False, help="Maybe not wavelets this time?")
@click.option("--displacement", type=float, default=0.0, help="Self-displacement gradient. Current implementation is slow :(")
@click.option("--spline-order", type=int, default=3, help="Spline point count. 0=Constant, 1=Linear, 3=Bicubic, others may not work.")
@click.option("--octaves", type=int, default=3, help="Octave count. Number of multi-res layers. Typically 1-8")
@click.option("--name", default="multires", help="Base filename for image output")
@click.pass_context
def multires(ctx, freq, width, height, channels, octaves, ridged, wavelet, displacement, spline_order, name):
    with tf.Session().as_default():
        tensor = generators.multires(freq, width, height, channels, octaves, ridged=ridged, wavelet=wavelet, displacement=displacement,
                                     spline_order=spline_order)

        tensor = _post_process(tensor, ctx.obj)

        _save(tensor, name)
