import click
import tensorflow as tf

import noisemaker.generators as generators
import noisemaker.effects as effects


def _save(tensor, name="out"):
    tensor = effects.normalize(tensor)
    tensor = tf.image.convert_image_dtype(tensor, tf.uint8, saturate=True)

    png = tf.image.encode_png(tensor).eval()

    with open("{0}.png".format(name), "wb") as fh:
        fh.write(png)

    print("Saved noise to {0}.png".format(name))


def _post_process(tensor, args):
    """
    """

    for kernel in effects.ConvKernel:
        if args.get(kernel.name):
            tensor =  effects.convolve(kernel, tensor)

    return tensor


@click.group()
@click.option("--emboss", is_flag=True, default=False)
@click.option("--shadow", is_flag=True, default=False)
@click.option("--edges", is_flag=True, default=False)
@click.option("--sharpen", is_flag=True, default=False)
@click.option("--unsharp-mask", is_flag=True, default=False)
@click.pass_context
def main(ctx, **kwargs):
    """
    """

    ctx.obj = kwargs


@main.command()
@click.option("--freq", type=int, default=4)
@click.option("--width", type=int, default=1024)
@click.option("--height", type=int, default=1024)
@click.option("--channels", type=int, default=3)
@click.option("--ridged/--no-ridged", is_flag=True, default=False)
@click.option("--wavelet/--no-wavelet", is_flag=True, default=False)
@click.option("--displacement", type=float, default=0.0)
@click.argument("name", required=False)
@click.pass_context
def gaussian(ctx, freq, width, height, channels, ridged, wavelet, displacement, name):
    with tf.Session().as_default():
        tensor = generators.gaussian(freq, width, height, channels, ridged=ridged, wavelet=wavelet, displacement=displacement)

        tensor = _post_process(tensor, ctx.obj)

        _save(tensor, name or "gaussian")


@main.command()
@click.option("--freq", type=int, default=4)
@click.option("--width", type=int, default=1024)
@click.option("--height", type=int, default=1024)
@click.option("--channels", type=int, default=3)
@click.option("--octaves", type=int, default=3)
@click.option("--ridged/--no-ridged", is_flag=True, default=True)
@click.option("--wavelet/--no-wavelet", is_flag=True, default=True)
@click.option("--displacement", type=float, default=0.0)
@click.option("--layer-displacement", type=float, default=0.0)
@click.argument("name", required=False)
@click.pass_context
def multires(ctx, freq, width, height, channels, octaves, ridged, wavelet, displacement, layer_displacement, name):
    with tf.Session().as_default():
        tensor = generators.multires(freq, width, height, channels, octaves, ridged=ridged, wavelet=wavelet, displacement=displacement, layer_displacement=layer_displacement)

        tensor = _post_process(tensor, ctx.obj)

        _save(tensor, name or "multires")
