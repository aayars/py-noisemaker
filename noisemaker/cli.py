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


@click.group()
def main():
    pass


@main.command()
@click.option("--freq", type=int, default=4)
@click.option("--width", type=int, default=1024)
@click.option("--height", type=int, default=1024)
@click.option("--channels", type=int, default=3)
@click.option("--ridged/--no-ridged", is_flag=True, default=False)
@click.option("--wavelet/--no-wavelet", is_flag=True, default=False)
@click.option("--displacement", type=float, default=0.0)
@click.argument("name", required=False)
def gaussian(freq, width, height, channels, ridged, wavelet, displacement, name):
    with tf.Session().as_default():
        tensor = generators.gaussian(freq, width, height, channels, ridged=ridged, wavelet=wavelet, displacement=displacement)

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
@click.argument("name", required=False)
def multires(freq, width, height, channels, octaves, ridged, wavelet, displacement, name):
    with tf.Session().as_default():
        tensor = generators.multires(freq, width, height, channels, octaves, ridged=ridged, wavelet=wavelet, displacement=displacement)

        _save(tensor, name or "multires")
