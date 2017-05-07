import click
import tensorflow as tf

import noisemaker.effects as effects
import noisemaker.generators as generators
import noisemaker.recipes as recipes


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


def _apply_conv_kernels(tensor, shape, args):
    """
    Apply convolution kernels from given CLI options.

    :param Tensor tensor:
    :param dict args:
    :return: Tensor
    """

    for kernel in effects.ConvKernel:
        if args.get(kernel.name):
            tensor =  effects.convolve(kernel, tensor, shape)

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


@main.command(help="Scaled random values.")
@click.option("--freq", type=int, default=4, help="Heightwise noise frequency")
@click.option("--width", type=int, default=1024, help="Image output width")
@click.option("--height", type=int, default=1024, help="Image output height")
@click.option("--channels", type=int, default=3, help="Channel count. 1=Gray, 3=RGB, others may not work.")
@click.option("--ridged/--no-ridged", is_flag=True, default=False, help="\"Crease\" in the middle. (1 - abs(n * 2 - 1))")
@click.option("--wavelet/--no-wavelet", is_flag=True, default=False, help="Maybe not wavelets this time?")
@click.option("--refract", type=float, default=0.0, help="Self-distortion gradient.")
@click.option("--reindex", type=float, default=0.0, help="Self-reindexing gradient.")
@click.option("--clut", type=str, default=0.0, help="Color lookup table (PNG or JPG)")
@click.option("--clut-horizontal", is_flag=True, default=False, help="Preserve clut Y axis")
@click.option("--clut-range", type=float, default=.5, help="Gather distance for clut.")
@click.option("--worms", is_flag=True, default=False, help="Plot field flow path of \"worms\" through noise.")
@click.option("--worm-behavior", type=int, default=0, help="0=Obedient, 1=Crosshatch, 2=Unruly, 3=Chaotic")
@click.option("--worm-density", type=float, default=4.0, help="Worm density multiplier (larger is slower)")
@click.option("--worm-duration", type=float, default=4.0, help="Worm iteration multiplier (larger is slower)")
@click.option("--worm-stride", type=float, default=1.0, help="Mean travel distance per iteration")
@click.option("--worm-stride-deviation", type=float, default=.05, help="Travel distance deviation per worm")
@click.option("--worm-bg", type=float, default=.5, help="Worms background color brightness")
@click.option("--worm-kink", type=float, default=1.0, help="Worms twistiness")
@click.option("--wormhole", is_flag=True, default=False, help="\"Scattered\" values. Not worms. Non-iterative (fast)")
@click.option("--wormhole-kink", type=float, default=2.5, help="Wormhole kinkiness")
@click.option("--wormhole-stride", type=float, default=.1, help="Wormhole thickness range")
@click.option("--voronoi", is_flag=True, default=False, help="Voronoi cells")
@click.option("--voronoi-density", type=float, default=.1, help="Voronoi cell count multiplier")
@click.option("--sobel", is_flag=True, default=False, help="Apply Sobel operator.")
@click.option("--normals", is_flag=True, default=False, help="Generate a tangent-space normal map.")
@click.option("--deriv", is_flag=True, default=False, help="Derivative noise.")
@click.option("--distrib", type=int, default=0, help="Random distribution type. 0=Normal, 1=Uniform, 2=Exponential.")
@click.option("--spline-order", type=int, default=3, help="Spline point count. 0=Constant, 1=Linear, 2=Cosine, 3=Bicubic")
@click.option("--seed", type=int, required=False, help="Random seed for reproducible output. Ineffective with exponential.")
@click.option("--glitch", is_flag=True, default=False, help="Glitch effect")
@click.option("--vhs", is_flag=True, default=False, help="VHS effect")
@click.option("--crt", is_flag=True, default=False, help="CRT effect")
@click.option("--name", default="basic", help="Base filename for image output")
@click.pass_context
def basic(ctx, freq, width, height, channels, ridged, wavelet, refract, reindex, clut, clut_horizontal, clut_range,
          worms, worm_behavior, worm_density, worm_duration, worm_stride, worm_stride_deviation, worm_bg, worm_kink, wormhole, wormhole_kink, wormhole_stride,
          voronoi, voronoi_density, sobel, normals, deriv, spline_order, distrib, seed, glitch, vhs, crt, name):

    with tf.Session().as_default():
        shape = [height, width, channels]

        tensor = generators.basic(freq, shape, ridged=ridged, wavelet=wavelet,
                                  refract_range=refract, reindex_range=reindex, clut=clut, clut_horizontal=clut_horizontal, clut_range=clut_range,
                                  with_worms=worms, worm_behavior=worm_behavior, worm_density=worm_density, worm_duration=worm_duration,
                                  worm_stride=worm_stride, worm_stride_deviation=worm_stride_deviation, worm_bg=worm_bg, worm_kink=worm_kink,
                                  with_wormhole=wormhole, wormhole_kink=wormhole_kink, wormhole_stride=wormhole_stride,
                                  with_voronoi=voronoi, voronoi_density=voronoi_density,
                                  with_sobel=sobel, with_normal_map=normals, deriv=deriv, spline_order=spline_order, distrib=distrib, seed=seed,
                                  )

        tensor = _apply_conv_kernels(tensor, shape, ctx.obj)

        tensor = recipes.post_process(tensor, shape, with_glitch=glitch, with_vhs=vhs, with_crt=crt)

        _save(tensor, name)


@main.command(help="Multiple layers (octaves). For each octave: freq increases, amplitude decreases.")
@click.option("--freq", type=int, default=4, help="Heightwise bottom layer frequency")
@click.option("--width", type=int, default=1024, help="Image output width")
@click.option("--height", type=int, default=1024, help="Image output height")
@click.option("--channels", type=int, default=3, help="Channel count. 1=Gray, 3=RGB, others may not work.")
@click.option("--ridged/--no-ridged", is_flag=True, default=True, help="\"Crease\" in the middle. (1 - abs(n * 2 - 1))")
@click.option("--wavelet/--no-wavelet", is_flag=True, default=False, help="Maybe not wavelets this time?")
@click.option("--refract", type=float, default=0.0, help="Self-distortion gradient.")
@click.option("--layer-refract", type=float, default=0.0, help="Per-octave self-distortion gradient.")
@click.option("--reindex", type=float, default=0.0, help="Self-reindexing gradient.")
@click.option("--layer-reindex", type=float, default=0.0, help="Per-octave self-reindexing gradient.")
@click.option("--clut", type=str, default=0.0, help="Color lookup table (PNG or JPG)")
@click.option("--clut-horizontal", is_flag=True, default=False, help="Preserve clut Y axis")
@click.option("--clut-range", type=float, default=.5, help="Gather distance for clut.")
@click.option("--worms", is_flag=True, default=False, help="Plot field flow path of \"worms\" through noise.")
@click.option("--worm-behavior", type=int, default=0, help="0=Obedient, 1=Crosshatch, 2=Unruly, 3=Chaotic")
@click.option("--worm-density", type=float, default=4.0, help="Worm density multiplier (larger is slower)")
@click.option("--worm-duration", type=float, default=4.0, help="Worm iteration multiplier (larger is slower)")
@click.option("--worm-stride", type=float, default=1.0, help="Mean travel distance per iteration")
@click.option("--worm-stride-deviation", type=float, default=.05, help="Travel distance deviation per worm")
@click.option("--worm-bg", type=float, default=.5, help="Worms background color brightness")
@click.option("--worm-kink", type=float, default=1.0, help="Worms twistiness")
@click.option("--wormhole", is_flag=True, default=False, help="\"Scattered\" values. Not worms. Non-iterative (fast)")
@click.option("--wormhole-kink", type=float, default=2.5, help="Wormhole kinkiness")
@click.option("--wormhole-stride", type=float, default=.1, help="Wormhole thickness range")
@click.option("--voronoi", is_flag=True, default=False, help="Voronoi cells")
@click.option("--voronoi-density", type=float, default=.1, help="Voronoi cell count multiplier")
@click.option("--sobel", is_flag=True, default=False, help="Apply Sobel operator.")
@click.option("--normals", is_flag=True, default=False, help="Generate a tangent-space normal map.")
@click.option("--deriv", is_flag=True, default=False, help="Derivative noise.")
@click.option("--distrib", type=int, default=0, help="Random distribution type. 0=Normal, 1=Uniform, 2=Exponential.")
@click.option("--spline-order", type=int, default=3, help="Spline point count. 0=Constant, 1=Linear, 2=Cosine, 3=Bicubic")
@click.option("--seed", type=int, required=False, help="Random seed for reproducible output. Ineffective with exponential.")
@click.option("--glitch", is_flag=True, default=False, help="Glitch effect")
@click.option("--vhs", is_flag=True, default=False, help="VHS effect")
@click.option("--crt", is_flag=True, default=False, help="CRT effect")
@click.option("--octaves", type=int, default=3, help="Octave count. Number of multi-res layers. Typically 1-8")
@click.option("--name", default="multires", help="Base filename for image output")
@click.pass_context
def multires(ctx, freq, width, height, channels, octaves, ridged, wavelet, refract, layer_refract, reindex, layer_reindex,
             clut, clut_horizontal, clut_range, worms, worm_behavior, worm_density, worm_duration, worm_stride, worm_stride_deviation,
             worm_bg, worm_kink, wormhole, wormhole_kink, wormhole_stride, sobel, normals, deriv, spline_order, distrib, seed,
             voronoi, voronoi_density, glitch, vhs, crt, name):

    with tf.Session().as_default():
        shape = [height, width, channels]

        tensor = generators.multires(freq, shape, octaves, ridged=ridged, wavelet=wavelet,
                                     refract_range=refract, layer_refract_range=layer_refract,
                                     reindex_range=reindex, layer_reindex_range=layer_reindex,
                                     clut=clut, clut_horizontal=clut_horizontal, clut_range=clut_range,
                                     with_worms=worms, worm_behavior=worm_behavior, worm_density=worm_density, worm_duration=worm_duration,
                                     worm_stride=worm_stride, worm_stride_deviation=worm_stride_deviation, worm_bg=worm_bg, worm_kink=worm_kink,
                                     with_wormhole=wormhole, wormhole_kink=wormhole_kink, wormhole_stride=wormhole_stride,
                                     with_voronoi=voronoi, voronoi_density=voronoi_density,
                                     with_sobel=sobel, with_normal_map=normals, deriv=deriv, spline_order=spline_order, distrib=distrib, seed=seed,
                                     )

        tensor = _apply_conv_kernels(tensor, shape, ctx.obj)

        tensor = recipes.post_process(tensor, shape, with_glitch=glitch, with_vhs=vhs, with_crt=crt)

        _save(tensor, name)