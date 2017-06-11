import click
import tensorflow as tf

from noisemaker.util import save

import noisemaker.effects as effects
import noisemaker.generators as generators
import noisemaker.recipes as recipes


CLICK_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 160}


def option(*param_decls, **attrs):
    """ Add a Click option. """

    def decorator(f):
        if "default" in attrs:
            attrs["help"] += "  [default: {0}]".format(attrs["default"])

        return click.option(*param_decls, **attrs)(f)

    return decorator


def freq_option(**attrs):
    attrs.setdefault("help", "Heightwise noise frequency")
    attrs.setdefault("type", int)
    attrs.setdefault("default", 3)

    return option("--freq", **attrs)


def width_option(**attrs):
    attrs.setdefault("help", "Output width, in pixels")
    attrs.setdefault("type", int)
    attrs.setdefault("default", 1024)
    attrs.setdefault("required", True)

    return option("--width", **attrs)


def height_option(**attrs):
    attrs.setdefault("help", "Output height, in pixels")
    attrs.setdefault("type", int)
    attrs.setdefault("default", 1024)
    attrs.setdefault("required", True)

    return option("--height", **attrs)


def channels_option(**attrs):
    attrs.setdefault("help", "Color channel count (1=gray, 2=gray+alpha, 3=RGB, 4=RGB+alpha)")
    attrs.setdefault("type", int)
    attrs.setdefault("default", 3)

    return option("--channels", **attrs)


def ridges_option(**attrs):
    attrs.setdefault("help", "\"Crease\" at midpoint values: abs(noise * 2 - 1)")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--ridges", **attrs)


def wavelet_option(**attrs):
    attrs.setdefault("help", "Wavelets: What are they even?")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--wavelet", **attrs)


def refract_option(**attrs):
    attrs.setdefault("help", "Domain warping: Self-displacement range (1.0 = entire image)")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.0)

    return option("--refract", **attrs)


def reindex_option(**attrs):
    attrs.setdefault("help", "Color re-indexing range (1.0 = entire image)")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.0)

    return option("--reindex", **attrs)


def clut_option(**attrs):
    attrs.setdefault("help", "Color lookup table (path to PNG or JPEG image)")
    attrs.setdefault("type", str)

    return option("--clut", **attrs)


def clut_range_option(**attrs):
    attrs.setdefault("help", "CLUT maximum pixel gather distance (1.0 = entire image)")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.5)

    return option("--clut-range", **attrs)


def clut_horizontal_option(**attrs):
    attrs.setdefault("help", "Preserve CLUT vertical axis")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--clut-horizontal", **attrs)


def worms_option(**attrs):
    attrs.setdefault("help", "Iterative \"worm\" field flow paths")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--worms", **attrs)


def worms_behavior_option(**attrs):
    attrs.setdefault("help", "0=Obedient, 1=Crosshatch, 2=Unruly, 3=Chaotic")
    attrs.setdefault("type", int)
    attrs.setdefault("default", 0)

    return option("--worms-behavior", **attrs)


def worms_density_option(**attrs):
    attrs.setdefault("help", "Worm density multiplier (larger is more costly)")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 4.0)

    return option("--worms-density", **attrs)


def worms_duration_option(**attrs):
    attrs.setdefault("help", "Worm iteration multiplier (larger is more costly)")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 4.0)

    return option("--worms-duration", **attrs)


def worms_stride_option(**attrs):
    attrs.setdefault("help", "Mean worm pixel displacement per iteration")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 1.0)

    return option("--worms-stride", **attrs)


def worms_stride_deviation_option(**attrs):
    attrs.setdefault("help", "Per-worm random stride variance")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.0)

    return option("--worms-stride-deviation", **attrs)


def worms_bg_option(**attrs):
    attrs.setdefault("help", "Worms background color brightness")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.5)

    return option("--worms-bg", **attrs)


def worms_kink_option(**attrs):
    attrs.setdefault("help", "Worm rotation range (1.0 = 360 degrees)")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 1.0)

    return option("--worms-kink", **attrs)


def wormhole_option(**attrs):
    attrs.setdefault("help", "Domain warping: Non-iterative per-pixel field flow")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--wormhole", **attrs)


def wormhole_stride_option(**attrs):
    attrs.setdefault("help", "Max per-pixel displacement range (1.0 = entire image)")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 1.0)

    return option("--wormhole-stride", **attrs)


def wormhole_kink_option(**attrs):
    attrs.setdefault("help", "Per-pixel rotation range (1.0 = 360 degrees)")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 1.0)

    return option("--wormhole-kink", **attrs)


def voronoi_option(**attrs):
    attrs.setdefault("help", "Voronoi cells (Worley noise)")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--voronoi", **attrs)


def voronoi_density_option(**attrs):
    attrs.setdefault("help", "Cell count multiplier (1.0 = min(height, width); larger is more costly)")
    attrs.setdefault("type", float)
    attrs.setdefault("default", .1)

    return option("--voronoi-density", **attrs)


def voronoi_func_option(**attrs):
    attrs.setdefault("help", "Voronoi distance function (0=Euclidean, 1=Manhattan, 2=Chebychev)")
    attrs.setdefault("type", int)
    attrs.setdefault("default", 0)

    return option("--voronoi-func", **attrs)


def voronoi_nth_option(**attrs):
    attrs.setdefault("help", "Plot Nth nearest, or -Nth farthest")
    attrs.setdefault("type", int)
    attrs.setdefault("default", 0)

    return option("--voronoi-nth", **attrs)


def sobel_option(**attrs):
    attrs.setdefault("help", "Apply Sobel operator")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--sobel", **attrs)


def sobel_func_option(**attrs):
    attrs.setdefault("help", "Sobel distance function (0=Euclidean, 1=Manhattan, 2=Chebychev)")
    attrs.setdefault("type", int)
    attrs.setdefault("default", 0)

    return option("--sobel-func", **attrs)


def normals_option(**attrs):
    attrs.setdefault("help", "Generate a tangent-space normal map")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--normals", **attrs)


def deriv_option(**attrs):
    attrs.setdefault("help", "Calculate derivative (rate of change)")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--deriv", **attrs)


def deriv_func_option(**attrs):
    attrs.setdefault("help", "Derivative distance function (0=Euclidean, 1=Manhattan, 2=Chebychev)")
    attrs.setdefault("type", int)
    attrs.setdefault("default", 0)

    return option("--deriv-func", **attrs)


def distrib_option(**attrs):
    attrs.setdefault("help", "Random distribution (0=normal, 1=uniform, 2=exponential, 3=laplace, 4=lognormal)")
    attrs.setdefault("type", int)
    attrs.setdefault("default", 0)

    return option("--distrib", **attrs)


def interp_option(**attrs):
    attrs.setdefault("help", "Interpolation type (0=constant, 1=linear, 2=cosine, 3=bicubic)")
    attrs.setdefault("type", int)
    attrs.setdefault("default", 3)

    return option("--interp", **attrs)


def posterize_option(**attrs):
    attrs.setdefault("help", "Posterize levels (per channel)")
    attrs.setdefault("type", int)
    attrs.setdefault("default", 0)

    return option("--posterize", **attrs)


def glitch_option(**attrs):
    attrs.setdefault("help", "Glitch effect")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--glitch", **attrs)


def vhs_option(**attrs):
    attrs.setdefault("help", "VHS tracking effect")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--vhs", **attrs)


def crt_option(**attrs):
    attrs.setdefault("help", "CRT scanline effect")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--crt", **attrs)


def scan_error_option(**attrs):
    attrs.setdefault("help", "Analog scanline error effect")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--scan-error", **attrs)


def snow_option(**attrs):
    attrs.setdefault("help", "Analog broadcast snow (0.0=off, 1.0=saturated)")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.0)

    return option("--snow", **attrs)


def dither_option(**attrs):
    attrs.setdefault("help", "Per-pixel brightness jitter")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.0)

    return option("--dither", **attrs)


def name_option(**attrs):
    attrs.setdefault("help", "Base filename for image output")
    attrs.setdefault("type", str)
    attrs.setdefault("default", "noise.png")

    return option("--name", **attrs)


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
        """, context_settings=CLICK_CONTEXT_SETTINGS)
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
@freq_option()
@width_option()
@height_option()
@channels_option()
@ridges_option()
@interp_option()
@distrib_option()
@refract_option()
@reindex_option()
@clut_option()
@clut_range_option()
@clut_horizontal_option()
@voronoi_option()
@voronoi_density_option()
@voronoi_func_option()
@voronoi_nth_option()
@wormhole_option()
@wormhole_stride_option()
@wormhole_kink_option()
@worms_option()
@worms_behavior_option()
@worms_density_option()
@worms_duration_option()
@worms_stride_option()
@worms_stride_deviation_option()
@worms_kink_option()
@worms_bg_option()
@sobel_option()
@sobel_func_option()
@normals_option()
@deriv_option()
@deriv_func_option()
@posterize_option()
@glitch_option()
@vhs_option()
@crt_option()
@scan_error_option()
@snow_option()
@dither_option()
@wavelet_option()
@name_option()
@click.pass_context
def basic(ctx, freq, width, height, channels, ridges, wavelet, refract, reindex, clut, clut_horizontal, clut_range,
          worms, worms_behavior, worms_density, worms_duration, worms_stride, worms_stride_deviation, worms_bg, worms_kink, wormhole, wormhole_kink, wormhole_stride,
          voronoi, voronoi_density, voronoi_func, voronoi_nth, sobel, sobel_Func, normals, deriv, deriv_func, interp, distrib, posterize,
          glitch, vhs, crt, scan_error, snow, dither, name):

    with tf.Session().as_default():
        shape = [height, width, channels]

        tensor = generators.basic(freq, shape, ridges=ridges, wavelet=wavelet,
                                  refract_range=refract, reindex_range=reindex, clut=clut, clut_horizontal=clut_horizontal, clut_range=clut_range,
                                  with_worms=worms, worms_behavior=worms_behavior, worms_density=worms_density, worms_duration=worms_duration,
                                  worms_stride=worms_stride, worms_stride_deviation=worms_stride_deviation, worms_bg=worms_bg, worms_kink=worms_kink,
                                  with_wormhole=wormhole, wormhole_kink=wormhole_kink, wormhole_stride=wormhole_stride,
                                  with_voronoi=voronoi, voronoi_density=voronoi_density, voronoi_func=voronoi_func, voronoi_nth=voronoi_nth,
                                  with_sobel=sobel, sobel_func=sobel_func, with_normal_map=normals, deriv=deriv, deriv_func=deriv_func, spline_order=interp, distrib=distrib,
                                  posterize_levels=posterize
                                  )

        tensor = _apply_conv_kernels(tensor, shape, ctx.obj)

        tensor = recipes.post_process(tensor, shape, with_glitch=glitch, with_vhs=vhs, with_crt=crt, with_scan_error=scan_error, with_snow=snow, with_dither=dither)

        save(tensor, name)


@main.command(help="Multiple layers (octaves). For each octave: freq increases, amplitude decreases.")
@freq_option()
@width_option()
@height_option()
@channels_option()
@ridges_option()
@click.option("--octaves", type=int, default=3, help="Octave count: Number of multi-res layers")
@click.option("--layer-refract", type=float, default=0.0, help="Per-octave self-distortion gradient")
@click.option("--layer-reindex", type=float, default=0.0, help="Per-octave self-reindexing gradient")
@interp_option()
@distrib_option()
@refract_option()
@reindex_option()
@clut_option()
@clut_range_option()
@clut_horizontal_option()
@voronoi_option()
@voronoi_density_option()
@voronoi_func_option()
@voronoi_nth_option()
@wormhole_option()
@wormhole_stride_option()
@wormhole_kink_option()
@worms_option()
@worms_behavior_option()
@worms_density_option()
@worms_duration_option()
@worms_stride_option()
@worms_stride_deviation_option()
@worms_kink_option()
@worms_bg_option()
@sobel_option()
@sobel_func_option()
@normals_option()
@deriv_option()
@deriv_func_option()
@posterize_option()
@glitch_option()
@vhs_option()
@crt_option()
@scan_error_option()
@snow_option()
@dither_option()
@wavelet_option()
@name_option()
@click.pass_context
def multires(ctx, freq, width, height, channels, octaves, ridges, wavelet, refract, layer_refract, reindex, layer_reindex,
             clut, clut_horizontal, clut_range, worms, worms_behavior, worms_density, worms_duration, worms_stride, worms_stride_deviation,
             worms_bg, worms_kink, wormhole, wormhole_kink, wormhole_stride, sobel, sobel_func, normals, deriv, deriv_func, interp, distrib, posterize,
             voronoi, voronoi_density, voronoi_func, voronoi_nth, glitch, vhs, crt, scan_error, snow, dither, name):

    with tf.Session().as_default():
        shape = [height, width, channels]

        tensor = generators.multires(freq, shape, octaves, ridges=ridges, wavelet=wavelet,
                                     refract_range=refract, layer_refract_range=layer_refract,
                                     reindex_range=reindex, layer_reindex_range=layer_reindex,
                                     clut=clut, clut_horizontal=clut_horizontal, clut_range=clut_range,
                                     with_worms=worms, worms_behavior=worms_behavior, worms_density=worms_density, worms_duration=worms_duration,
                                     worms_stride=worms_stride, worms_stride_deviation=worms_stride_deviation, worms_bg=worms_bg, worms_kink=worms_kink,
                                     with_wormhole=wormhole, wormhole_kink=wormhole_kink, wormhole_stride=wormhole_stride,
                                     with_voronoi=voronoi, voronoi_density=voronoi_density, voronoi_func=voronoi_func, voronoi_nth=voronoi_nth,
                                     with_sobel=sobel, sobel_func=sobel_func, with_normal_map=normals, deriv=deriv, deriv_func=deriv_func, spline_order=interp,
                                     distrib=distrib, posterize_levels=posterize,
                                     )

        tensor = _apply_conv_kernels(tensor, shape, ctx.obj)

        tensor = recipes.post_process(tensor, shape, with_glitch=glitch, with_vhs=vhs, with_crt=crt, with_scan_error=scan_error, with_snow=snow, with_dither=dither)

        save(tensor, name)