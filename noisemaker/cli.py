import os

import click
import tensorflow as tf

from noisemaker.util import save

import noisemaker.effects as effects
import noisemaker.generators as generators
import noisemaker.recipes as recipes

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

CLICK_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 160}

# Boilerplate help strings

ALPHA_BLENDING_HINT = "alpha blending amount (0.0 = 0%, 1.0 = 100%)"

DISTANCE_HINT = "(1=Euclidean, 2=Manhattan, 3=Chebyshev)"

ENTIRE_IMAGE_HINT = "(1.0 = height/width of entire image)"

FREQ_HINT = "(must be >= 2)"

INTERPOLATION_HINT = "(0=constant, 1=linear, 2=cosine, 3=bicubic)"

NEAREST_NEIGHBOR_HINT = "(1.0 = as far as nearest neighbor)"


def validate_at_least_one(allow_none=False):
    """
    """

    def validate(ctx, param, value):
        if value <= 1 or (value is None and not allow_none):
            raise click.BadParameter("invalid choice: {0}. (choose a value greater than 1)".format(value))

    return validate


def validate_enum(cls):
    """
    """

    def validate(ctx, param, value):
        if value is not None and value not in [m.value for m in cls]:
            raise click.BadParameter("invalid choice: {0}. (choose from {1})".format(value, ", ".join(["{0} ({1})".format(m.value, m.name) for m in cls])))

    return validate


def bool_option(attr, **attrs):
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option(attr, **attrs)


def float_option(attr, **attrs):
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.0)

    return option(attr, **attrs)


def int_option(attr, **attrs):
    attrs.setdefault("type", int)
    attrs.setdefault("default", 0)

    return option(attr, **attrs)


def str_option(attr, **attrs):
    attrs.setdefault("type", str)
    attrs.setdefault("default", "")

    return option(attr, **attrs)


def option(*param_decls, **attrs):
    """ Add a Click option. """

    def decorator(f):
        if attrs.get("default") not in (None, False):
            attrs["help"] += "  [default: {0}]".format(attrs["default"])

        return click.option(*param_decls, **attrs)(f)

    return decorator


def freq_option(**attrs):
    attrs.setdefault("help", "Minimum noise frequency {0}".format(FREQ_HINT))

    return int_option("--freq", default=3, callback=validate_at_least_one(), **attrs)


def width_option(**attrs):
    attrs.setdefault("help", "Output width, in pixels")

    return int_option("--width", default=1024, **attrs)


def height_option(**attrs):
    attrs.setdefault("help", "Output height, in pixels")

    return int_option("--height", default=1024, **attrs)


def channels_option(**attrs):
    attrs.setdefault("help", "Color channel count (1=gray, 2=gray+alpha, 3=HSV/RGB, 4=RGB+alpha)")

    return option("--channels", type=click.IntRange(1, 4), default=3, **attrs)


def octaves_option(**attrs):
    attrs.setdefault("help", "Octave count: Number of multi-res layers")

    return option("--octaves", type=click.IntRange(1, 10), default=1, **attrs)


def ridges_option(**attrs):
    attrs.setdefault("help", "\"Crease\" at midpoint values: abs(noise * 2 - 1)")

    return bool_option("--ridges", **attrs)


def distrib_option(**attrs):
    attrs.setdefault("help", "Value distribution")

    return option("--distrib", type=click.Choice([m.name for m in generators.ValueDistribution]), default="normal", **attrs)


def corners_option(**attrs):
    attrs.setdefault("help", "Value distribution: Pin pixels to corners, instead of image center.")

    return bool_option("--corners", **attrs)


def mask_option(**attrs):
    attrs.setdefault("help", "Value distribution: Hot pixel mask")

    return option("--mask", type=click.Choice([m.name for m in generators.ValueMask]), **attrs)


def interp_option(**attrs):
    attrs.setdefault("help", "Interpolation type {0}".format(INTERPOLATION_HINT))

    return int_option("--interp", callback=validate_enum(effects.InterpolationType), default=3, **attrs)


def sin_option(**attrs):
    attrs.setdefault("help", "Apply sin function to noise basis")

    return float_option("--sin", **attrs)


def wavelet_option(**attrs):
    attrs.setdefault("help", "Wavelets: What are they even?")

    return bool_option("--wavelet", **attrs)


def lattice_drift_option(**attrs):
    attrs.setdefault("help", "Domain warping: Lattice deform range {0}".format(NEAREST_NEIGHBOR_HINT))

    return float_option("--lattice-drift", **attrs)


def vortex_option(**attrs):
    attrs.setdefault("help", "Vortex tiling amount")

    return float_option("--vortex", **attrs)


def warp_option(**attrs):
    attrs.setdefault("help", "Octave Warp: Orthogonal displacement range {0}".format(ENTIRE_IMAGE_HINT))

    return float_option("--warp", **attrs)


def warp_octaves_option(**attrs):
    attrs.setdefault("help", "Octave Warp: Octave count for --warp")

    return option("--warp-octaves", type=click.IntRange(1, 10), default=3, **attrs)


def warp_interp_option(**attrs):
    attrs.setdefault("help", "Octave Warp: Interpolation type {0}".format(INTERPOLATION_HINT))

    return int_option("--warp-interp", callback=validate_enum(effects.InterpolationType), **attrs)


def warp_freq_option(**attrs):
    attrs.setdefault("help", "Octave Warp: Override --freq for warp frequency {0}".format(FREQ_HINT))

    return int_option("--warp-freq", callback=validate_at_least_one(allow_none=True), **attrs)


def post_reflect_option(**attrs):
    attrs.setdefault("help", "Domain warping: Post-reduce derivative-based displacement range {0}".format(ENTIRE_IMAGE_HINT))

    return float_option("--post-reflect", **attrs)


def post_refract_option(**attrs):
    attrs.setdefault("help", "Domain warping: Post-reduce self-displacement range {0}".format(ENTIRE_IMAGE_HINT))

    return float_option("--post-refract", **attrs)


def reflect_option(**attrs):
    attrs.setdefault("help", "Domain warping: Per-octave derivative-based displacement range {0}".format(ENTIRE_IMAGE_HINT))

    return float_option("--reflect", **attrs)


def refract_option(**attrs):
    attrs.setdefault("help", "Domain warping: Per-octave self-displacement range {0}".format(ENTIRE_IMAGE_HINT))

    return float_option("--refract", **attrs)


def reindex_option(**attrs):
    attrs.setdefault("help", "Color re-indexing range {0}".format(ENTIRE_IMAGE_HINT))

    return float_option("--reindex", **attrs)


def clut_option(**attrs):
    attrs.setdefault("help", "Color lookup table (path to PNG or JPEG image)")

    return str_option("--clut", **attrs)


def clut_range_option(**attrs):
    attrs.setdefault("help", "CLUT: Maximum pixel gather distance {0}".format(ENTIRE_IMAGE_HINT))

    return float_option("--clut-range", default=0.5, **attrs)


def clut_horizontal_option(**attrs):
    attrs.setdefault("help", "CLUT: Preserve vertical axis")

    return bool_option("--clut-horizontal", **attrs)


def worms_option(**attrs):
    attrs.setdefault("help", "Iterative \"worm\" field flow (1=Obedient, 2=Crosshatch, 3=Unruly, 4=Chaotic)")

    return int_option("--worms", callback=validate_enum(effects.WormBehavior), **attrs)


def worms_density_option(**attrs):
    attrs.setdefault("help", "Worms: Density multiplier (larger is more costly)")

    return float_option("--worms-density", default=4.0, **attrs)


def worms_duration_option(**attrs):
    attrs.setdefault("help", "Worms: Iteration multiplier (larger is more costly)")

    return float_option("--worms-duration", default=4.0, **attrs)


def worms_stride_option(**attrs):
    attrs.setdefault("help", "Worms: Mean pixel displacement per iteration")

    return float_option("--worms-stride", default=1.0, **attrs)


def worms_stride_deviation_option(**attrs):
    attrs.setdefault("help", "Worms: Per-worm random stride variance")

    return float_option("--worms-stride-deviation", **attrs)


def worms_bg_option(**attrs):
    attrs.setdefault("help", "Worms: Background color brightness")

    return float_option("--worms-bg", **attrs)


def worms_kink_option(**attrs):
    attrs.setdefault("help", "Worms: Rotation range (1.0 = 360 degrees)")

    return float_option("--worms-kink", default=1.0, **attrs)


def wormhole_option(**attrs):
    attrs.setdefault("help", "Non-iterative per-pixel field flow")

    return bool_option("--wormhole", **attrs)


def wormhole_stride_option(**attrs):
    attrs.setdefault("help", "Wormhole: Max per-pixel displacement range {0}".format(ENTIRE_IMAGE_HINT))

    return float_option("--wormhole-stride", default=0.1, **attrs)


def wormhole_kink_option(**attrs):
    attrs.setdefault("help", "Wormhole: Per-pixel rotation range (1.0 = 360 degrees)")

    return float_option("--wormhole-kink", default=1.0, **attrs)


def erosion_worms_option(**attrs):
    attrs.setdefault("help", "Experimental erosion worms (Does not use worms settings)")

    return bool_option("--erosion-worms", **attrs)


def dla_option(**attrs):
    attrs.setdefault("help", "Diffusion-limited aggregation (DLA) {0}".format(ALPHA_BLENDING_HINT))

    return float_option("--dla", **attrs)


def dla_padding_option(**attrs):
    attrs.setdefault("help", "DLA: Pixel padding (smaller is slower)")

    return int_option("--dla-padding", default=2, **attrs)


def voronoi_option(**attrs):
    attrs.setdefault("help", "Generate a Voronoi diagram (0=Off, 1=Range, 2=Color Range, 3=Indexed, 4=Color Map, 5=Blended, 6=Flow, 7=Collage)")

    return int_option("--voronoi", callback=validate_enum(effects.VoronoiDiagramType), **attrs)


def voronoi_func_option(**attrs):
    attrs.setdefault("help", "Voronoi: Distance function {0}".format(DISTANCE_HINT))

    return int_option("--voronoi-func", callback=validate_enum(effects.DistanceFunction), default=1, **attrs)


def voronoi_nth_option(**attrs):
    attrs.setdefault("help", "Voronoi: Plot Nth nearest, or -Nth farthest")

    return int_option("--voronoi-nth", **attrs)


def voronoi_alpha_option(**attrs):
    attrs.setdefault("help", "Voronoi: Basis {0}".format(ALPHA_BLENDING_HINT))

    return float_option("--voronoi-alpha", default=1.0, **attrs)


def voronoi_refract_option(**attrs):
    attrs.setdefault("help", "Voronoi: Domain warp input tensor {0}".format(ENTIRE_IMAGE_HINT))

    return float_option("--voronoi-refract", **attrs)


def voronoi_inverse_option(**attrs):
    attrs.setdefault("help", "Voronoi: Inverse range")

    return bool_option("--voronoi-inverse", **attrs)


def point_freq_option(**attrs):
    attrs.setdefault("help", "Voronoi/DLA: Approximate lengthwise point cloud frequency (freq * freq = count)")

    return option("--point-freq", type=click.IntRange(1, 10), default=3, **attrs)


def point_distrib_option(**attrs):
    attrs.setdefault("help", "Voronoi/DLA: Point cloud distribution")

    return option("--point-distrib", type=click.Choice([m.name for m in effects.PointDistribution]), default="random", **attrs)


def point_corners_option(**attrs):
    attrs.setdefault("help", "Voronoi/DLA: Pin diagram to corners, instead of image center.")

    return bool_option("--point-corners", **attrs)


def point_generations_option(**attrs):
    attrs.setdefault("help", "Voronoi/DLA: Penrose-ish generations. When using, keep this and freq below ~3 or you will run OOM easily.")

    return option("--point-generations", type=click.IntRange(1, 3), default=1, **attrs)


def point_drift_option(**attrs):
    attrs.setdefault("help", "Voronoi/DLA: Point drift range {0}".format(NEAREST_NEIGHBOR_HINT))

    return float_option("--point-drift", **attrs)


def sobel_option(**attrs):
    attrs.setdefault("help", "Post-processing: Apply Sobel operator {0}".format(DISTANCE_HINT))

    return int_option("--sobel", callback=validate_enum(effects.DistanceFunction), **attrs)


def outline_option(**attrs):
    attrs.setdefault("help", "Post-processing: Apply Sobel operator, and multiply {0}".format(DISTANCE_HINT))

    return int_option("--outline", callback=validate_enum(effects.DistanceFunction), **attrs)


def normals_option(**attrs):
    attrs.setdefault("help", "Post-processing: Generate a tangent-space normal map")

    return bool_option("--normals", **attrs)


def post_deriv_option(**attrs):
    attrs.setdefault("help", "Derivatives: Extract post-reduce rate of change {0}".format(DISTANCE_HINT))

    return int_option("--post-deriv", callback=validate_enum(effects.DistanceFunction), **attrs)


def deriv_option(**attrs):
    attrs.setdefault("help", "Derivatives: Extract per-octave rate of change {0}".format(DISTANCE_HINT))

    return int_option("--deriv", callback=validate_enum(effects.DistanceFunction), **attrs)


def deriv_alpha_option(**attrs):
    attrs.setdefault("help", "Derivatives: Per-octave {0}".format(ALPHA_BLENDING_HINT))

    return float_option("--deriv-alpha", default=1.0, **attrs)


def posterize_option(**attrs):
    attrs.setdefault("help", "Post-processing: Posterize levels (per channel)")

    return int_option("--posterize", **attrs)


def bloom_option(**attrs):
    attrs.setdefault("help", "Post-processing: Bloom {0}".format(ALPHA_BLENDING_HINT))

    return float_option("--bloom", **attrs)


def glitch_option(**attrs):
    attrs.setdefault("help", "Glitch effects: Bit-shit")

    return bool_option("--glitch", **attrs)


def vhs_option(**attrs):
    attrs.setdefault("help", "Glitch effects: VHS tracking")

    return bool_option("--vhs", **attrs)


def crt_option(**attrs):
    attrs.setdefault("help", "Glitch effects: CRT scanline")

    return bool_option("--crt", **attrs)


def scan_error_option(**attrs):
    attrs.setdefault("help", "Glitch effects: Analog scanline error")

    return bool_option("--scan-error", **attrs)


def snow_option(**attrs):
    attrs.setdefault("help", "Glitch effects: Analog broadcast snow (0.0=off, 1.0=saturated)")

    return float_option("--snow", **attrs)


def dither_option(**attrs):
    attrs.setdefault("help", "Glitch effects: Per-pixel brightness jitter")

    return float_option("--dither", **attrs)


def aberration_option(**attrs):
    attrs.setdefault("help", "Glitch effects: Chromatic aberration distance (e.g. .0075)")

    return float_option("--aberration", **attrs)


def emboss_option(**attrs):
    attrs.setdefault("help", "Convolution kernel: Emboss {0}".format(ALPHA_BLENDING_HINT))

    return float_option("--emboss", **attrs)


def shadow_option(**attrs):
    attrs.setdefault("help", "Convolution kernel: Shadow {0}".format(ALPHA_BLENDING_HINT))

    return float_option("--shadow", **attrs)


def edges_option(**attrs):
    attrs.setdefault("help", "Convolution kernel: Edges {0}".format(ALPHA_BLENDING_HINT))

    return float_option("--edges", **attrs)


def sharpen_option(**attrs):
    attrs.setdefault("help", "Convolution kernel: Sharpen {0}".format(ALPHA_BLENDING_HINT))

    return float_option("--sharpen", **attrs)


def unsharp_mask_option(**attrs):
    attrs.setdefault("help", "Convolution kernel: Unsharp mask {0}".format(ALPHA_BLENDING_HINT))

    return float_option("--unsharp-mask", **attrs)


def invert_option(**attrs):
    attrs.setdefault("help", "Convolution kernel: Invert {0}".format(ALPHA_BLENDING_HINT))

    return float_option("--invert", **attrs)


def rgb_option(**attrs):
    attrs.setdefault("help", "Use RGB noise basis instead of HSV")

    return bool_option("--rgb", **attrs)


def hsv_range_option(**attrs):
    attrs.setdefault("help", "HSV: Hue range (0..1+")

    return float_option("--hsv-range", default=0.25, **attrs)


def hsv_rotation_option(**attrs):
    attrs.setdefault("help", "HSV: Hue rotation (0..1)")

    return float_option("--hsv-rotation", **attrs)


def hsv_saturation_option(**attrs):
    attrs.setdefault("help", "HSV: Saturation (0..1+)")

    return float_option("--hsv-saturation", default=1.0, **attrs)


def input_dir_option(**attrs):
    attrs.setdefault("help", "Input directory containing .jpg and/or .png images, for collage functions")

    return str_option("--input-dir", **attrs)


def name_option(**attrs):
    attrs.setdefault("help", "Base filename for image output")

    return str_option("--name", default="noise.png", **attrs)


@click.command(help="""
        Noisemaker - Visual noise generator

        https://github.com/aayars/py-noisemaker
        """, context_settings=CLICK_CONTEXT_SETTINGS)
@freq_option()
@width_option()
@height_option()
@channels_option()
@octaves_option()
@ridges_option()
@deriv_option()
@deriv_alpha_option()
@post_deriv_option()
@interp_option()
@sin_option()
@distrib_option()
@corners_option()
@mask_option()
@lattice_drift_option()
@vortex_option()
@warp_option()
@warp_octaves_option()
@warp_interp_option()
@warp_freq_option()
@post_reflect_option()
@reflect_option()
@post_refract_option()
@refract_option()
@reindex_option()
@clut_option()
@clut_range_option()
@clut_horizontal_option()
@voronoi_option()
@voronoi_func_option()
@voronoi_nth_option()
@voronoi_alpha_option()
@voronoi_refract_option()
@voronoi_inverse_option()
@dla_option()
@dla_padding_option()
@point_freq_option()
@point_distrib_option()
@point_corners_option()
@point_generations_option()
@point_drift_option()
@wormhole_option()
@wormhole_stride_option()
@wormhole_kink_option()
@worms_option()
@worms_density_option()
@worms_duration_option()
@worms_stride_option()
@worms_stride_deviation_option()
@worms_kink_option()
@worms_bg_option()
@erosion_worms_option()
@sobel_option()
@outline_option()
@normals_option()
@posterize_option()
@bloom_option()
@glitch_option()
@vhs_option()
@crt_option()
@scan_error_option()
@snow_option()
@dither_option()
@aberration_option()
@emboss_option()
@shadow_option()
@edges_option()
@sharpen_option()
@unsharp_mask_option()
@invert_option()
@rgb_option()
@hsv_range_option()
@hsv_rotation_option()
@hsv_saturation_option()
@input_dir_option()
@wavelet_option()
@name_option()
@click.pass_context
def main(ctx, freq, width, height, channels, octaves, ridges, sin, wavelet, lattice_drift, vortex, warp, warp_octaves, warp_interp, warp_freq, reflect, refract, reindex,
         post_reflect, post_refract, clut, clut_horizontal, clut_range, worms, worms_density, worms_duration, worms_stride, worms_stride_deviation,
         worms_bg, worms_kink, wormhole, wormhole_kink, wormhole_stride, sobel, outline, normals, post_deriv, deriv, deriv_alpha, interp, distrib, corners, mask, posterize,
         erosion_worms, voronoi, voronoi_func, voronoi_nth, voronoi_alpha, voronoi_refract, voronoi_inverse,
         glitch, vhs, crt, scan_error, snow, dither, aberration, bloom, rgb, hsv_range, hsv_rotation, hsv_saturation, input_dir,
         dla, dla_padding, point_freq, point_distrib, point_corners, point_generations, point_drift,
         name, **convolve_kwargs):

    shape = [height, width, channels]

    tensor = generators.multires(freq, shape, octaves, ridges=ridges, sin=sin, wavelet=wavelet, lattice_drift=lattice_drift,
                                 reflect_range=reflect, refract_range=refract, reindex_range=reindex,
                                 post_reflect_range=post_reflect, post_refract_range=post_refract,
                                 clut=clut, clut_horizontal=clut_horizontal, clut_range=clut_range,
                                 with_worms=worms, worms_density=worms_density, worms_duration=worms_duration,
                                 worms_stride=worms_stride, worms_stride_deviation=worms_stride_deviation, worms_bg=worms_bg, worms_kink=worms_kink,
                                 with_wormhole=wormhole, wormhole_kink=wormhole_kink, wormhole_stride=wormhole_stride, with_erosion_worms=erosion_worms,
                                 with_voronoi=voronoi, voronoi_func=voronoi_func, voronoi_nth=voronoi_nth,
                                 voronoi_alpha=voronoi_alpha, voronoi_refract=voronoi_refract, voronoi_inverse=voronoi_inverse,
                                 with_dla=dla, dla_padding=dla_padding, point_freq=point_freq, point_distrib=point_distrib, point_center=not point_corners,
                                 point_generations=point_generations, point_drift=point_drift,
                                 with_outline=outline, with_sobel=sobel, with_normal_map=normals, post_deriv=post_deriv, deriv=deriv, deriv_alpha=deriv_alpha,
                                 spline_order=interp, distrib=distrib, corners=corners, mask=mask,
                                 warp_range=warp, warp_octaves=warp_octaves, warp_interp=warp_interp, warp_freq=warp_freq,
                                 posterize_levels=posterize, vortex_range=vortex,
                                 hsv=not rgb, hsv_range=hsv_range, hsv_rotation=hsv_rotation, hsv_saturation=hsv_saturation, input_dir=input_dir,
                                 with_aberration=aberration, with_bloom=bloom, **convolve_kwargs)

    tensor = recipes.post_process(tensor, shape, freq, with_glitch=glitch, with_vhs=vhs, with_crt=crt, with_scan_error=scan_error, with_snow=snow, with_dither=dither)

    with tf.Session().as_default():
        save(tensor, name)

    print(name)