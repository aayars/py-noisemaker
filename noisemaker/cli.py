import os

import click

import noisemaker.effects as effects
import noisemaker.generators as generators


# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


CLICK_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 160}

# Boilerplate help strings

ALPHA_BLENDING_HINT = "alpha blending amount (0.0 = 0%, 1.0 = 100%)"

DISTANCE_HINT = "(1=Euclidean, 2=Manhattan, 3=Chebyshev)"

ENTIRE_IMAGE_HINT = "(1.0 = height/width of entire image)"

FREQ_HINT = "(must be >= 2)"

INTERPOLATION_HINT = "(0=constant, 1=linear, 2=cosine, 3=bicubic)"

NEAREST_NEIGHBOR_HINT = "(1.0 = as far as nearest neighbor)"


def validate_more_than_one(allow_none=False):
    """
    """

    def validate(ctx, param, value):
        is_valid = False

        if value is None:
            is_valid = allow_none

        elif value > 1:
            is_valid = True

        if not is_valid:
            raise click.BadParameter("invalid choice: {0}. (choose a value greater than 1)".format(value))

        return value

    return validate


def validate_enum(cls):
    """
    """

    def validate(ctx, param, value):
        if value is not None and value not in [m.value for m in cls]:
            raise click.BadParameter("invalid choice: {0}. (choose from {1})".format(value, ", ".join(["{0} ({1})".format(m.value, m.name) for m in cls])))

        return value

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
    attrs.setdefault("default", None)

    return option(attr, **attrs)


def option(*param_decls, **attrs):
    """ Add a Click option. """

    def decorator(f):
        if isinstance(attrs.get("type"), click.IntRange):
            r = attrs["type"]

            attrs["help"] += "  [range: {0}-{1}]".format(r.min, r.max)

        if attrs.get("default") not in (None, False, 0):
            attrs["help"] += "  [default: {0}]".format(attrs["default"])

        return click.option(*param_decls, **attrs)(f)

    return decorator


def freq_option(**attrs):
    attrs.setdefault("help", "Minimum noise frequency {0}".format(FREQ_HINT))

    return int_option("--freq", default=3, callback=validate_more_than_one(), **attrs)


def width_option(**attrs):
    attrs.setdefault("help", "Output width, in pixels")

    return int_option("--width", default=1024, **attrs)


def height_option(**attrs):
    attrs.setdefault("help", "Output height, in pixels")

    return int_option("--height", default=1024, **attrs)


def channels_option(**attrs):
    attrs.setdefault("help", "Color channel count (1=gray, 2=gray+alpha, 3=HSV/RGB, 4=RGB+alpha)")

    return int_option("--channels", type=click.IntRange(1, 4), default=3, **attrs)


def octaves_option(**attrs):
    attrs.setdefault("help", "Octave count: Number of multi-res layers")

    return int_option("--octaves", type=click.IntRange(1, 10), default=1, **attrs)


def ridges_option(**attrs):
    attrs.setdefault("help", "\"Crease\" at midpoint values: abs(noise * 2 - 1)")

    return bool_option("--ridges", **attrs)


def distrib_option(**attrs):
    attrs.setdefault("help", "Value distribution")

    return str_option("--distrib", type=click.Choice([m.name for m in generators.ValueDistribution]), default="normal", **attrs)


def corners_option(**attrs):
    attrs.setdefault("help", "Value distribution: Pin pixels to corners, instead of image center.")

    return bool_option("--corners", **attrs)


def mask_option(**attrs):
    attrs.setdefault("help", "Value distribution: Hot pixel mask")

    return str_option("--mask", type=click.Choice([m.name for m in generators.ValueMask]), **attrs)


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

    return int_option("--warp-octaves", type=click.IntRange(1, 10), default=3, **attrs)


def warp_interp_option(**attrs):
    attrs.setdefault("help", "Octave Warp: Interpolation type {0}".format(INTERPOLATION_HINT))

    return int_option("--warp-interp", default=None, callback=validate_enum(effects.InterpolationType), **attrs)


def warp_freq_option(**attrs):
    attrs.setdefault("help", "Octave Warp: Override --freq for warp frequency {0}".format(FREQ_HINT))

    return int_option("--warp-freq", callback=validate_more_than_one(allow_none=True), default=None, **attrs)


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


def reverb_option(**attrs):
    attrs.setdefault("help", "Post-reduce tiled octave count")

    return int_option("--reverb", type=click.IntRange(1, 10), default=None, **attrs)


def clut_option(**attrs):
    attrs.setdefault("help", "Color lookup table (path to PNG or JPEG image)")

    return str_option("--clut", type=click.Path(exists=True, dir_okay=False, resolve_path=True), **attrs)


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


def worms_alpha_option(**attrs):
    attrs.setdefault("help", "Worms: Output {0}".format(ALPHA_BLENDING_HINT))

    return float_option("--worms-alpha", default=.875, **attrs)


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


def point_freq_option(default=3.0, **attrs):
    attrs.setdefault("help", "Voronoi/DLA: Approximate lengthwise point cloud frequency (freq * freq = count)")

    return int_option("--point-freq", type=click.IntRange(1, 10), default=default, **attrs)


def point_distrib_option(**attrs):
    attrs.setdefault("help", "Voronoi/DLA: Point cloud distribution")

    return str_option("--point-distrib", type=click.Choice([m.name for m in effects.PointDistribution]), default="random", **attrs)


def point_corners_option(**attrs):
    attrs.setdefault("help", "Voronoi/DLA: Pin diagram to corners, instead of image center.")

    return bool_option("--point-corners", **attrs)


def point_generations_option(**attrs):
    attrs.setdefault("help", "Voronoi/DLA: Penrose-ish generations. When using, keep --point-freq below ~3 to avoid OOM")

    return int_option("--point-generations", type=click.IntRange(1, 3), default=1, **attrs)


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

    return bool_option("--glitch/--no-glitch", **attrs)


def vhs_option(**attrs):
    attrs.setdefault("help", "Glitch effects: VHS tracking")

    return bool_option("--vhs/--no-vhs", **attrs)


def crt_option(**attrs):
    attrs.setdefault("help", "Glitch effects: CRT scanline")

    return bool_option("--crt/--no-crt", **attrs)


def scan_error_option(**attrs):
    attrs.setdefault("help", "Glitch effects: Analog scanline error")

    return bool_option("--scan-error/--no-scan-error", **attrs)


def snow_option(**attrs):
    attrs.setdefault("help", "Glitch effects: Analog broadcast snow (0.0=off, 1.0=saturated)")

    return float_option("--snow", **attrs)


def dither_option(**attrs):
    attrs.setdefault("help", "Glitch effects: Per-pixel brightness jitter")

    return float_option("--dither", **attrs)


def aberration_option(**attrs):
    attrs.setdefault("help", "Glitch effects: Chromatic aberration distance (e.g. .0075)")

    return float_option("--aberration", **attrs)


def light_leak_option(**attrs):
    attrs.setdefault("help", "Art effects: Light leak".format(ALPHA_BLENDING_HINT))

    return float_option("--light-leak", **attrs)


def vignette_option(**attrs):
    attrs.setdefault("help", "Art effects: Vignette {0}".format(ALPHA_BLENDING_HINT))

    return float_option("--vignette", **attrs)


def vignette_brightness_option(**attrs):
    attrs.setdefault("help", "Art effects: Vignette edge brightness (0-1)")

    return float_option("--vignette-brightness", **attrs)


def pop_option(**attrs):
    attrs.setdefault("help", "Art effects: Pop art".format(ALPHA_BLENDING_HINT))

    return bool_option("--pop", **attrs)


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
    attrs.setdefault("help", "HSV: Hue range (0..1+)")

    return float_option("--hsv-range", default=0.25, **attrs)


def hsv_rotation_option(**attrs):
    attrs.setdefault("help", "HSV: Hue rotation (0..1)")

    return float_option("--hsv-rotation", default=None, **attrs)


def hsv_saturation_option(**attrs):
    attrs.setdefault("help", "HSV: Saturation (0..1+)")

    return float_option("--hsv-saturation", default=1.0, **attrs)


def input_dir_option(**attrs):
    attrs.setdefault("help", "Input directory containing .jpg and/or .png images, for collage functions")

    return str_option("--input-dir", type=click.Path(exists=True, file_okay=False, resolve_path=True), **attrs)


def name_option(default=None, **attrs):
    attrs.setdefault("help", "Filename for image output (should end with .png or .jpg)")

    return str_option("--name", type=click.Path(dir_okay=False), default=default or "noise.png", **attrs)