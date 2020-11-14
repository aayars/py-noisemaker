"""Common CLI boilerplate for Noisemaker"""

import click

from noisemaker.constants import (
    DistanceMetric,
    InterpolationType,
    PointDistribution,
    ValueDistribution,
    ValueMask,
    VoronoiDiagramType,
    WormBehavior
)

from noisemaker.palettes import PALETTES as palettes

import noisemaker.masks as masks

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def show_values(enum_class):
    out = []

    for member in enum_class:
        out.append(f"{member.value}={member.name}")

    return f"({', '.join(out)})"


CLICK_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 160}

# Boilerplate help strings

ALPHA_BLENDING_HINT = "alpha blending amount (0.0 = 0%, 1.0 = 100%)"

DISTANCE_HINT = show_values(DistanceMetric)

ENTIRE_IMAGE_HINT = "(1.0 = height/width of entire image)"

FREQ_HINT = "(must be >= 2)"

INTERPOLATION_HINT = show_values(InterpolationType)

NEAREST_NEIGHBOR_HINT = "(1.0 = as far as nearest neighbor)"

Y_FROM_OFFSET_HINT = "Use offset X values for Y (instead of sin/cos)"


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
    attrs.setdefault("default", attrs.get("default", False))

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


def multi_str_option(attr, **attrs):
    return str_option(attr, multiple=True, **attrs)


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


def width_option(default=1024, **attrs):
    attrs.setdefault("help", "Output width, in pixels")

    return int_option("--width", default=default, **attrs)


def height_option(default=1024, **attrs):
    attrs.setdefault("help", "Output height, in pixels")

    return int_option("--height", default=default, **attrs)


def channels_option(**attrs):
    attrs.setdefault("help", "Color channel count (1=gray, 2=gray+alpha, 3=HSV/RGB, 4=RGB+alpha)")

    return int_option("--channels", type=click.IntRange(1, 4), default=3, **attrs)


def time_option(**attrs):
    attrs.setdefault("help", "Time value for Z axis (simplex only)")

    return float_option("--time", default=0.0, **attrs)


def octaves_option(**attrs):
    attrs.setdefault("help", "Octave count: Number of multi-res layers")

    return int_option("--octaves", type=click.IntRange(1, 10), default=1, **attrs)


def reduce_max_option(**attrs):
    attrs.setdefault("help", "Blend maximum per-octave values, instead of adding")

    return bool_option("--reduce-max/--no-reduce-max", default=False, **attrs)


def ridges_option(**attrs):
    attrs.setdefault("help", "Per-octave \"crease\" at midpoint values: abs(noise * 2 - 1)")

    return bool_option("--ridges", **attrs)


def post_ridges_option(**attrs):
    attrs.setdefault("help", "Post-reduce \"crease\" at midpoint values: abs(noise * 2 - 1)")

    return bool_option("--post-ridges", **attrs)


def distrib_option(**attrs):
    attrs.setdefault("help", "Value distribution")

    return str_option("--distrib", type=click.Choice([m.name for m in ValueDistribution]), default="normal", **attrs)


def corners_option(**attrs):
    attrs.setdefault("help", "Value distribution: Pin pixels to corners, instead of image center.")

    return bool_option("--corners", **attrs)


def mask_option(**attrs):
    attrs.setdefault("help", "Value distribution: Hot pixel mask")

    return str_option("--mask", type=click.Choice([m.name for m in ValueMask]), **attrs)


def mask_inverse_option(**attrs):
    attrs.setdefault("help", "Mask: Invert hot pixels")

    return bool_option("--mask-inverse", **attrs)


def glyph_map_option(**attrs):
    attrs.setdefault("help", "Mask: Glyph map brightness atlas mask")

    choices = sorted(m.name for m in masks.square_masks())

    return str_option("--glyph-map", type=click.Choice(choices), **attrs)


def glyph_map_colorize_option(**attrs):
    attrs.setdefault("help", "Glyph map: Colorize exploded pixels")

    return bool_option("--glyph-map-colorize", **attrs)


def glyph_map_zoom_option(**attrs):
    attrs.setdefault("help", "Glyph map: Exploded pixel zoom factor")

    return float_option("--glyph-map-zoom", default=4.0, **attrs)


def glyph_map_alpha_option(**attrs):
    attrs.setdefault("help", "Glyph map: Output {0}".format(ALPHA_BLENDING_HINT))

    return float_option("--glyph-map-alpha", default=1.0, **attrs)


def composite_option(**attrs):
    attrs.setdefault("help", "Mask: Composite video effect mask")

    return str_option("--composite", type=click.Choice([m.name for m in ValueMask.rgb_members()]), **attrs)


def composite_zoom_option(**attrs):
    attrs.setdefault("help", "Composite video effect: Exploded pixel zoom factor")

    return float_option("--composite-zoom", default=2.0, **attrs)


def interp_option(**attrs):
    attrs.setdefault("help", "Interpolation type {0}".format(INTERPOLATION_HINT))

    return int_option("--interp", callback=validate_enum(InterpolationType), default=3, **attrs)


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

    return int_option("--warp-interp", default=None, callback=validate_enum(InterpolationType), **attrs)


def warp_freq_option(**attrs):
    attrs.setdefault("help", "Octave Warp: Override --freq for warp frequency {0}".format(FREQ_HINT))

    return int_option("--warp-freq", callback=validate_more_than_one(allow_none=True), default=None, **attrs)


def warp_map_option(**attrs):
    attrs.setdefault("help", "Octave Warp: Filename of image with brightness values")

    return str_option("--warp-map", type=click.Path(exists=True, dir_okay=False, resolve_path=True), **attrs)


def post_reindex_option(**attrs):
    attrs.setdefault("help", "Post-reduce color re-indexing range {0}".format(ENTIRE_IMAGE_HINT))

    return float_option("--post-reindex", **attrs)


def post_reflect_option(**attrs):
    attrs.setdefault("help", "Domain warping: Post-reduce derivative-based displacement range {0}".format(ENTIRE_IMAGE_HINT))

    return float_option("--post-reflect", **attrs)


def post_refract_option(**attrs):
    attrs.setdefault("help", "Domain warping: Post-reduce self-displacement range {0}".format(ENTIRE_IMAGE_HINT))

    return float_option("--post-refract", **attrs)


def post_refract_y_from_offset_option(**attrs):
    attrs.setdefault("help", "Domain warping: Post-reduce refract: {0}".format(Y_FROM_OFFSET_HINT))

    return bool_option("--post-refract-y-from-offset/--no-post-refract-y-from-offset", default=True, **attrs)


def reflect_option(**attrs):
    attrs.setdefault("help", "Domain warping: Per-octave derivative-based displacement range {0}".format(ENTIRE_IMAGE_HINT))

    return float_option("--reflect", **attrs)


def refract_option(**attrs):
    attrs.setdefault("help", "Domain warping: Per-octave self-displacement range {0}".format(ENTIRE_IMAGE_HINT))

    return float_option("--refract", **attrs)


def refract_y_from_offset_option(**attrs):
    attrs.setdefault("help", "Domain warping: Per-octave refract: {0}".format(Y_FROM_OFFSET_HINT))

    return bool_option("--refract-y-from-offset/--no-refract-y-from-offset", **attrs)


def ripple_option(**attrs):
    attrs.setdefault("help", "Ripple effect: Displacement range {0}".format(ENTIRE_IMAGE_HINT))

    return float_option("--ripple", default=None, **attrs)


def ripple_freq_option(**attrs):
    attrs.setdefault("help", "Ripple effect: Override --freq for ripple frequency {0}".format(FREQ_HINT))

    return int_option("--ripple-freq", default=3, **attrs)


def ripple_kink_option(**attrs):
    attrs.setdefault("help", "Ripple effect: Ripple amplitude")

    return float_option("--ripple-kink", default=1.0, **attrs)


def reindex_option(**attrs):
    attrs.setdefault("help", "Color re-indexing range {0}".format(ENTIRE_IMAGE_HINT))

    return float_option("--reindex", **attrs)


def reverb_option(**attrs):
    attrs.setdefault("help", "Post-reduce tiled octave count")

    return int_option("--reverb", type=click.IntRange(1, 10), default=None, **attrs)


def reverb_iterations_option(**attrs):
    attrs.setdefault("help", "Reverb: Re-reverberate N times")

    return int_option("--reverb-iterations", type=click.IntRange(1, 4), default=1, **attrs)


def palette_option(**attrs):
    attrs.setdefault("help", "Apply named cosine palette")

    return str_option('--palette', type=click.Choice(sorted(palettes)), **attrs)


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
    attrs.setdefault("help", f"Iterative \"worm\" field flow {show_values(WormBehavior)}")

    return int_option("--worms", callback=validate_enum(WormBehavior), **attrs)


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
    attrs.setdefault("help", f"Generate a Voronoi diagram {show_values(VoronoiDiagramType)}")

    return int_option("--voronoi", callback=validate_enum(VoronoiDiagramType), **attrs)


def voronoi_metric_option(**attrs):
    attrs.setdefault("help", "Voronoi: Distance metric {0}".format(DISTANCE_HINT))

    return int_option("--voronoi-metric", callback=validate_enum(DistanceMetric), default=1, **attrs)


def voronoi_nth_option(**attrs):
    attrs.setdefault("help", "Voronoi: Plot Nth nearest, or -Nth farthest")

    return int_option("--voronoi-nth", **attrs)


def voronoi_alpha_option(**attrs):
    attrs.setdefault("help", "Voronoi: Basis {0}".format(ALPHA_BLENDING_HINT))

    return float_option("--voronoi-alpha", default=1.0, **attrs)


def voronoi_refract_option(**attrs):
    attrs.setdefault("help", "Voronoi: Domain warp input tensor {0}".format(ENTIRE_IMAGE_HINT))

    return float_option("--voronoi-refract", **attrs)


def voronoi_refract_y_from_offset_option(**attrs):
    attrs.setdefault("help", "Domain warping: Voronoi refract: {0}".format(Y_FROM_OFFSET_HINT))

    return bool_option("--voronoi-refract-y-from-offset/--no-voronoi-refract-y-from-offset", default=True, **attrs)


def voronoi_inverse_option(**attrs):
    attrs.setdefault("help", "Voronoi: Inverse range")

    return bool_option("--voronoi-inverse", **attrs)


def point_freq_option(default=3.0, **attrs):
    attrs.setdefault("help", "Voronoi/DLA: Approximate lengthwise point cloud frequency (freq * freq = count)")

    return int_option("--point-freq", type=click.IntRange(1, 10), default=default, **attrs)


def point_distrib_option(**attrs):
    attrs.setdefault("help", "Voronoi/DLA: Point cloud distribution")

    return str_option("--point-distrib",
            type=click.Choice(
                [m.name for m in PointDistribution]
                + [m.name for m in ValueMask.nonprocedural_members()]
            ), default="random", **attrs)


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

    return int_option("--sobel", callback=validate_enum(DistanceMetric), **attrs)


def outline_option(**attrs):
    attrs.setdefault("help", "Post-processing: Apply Sobel operator, and multiply {0}".format(DISTANCE_HINT))

    return int_option("--outline", callback=validate_enum(DistanceMetric), **attrs)


def normals_option(**attrs):
    attrs.setdefault("help", "Post-processing: Generate a tangent-space normal map")

    return bool_option("--normals", **attrs)


def post_deriv_option(**attrs):
    attrs.setdefault("help", "Derivatives: Extract post-reduce rate of change {0}".format(DISTANCE_HINT))

    return int_option("--post-deriv", callback=validate_enum(DistanceMetric), **attrs)


def deriv_option(**attrs):
    attrs.setdefault("help", "Derivatives: Extract per-octave rate of change {0}".format(DISTANCE_HINT))

    return int_option("--deriv", callback=validate_enum(DistanceMetric), **attrs)


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
    attrs.setdefault("help", "Art effects: Pop art")

    return bool_option("--pop", **attrs)


def convolve_option(**attrs):
    attrs.setdefault("help", "Convolution kernel: May be specified multiple times")

    return multi_str_option("--convolve", type=click.Choice([m.name.replace('conv2d_', '') for m in ValueMask.conv2d_members()]), **attrs)


def shadow_option(**attrs):
    attrs.setdefault("help", "Shadow {0}".format(ALPHA_BLENDING_HINT))

    return float_option("--shadow", **attrs)


def rgb_option(**attrs):
    attrs.setdefault("help", "Use RGB noise basis instead of HSV")

    return bool_option("--rgb", **attrs)


def hue_range_option(**attrs):
    attrs.setdefault("help", "HSV: Hue range (0..1+)")

    return float_option("--hue-range", default=0.25, **attrs)


def hue_rotation_option(**attrs):
    attrs.setdefault("help", "HSV: Hue rotation (0..1)")

    return float_option("--hue-rotation", default=None, **attrs)


def saturation_option(**attrs):
    attrs.setdefault("help", "HSV: Saturation (0..1+)")

    return float_option("--saturation", default=1.0, **attrs)


def hue_distrib_option(**attrs):
    attrs.setdefault("help", "HSV: Override value distribution for hue")

    return str_option("--hue-distrib", type=click.Choice([m.name for m in ValueDistribution]), default=None, **attrs)


def saturation_distrib_option(**attrs):
    attrs.setdefault("help", "HSV: Override value distribution for saturation")

    return str_option("--saturation-distrib", type=click.Choice([m.name for m in ValueDistribution]), default=None, **attrs)


def brightness_distrib_option(**attrs):
    attrs.setdefault("help", "HSV: Override value distribution for brightness")

    return str_option("--brightness-distrib", type=click.Choice([m.name for m in ValueDistribution]), default=None, **attrs)


def post_hue_rotation_option(**attrs):
    attrs.setdefault("help", "HSV: Post-reduce hue rotation (-0.5 .. 0.5)")

    return float_option("--post-hue-rotation", default=None, **attrs)


def post_saturation_option(**attrs):
    attrs.setdefault("help", "HSV: Post-reduce saturation")

    return float_option("--post-saturation", default=None, **attrs)


def post_contrast_option(**attrs):
    attrs.setdefault("help", "HSV: Post-reduce contrast adjustment")

    return float_option("--post-contrast", default=None, **attrs)


def density_map_option(**attrs):
    attrs.setdefault("help", "Map values to color density histogram")

    return bool_option("--density", default=False, **attrs)


def input_dir_option(**attrs):
    attrs.setdefault("help", "Input directory containing .jpg and/or .png images")

    return str_option("--input-dir", type=click.Path(exists=True, file_okay=False, resolve_path=True), **attrs)


def seed_option(**attrs):
    attrs.setdefault("help", "Random seed. Might not affect all things.")

    return int_option("--seed", default=None, **attrs)


def name_option(default=None, **attrs):
    attrs.setdefault("help", "Filename for image output (should end with .png or .jpg)")

    return str_option("--name", type=click.Path(dir_okay=False), default=default or "noise.png", **attrs)
