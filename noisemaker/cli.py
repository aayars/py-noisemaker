import click
import tensorflow as tf

from noisemaker.util import save

import noisemaker.effects as effects
import noisemaker.generators as generators
import noisemaker.recipes as recipes


CLICK_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 160}

# Boilerplate help strings
ENTIRE_IMAGE_HINT = "(1.0 = height/width of entire image)"
DISTANCE_HINT = "(1=Euclidean, 2=Manhattan, 3=Chebyshev)"
ALPHA_BLENDING_HINT = "alpha blending amount (0.0 - 1.0)"


def option(*param_decls, **attrs):
    """ Add a Click option. """

    def decorator(f):
        if "default" in attrs:
            attrs["help"] += "  [default: {0}]".format(attrs["default"])

        return click.option(*param_decls, **attrs)(f)

    return decorator


def freq_option(**attrs):
    attrs.setdefault("help", "Minimum noise frequency")
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


def octaves_option(**attrs):
    attrs.setdefault("help", "Octave count: Number of multi-res layers")
    attrs.setdefault("type", int)
    attrs.setdefault("default", 1)

    return option("--octaves", **attrs)


def ridges_option(**attrs):
    attrs.setdefault("help", "\"Crease\" at midpoint values: abs(noise * 2 - 1)")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--ridges", **attrs)


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


def sin_option(**attrs):
    attrs.setdefault("help", "Apply sin function to noise basis")
    attrs.setdefault("type", float)
    attrs.setdefault("default", False)

    return option("--sin", **attrs)


def wavelet_option(**attrs):
    attrs.setdefault("help", "Wavelets: What are they even?")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--wavelet", **attrs)


def lattice_drift_option(**attrs):
    attrs.setdefault("help", "Domain warping: Lattice deform range (1.0 = nearest neighbor)")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.0)

    return option("--lattice-drift", **attrs)


def vortex_option(**attrs):
    attrs.setdefault("help", "Vortex tiling amount")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.0)

    return option("--vortex", **attrs)


def warp_option(**attrs):
    attrs.setdefault("help", "Octave Warp: Orthogonal displacement range {0}".format(ENTIRE_IMAGE_HINT))
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.0)

    return option("--warp", **attrs)


def warp_octaves_option(**attrs):
    attrs.setdefault("help", "Octave Warp: Octave count for --warp")
    attrs.setdefault("type", int)
    attrs.setdefault("default", 3)

    return option("--warp-octaves", **attrs)


def warp_interp_option(**attrs):
    attrs.setdefault("help", "Octave Warp: Interpolation type (0=constant, 1=linear, 2=cosine, 3=bicubic)")
    attrs.setdefault("type", int)
    attrs.setdefault("default", None)

    return option("--warp-interp", **attrs)


def warp_freq_option(**attrs):
    attrs.setdefault("help", "Octave Warp: Frequency (Default: Use --freq if not given)")
    attrs.setdefault("type", int)
    attrs.setdefault("default", None)

    return option("--warp-freq", **attrs)


def post_reflect_option(**attrs):
    attrs.setdefault("help", "Domain warping: Reduced derivative-based displacement range {0}".format(ENTIRE_IMAGE_HINT))
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.0)

    return option("--post-reflect", **attrs)


def post_refract_option(**attrs):
    attrs.setdefault("help", "Domain warping: Reduced self-displacement range {0}".format(ENTIRE_IMAGE_HINT))
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.0)

    return option("--post-refract", **attrs)


def reflect_option(**attrs):
    attrs.setdefault("help", "Domain warping: Per-octave derivative-based displacement range {0}".format(ENTIRE_IMAGE_HINT))
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.0)

    return option("--reflect", **attrs)


def refract_option(**attrs):
    attrs.setdefault("help", "Domain warping: Per-octave self-displacement range {0}".format(ENTIRE_IMAGE_HINT))
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.0)

    return option("--refract", **attrs)


def reindex_option(**attrs):
    attrs.setdefault("help", "Color re-indexing range {0}".format(ENTIRE_IMAGE_HINT))
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.0)

    return option("--reindex", **attrs)


def clut_option(**attrs):
    attrs.setdefault("help", "Color lookup table (path to PNG or JPEG image)")
    attrs.setdefault("type", str)

    return option("--clut", **attrs)


def clut_range_option(**attrs):
    attrs.setdefault("help", "CLUT: Maximum pixel gather distance {0}".format(ENTIRE_IMAGE_HINT))
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.5)

    return option("--clut-range", **attrs)


def clut_horizontal_option(**attrs):
    attrs.setdefault("help", "CLUT: Preserve vertical axis")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--clut-horizontal", **attrs)


def worms_option(**attrs):
    attrs.setdefault("help", "Iterative \"worm\" field flow (1=Obedient, 2=Crosshatch, 3=Unruly, 4=Chaotic)")
    attrs.setdefault("type", int)
    attrs.setdefault("default", 0)

    return option("--worms", **attrs)


def worms_density_option(**attrs):
    attrs.setdefault("help", "Worms: Density multiplier (larger is more costly)")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 4.0)

    return option("--worms-density", **attrs)


def worms_duration_option(**attrs):
    attrs.setdefault("help", "Worms: Iteration multiplier (larger is more costly)")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 4.0)

    return option("--worms-duration", **attrs)


def worms_stride_option(**attrs):
    attrs.setdefault("help", "Worms: Mean pixel displacement per iteration")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 1.0)

    return option("--worms-stride", **attrs)


def worms_stride_deviation_option(**attrs):
    attrs.setdefault("help", "Worms: Per-worm random stride variance")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.0)

    return option("--worms-stride-deviation", **attrs)


def worms_bg_option(**attrs):
    attrs.setdefault("help", "Worms: Background color brightness")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.5)

    return option("--worms-bg", **attrs)


def worms_kink_option(**attrs):
    attrs.setdefault("help", "Worms: Rotation range (1.0 = 360 degrees)")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 1.0)

    return option("--worms-kink", **attrs)


def wormhole_option(**attrs):
    attrs.setdefault("help", "Non-iterative per-pixel field flow")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--wormhole", **attrs)


def wormhole_stride_option(**attrs):
    attrs.setdefault("help", "Wormhole: Max per-pixel displacement range {0}".format(ENTIRE_IMAGE_HINT))
    attrs.setdefault("type", float)
    attrs.setdefault("default", .1)

    return option("--wormhole-stride", **attrs)


def wormhole_kink_option(**attrs):
    attrs.setdefault("help", "Wormhole: Per-pixel rotation range (1.0 = 360 degrees)")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 1.0)

    return option("--wormhole-kink", **attrs)


def erosion_worms_option(**attrs):
    attrs.setdefault("help", "Experimental erosion worms (Does not use worms settings)")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--erosion-worms", **attrs)


def dla_option(**attrs):
    attrs.setdefault("help", "Diffusion-limited aggregation (DLA) alpha")
    attrs.setdefault("type", float)
    attrs.setdefault("default", None)

    return option("--dla", **attrs)


def dla_padding_option(**attrs):
    attrs.setdefault("help", "DLA: Pixel padding (smaller is slower)")
    attrs.setdefault("type", int)
    attrs.setdefault("default", 2)

    return option("--dla-padding", **attrs)


def voronoi_option(**attrs):
    attrs.setdefault("help", "Generate a Voronoi diagram (0=Off, 1=Range, 2=Color Range, 3=Indexed, 4=Color Map, 5=Blended, 6=Flow)")
    attrs.setdefault("type", int)
    attrs.setdefault("default", 0)

    return option("--voronoi", **attrs)


def voronoi_func_option(**attrs):
    attrs.setdefault("help", "Voronoi: Distance function {0}".format(DISTANCE_HINT))
    attrs.setdefault("type", int)
    attrs.setdefault("default", 1)

    return option("--voronoi-func", **attrs)


def voronoi_nth_option(**attrs):
    attrs.setdefault("help", "Voronoi: Plot Nth nearest, or -Nth farthest")
    attrs.setdefault("type", int)
    attrs.setdefault("default", 0)

    return option("--voronoi-nth", **attrs)


def voronoi_alpha_option(**attrs):
    attrs.setdefault("help", "Voronoi: Blend with original tensor (0.0 = Original, 1.0 = Voronoi)")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 1.0)

    return option("--voronoi-alpha", **attrs)


def voronoi_refract_option(**attrs):
    attrs.setdefault("help", "Voronoi: Domain warp input tensor {0}".format(ENTIRE_IMAGE_HINT))
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.0)

    return option("--voronoi-refract", **attrs)


def voronoi_inverse_option(**attrs):
    attrs.setdefault("help", "Voronoi: Inverse range")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--voronoi-inverse", **attrs)


def point_freq_option(**attrs):
    attrs.setdefault("help", "Voronoi/DLA: Approximate lengthwise point cloud frequency (freq * freq = count)")
    attrs.setdefault("type", click.IntRange(1, 10))
    attrs.setdefault("default", 3)

    return option("--point-freq", **attrs)


def point_distrib_option(**attrs):
    attrs.setdefault("help", "Voronoi/DLA: Point cloud distribution")
    attrs.setdefault("type", click.Choice([m.name for m in effects.PointDistribution]))
    attrs.setdefault("default", "random")

    return option("--point-distrib", **attrs)


def point_corners_option(**attrs):
    attrs.setdefault("help", "Voronoi/DLA: Pin diagram to corners, instead of image center.")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--point-corners", **attrs)


def point_generations_option(**attrs):
    attrs.setdefault("help", "Voronoi/DLA: Penrose generations. When using, keep this and freq below ~3 or you will run OOM easily.")
    attrs.setdefault("type", int)
    attrs.setdefault("default", 1)

    return option("--point-generations", **attrs)


def sobel_option(**attrs):
    attrs.setdefault("help", "Post-processing: Apply Sobel operator {0}".format(DISTANCE_HINT))
    attrs.setdefault("type", int)
    attrs.setdefault("default", None)

    return option("--sobel", **attrs)


def outline_option(**attrs):
    attrs.setdefault("help", "Post-processing: Apply Sobel operator, and multiply {0}".format(DISTANCE_HINT))
    attrs.setdefault("type", int)
    attrs.setdefault("default", None)

    return option("--outline", **attrs)


def normals_option(**attrs):
    attrs.setdefault("help", "Post-processing: Generate a tangent-space normal map")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--normals", **attrs)


def post_deriv_option(**attrs):
    attrs.setdefault("help", "Derivatives: Extract reduced rate of change {0}".format(DISTANCE_HINT))
    attrs.setdefault("type", int)
    attrs.setdefault("default", None)

    return option("--post-deriv", **attrs)


def deriv_option(**attrs):
    attrs.setdefault("help", "Derivatives: Extract per-octave rate of change {0}".format(DISTANCE_HINT))
    attrs.setdefault("type", int)
    attrs.setdefault("default", None)

    return option("--deriv", **attrs)


def deriv_alpha_option(**attrs):
    attrs.setdefault("help", "Derivatives: Per-octave alpha")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 1.0)

    return option("--deriv-alpha", **attrs)


def posterize_option(**attrs):
    attrs.setdefault("help", "Post-processing: Posterize levels (per channel)")
    attrs.setdefault("type", int)
    attrs.setdefault("default", 0)

    return option("--posterize", **attrs)


def bloom_option(**attrs):
    attrs.setdefault("help", "Post-processing: Bloom alpha")
    attrs.setdefault("type", float)
    attrs.setdefault("default", None)

    return option("--bloom", **attrs)


def glitch_option(**attrs):
    attrs.setdefault("help", "Glitch effects: Bit-shit")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--glitch", **attrs)


def vhs_option(**attrs):
    attrs.setdefault("help", "Glitch effects: VHS tracking")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--vhs", **attrs)


def crt_option(**attrs):
    attrs.setdefault("help", "Glitch effects: CRT scanline")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--crt", **attrs)


def scan_error_option(**attrs):
    attrs.setdefault("help", "Glitch effects: Analog scanline error")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--scan-error", **attrs)


def snow_option(**attrs):
    attrs.setdefault("help", "Glitch effects: Analog broadcast snow (0.0=off, 1.0=saturated)")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.0)

    return option("--snow", **attrs)


def dither_option(**attrs):
    attrs.setdefault("help", "Glitch effects: Per-pixel brightness jitter")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.0)

    return option("--dither", **attrs)


def aberration_option(**attrs):
    attrs.setdefault("help", "Glitch effects: Chromatic aberration distance (e.g. .0075)")
    attrs.setdefault("type", float)
    attrs.setdefault("default", None)

    return option("--aberration", **attrs)


def emboss_option(**attrs):
    attrs.setdefault("help", "Convolution kernel: Emboss {0}".format(ALPHA_BLENDING_HINT))
    attrs.setdefault("type", float)
    attrs.setdefault("default", None)

    return option("--emboss", **attrs)


def shadow_option(**attrs):
    attrs.setdefault("help", "Convolution kernel: Shadow {0}".format(ALPHA_BLENDING_HINT))
    attrs.setdefault("type", float)
    attrs.setdefault("default", None)

    return option("--shadow", **attrs)


def edges_option(**attrs):
    attrs.setdefault("help", "Convolution kernel: Edges {0}".format(ALPHA_BLENDING_HINT))
    attrs.setdefault("type", float)
    attrs.setdefault("default", None)

    return option("--edges", **attrs)


def sharpen_option(**attrs):
    attrs.setdefault("help", "Convolution kernel: Sharpen {0}".format(ALPHA_BLENDING_HINT))
    attrs.setdefault("type", float)
    attrs.setdefault("default", None)

    return option("--sharpen", **attrs)


def unsharp_mask_option(**attrs):
    attrs.setdefault("help", "Convolution kernel: Unsharp mask {0}".format(ALPHA_BLENDING_HINT))
    attrs.setdefault("type", float)
    attrs.setdefault("default", None)

    return option("--unsharp-mask", **attrs)


def invert_option(**attrs):
    attrs.setdefault("help", "Convolution kernel: Invert {0}".format(ALPHA_BLENDING_HINT))
    attrs.setdefault("type", float)
    attrs.setdefault("default", None)

    return option("--invert", **attrs)


def rgb_option(**attrs):
    attrs.setdefault("help", "Use RGB noise basis instead of HSV")
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", False)

    return option("--rgb", **attrs)


def hsv_range_option(**attrs):
    attrs.setdefault("help", "HSV: Hue range (0..1+")
    attrs.setdefault("type", float)
    attrs.setdefault("default", .25)

    return option("--hsv-range", **attrs)


def hsv_rotation_option(**attrs):
    attrs.setdefault("help", "HSV: Hue rotation (0..1)")
    attrs.setdefault("type", float)
    attrs.setdefault("default", None)

    return option("--hsv-rotation", **attrs)


def hsv_saturation_option(**attrs):
    attrs.setdefault("help", "HSV: Saturation (0..1+)")
    attrs.setdefault("type", float)
    attrs.setdefault("default", 1.0)

    return option("--hsv-saturation", **attrs)


def name_option(**attrs):
    attrs.setdefault("help", "Base filename for image output")
    attrs.setdefault("type", str)
    attrs.setdefault("default", "noise.png")

    return option("--name", **attrs)


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
@wavelet_option()
@name_option()
@click.pass_context
def main(ctx, freq, width, height, channels, octaves, ridges, sin, wavelet, lattice_drift, vortex, warp, warp_octaves, warp_interp, warp_freq, reflect, refract, reindex,
         post_reflect, post_refract, clut, clut_horizontal, clut_range, worms, worms_density, worms_duration, worms_stride, worms_stride_deviation,
         worms_bg, worms_kink, wormhole, wormhole_kink, wormhole_stride, sobel, outline, normals, post_deriv, deriv, deriv_alpha, interp, distrib, posterize,
         erosion_worms, voronoi, voronoi_func, voronoi_nth, voronoi_alpha, voronoi_refract, voronoi_inverse,
         glitch, vhs, crt, scan_error, snow, dither, aberration, bloom, rgb, hsv_range, hsv_rotation, hsv_saturation,
         dla, dla_padding, point_freq, point_distrib, point_corners, point_generations,
         name, **convolve_kwargs):

    with tf.Session().as_default():
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
                                     point_generations=point_generations,
                                     with_outline=outline, with_sobel=sobel, with_normal_map=normals, post_deriv=post_deriv, deriv=deriv, deriv_alpha=deriv_alpha,
                                     spline_order=interp, distrib=distrib, warp_range=warp, warp_octaves=warp_octaves, warp_interp=warp_interp, warp_freq=warp_freq,
                                     posterize_levels=posterize, vortex_range=vortex,
                                     hsv=not rgb, hsv_range=hsv_range, hsv_rotation=hsv_rotation, hsv_saturation=hsv_saturation,
                                     with_aberration=aberration, with_bloom=bloom, **convolve_kwargs)

        tensor = recipes.post_process(tensor, shape, freq, with_glitch=glitch, with_vhs=vhs, with_crt=crt, with_scan_error=scan_error, with_snow=snow, with_dither=dither)

        save(tensor, name)

        print(name)