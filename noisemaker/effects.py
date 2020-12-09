"""Low-level effects library for Noisemaker"""

import inspect
import math
import random

import numpy as np
import pyfastnoisesimd as fn
import tensorflow as tf
import tensorflow_addons as tfa

from noisemaker.constants import (
    DistanceMetric,
    InterpolationType,
    PointDistribution,
    ValueDistribution,
    ValueMask,
    VoronoiDiagramType,
    WormBehavior
)
from noisemaker.glyphs import load_glyphs
from noisemaker.palettes import PALETTES as palettes
from noisemaker.points import point_cloud

import noisemaker.masks as masks
import noisemaker.simplex as simplex
import noisemaker.util as util
import noisemaker.value as value

EFFECTS = {}


def post_process(tensor, shape, freq, ridges_hint=False, spline_order=InterpolationType.bicubic,
                 reflect_range=0.0, refract_range=0.0, reindex_range=0.0, refract_y_from_offset=False, refract_signed_range=False,
                 clut=None, clut_horizontal=False, clut_range=0.5,
                 with_worms=None, worms_density=4.0, worms_duration=4.0, worms_stride=1.0, worms_stride_deviation=.05,
                 worms_alpha=.5, worms_kink=1.0, worms_drunkenness=0.0, worms_drunken_spin=False,
                 with_sobel=None, with_normal_map=False, deriv=None, deriv_alpha=1.0, with_outline=False,
                 with_glowing_edges=False, with_wormhole=False, wormhole_kink=2.5, wormhole_stride=.1, wormhole_alpha=1.0,
                 with_voronoi=0, voronoi_nth=0, voronoi_metric=DistanceMetric.euclidean, voronoi_alpha=1.0, voronoi_refract=0.0, voronoi_inverse=False,
                 voronoi_refract_y_from_offset=True, posterize_levels=0,
                 with_erosion_worms=False, erosion_worms_density=50, erosion_worms_iterations=50, erosion_worms_contraction=1.0,
                 erosion_worms_alpha=1.0, erosion_worms_inverse=False, erosion_worms_xy_blend=None,
                 warp_range=0.0, warp_octaves=3, warp_interp=None, warp_freq=None, warp_map=None, warp_signed_range=True,
                 ripple_range=0.0, ripple_freq=None, ripple_kink=1.0,
                 vortex_range=0.0, with_aberration=None, with_dla=0.0, dla_padding=2,
                 point_freq=5, point_distrib=1000000, point_corners=False, point_generations=1, point_drift=0.0,
                 with_bloom=None, with_reverb=None, reverb_iterations=1, reverb_ridges=True,
                 with_light_leak=None, with_vignette=None, vignette_brightness=0.0, with_vaseline=0.0,
                 post_hue_rotation=None, post_saturation=None, post_brightness=None, post_contrast=None,
                 with_ridge=False, with_jpeg_decimate=None, with_conv_feedback=None, conv_feedback_alpha=.5,
                 with_density_map=False,
                 with_glyph_map=None, glyph_map_colorize=True, glyph_map_zoom=1.0, glyph_map_alpha=1.0,
                 with_composite=None, composite_zoom=4.0, with_sort=False, sort_angled=False, sort_darkest=False,
                 with_convolve=None, with_shadow=None, with_sketch=False,
                 with_lowpoly=False, lowpoly_distrib=1000000, lowpoly_freq=10, lowpoly_metric=DistanceMetric.euclidean,
                 angle=None,
                 with_simple_frame=False,
                 with_kaleido=None, kaleido_dist_metric=DistanceMetric.euclidean, kaleido_blend_edges=True,
                 with_wobble=None, with_palette=None,
                 with_glitch=False, with_vhs=False, with_crt=False, with_scan_error=False, with_snow=False, with_dither=False,
                 with_nebula=False, with_false_color=False, with_frame=False, with_scratches=False, with_fibers=False,
                 with_stray_hair=False, with_grime=False, with_watermark=False, with_ticker=False, with_texture=False,
                 with_pre_spatter=False, with_spatter=False, with_clouds=False, with_lens_warp=None, with_tint=None, with_degauss=False,
                 rgb=False, time=0.0, speed=1.0, **_):
    """
    Apply post-processing effects.

    :param Tensor tensor:
    :param list[int] shape:
    :param list[int] freq:
    :param int spline_order: Ortho spline point count (0=Constant, 1=Linear, 2=Cosine, 3=Bicubic)
    :param float reflect_range: Derivative distortion gradient.
    :param float refract_range: Self-distortion gradient.
    :param float refract_y_from_offset: Derive Y offset values from offsetting the image.
    :param float reindex_range: Self-reindexing gradient.
    :param str clut: PNG or JPG color lookup table filename.
    :param float clut_horizontal: Preserve clut Y axis.
    :param float clut_range: Gather range for clut.
    :param WormBehavior|None with_worms: Do worms.
    :param float worms_density: Worm density multiplier (larger == slower)
    :param float worms_duration: Iteration multiplier (larger == slower)
    :param float worms_stride: Mean travel distance per iteration
    :param float worms_stride_deviation: Per-worm travel distance deviation
    :param float worms_alpha: Fade worms (0..1)
    :param float worms_kink: Worm twistiness
    :param DistanceMetric|int sobel: Sobel operator distance metric
    :param DistanceMetric|int outline: Outlines distance metric (multiply)
    :param bool with_normal_map: Create a tangent-space normal map
    :param bool with_wormhole: Wormhole effect. What is this?
    :param float wormhole_kink: Wormhole kinkiness, if you're into that.
    :param float wormhole_stride: Wormhole thickness range
    :param float wormhole_alpha: Wormhole alpha blending
    :param VoronoiDiagramType|int with_voronoi: Voronoi diagram type (0=Off, 1=Range, 2=Color Range, 3=Indexed, 4=Color Map, 5=Blended, 6=Flow)
    :param int voronoi_nth: Voronoi Nth nearest
    :param DistanceMetric|int voronoi_metric: Voronoi distance metric
    :param float voronoi_alpha: Blend with original tensor (0.0 = Original, 1.0 = Voronoi)
    :param float voronoi_refract: Domain warp input tensor against Voronoi
    :param bool voronoi_refract_y_from_offset: Derive Y offsets from offsetting image
    :param bool voronoi_inverse: Inverse values for Voronoi 'range' types
    :param bool ridges_hint: Ridged multifractal hint for Voronoi
    :param DistanceMetric|int deriv: Derivative distance metric
    :param float deriv_alpha: Derivative alpha blending amount
    :param float posterize_levels: Posterize levels
    :param bool with_erosion_worms: Erosion worms
    :param float erosion_worms_density: Default: 50
    :param float erosion_worms_iterations: Default: 50
    :param float erosion_worms_contraction: Inverse of stride. Default: 1.0, smaller=longer steps
    :param float erosion_worms_alpha:
    :param bool erosion_worms_inverse:
    :param None|float erosion_worms_xy_blend:
    :param float vortex_range: Vortex tiling amount
    :param float warp_range: Orthogonal distortion gradient.
    :param int warp_octaves: Multi-res iteration count for warp
    :param int|None warp_interp: Override spline order for warp (None = use spline_order)
    :param int|None warp_freq: Override frequency for warp (None = use freq)
    :param str|None warp_map: File with brightness values for warp (None = generate noise)
    :param float ripple_range: Ripple range
    :param float ripple_freq: Ripple frequency
    :param float ripple_kink: Ripple twistiness
    :param float|None with_aberration: Chromatic aberration distance
    :param float|None with_bloom: Bloom alpha
    :param bool with_dla: Diffusion-limited aggregation alpha
    :param int dla_padding: DLA pixel padding
    :param int point_freq: Voronoi and DLA point frequency (freq * freq = count)
    :param PointDistribution|ValueMask|int point_distrib: Voronoi and DLA point cloud distribution
    :param bool point_corners: Pin Voronoi and DLA points to corners (False = pin to center)
    :param int point_generations: Penrose-ish generations. Keep it low, and keep freq low, or you will run OOM easily.
    :param float point_drift: Fudge point locations (1.0 = nearest neighbor)
    :param None|int with_reverb: Reverb octave count
    :param int reverb_iterations: Re-reverberation N times
    :param bool reverb_ridges: Ridged reverb layers (False to disable)
    :param None|float with_light_leak: Light leak effect alpha
    :param None|float with_vignette: Vignette effect alpha
    :param None|float vignette_brightness: Vignette effect brightness
    :param None|float post_hue_rotation: Rotate hue (-.5 - .5)
    :param None|float post_saturation: Adjust saturation (0 - 1)
    :param None|float post_brightness: Adjust brightness
    :param None|float post_contrast: Adjust contrast
    :param bool with_ridge: Crease at midpoint values
    :param None|float with_shadow: Sobel-based shading alpha
    :param None|int with_jpeg_decimate: Conv2D feedback + JPEG encode/decode iteration count
    :param None|int with_conv_feedback: Conv2D feedback iterations
    :param float conv_feedback_alpha: Conv2D feedback alpha
    :param bool with_density_map: Map values to color histogram
    :param ValueMask|None with_glyph_map: Map values to glyph brightness. Square masks only for now
    :param bool glyph_map_colorize: Colorize glyphs from on average input colors
    :param float glyph_map_zoom: Scale glyph output
    :param float glyph_map_alpha: Fade glyph output
    :param None|ValueMask with_composite: Composite video effect
    :param float composite_zoom: Composite subpixel scaling
    :param bool with_sort: Pixel sort
    :param bool sort_angled: Pixel sort along a random angle
    :param bool sort_darkest: Pixel sort order by darkest instead of brightest
    :param None|list[str|ValueMask] convolve: List of ValueMasks to apply as convolution kernels
    :param bool with_sketch: Pencil sketch effect
    :param bool with_lowpoly: Low-poly art effect
    :param PointDistribution lowpoly_distrib: Point distribution for low-poly art effect
    :param int lowpoly_freq: Point frequency for low-poly art effect
    :param DistanceMetric lowpoly_metric: Low-poly effect distance metric
    :param None|float angle: Rotation angle
    :param None|bool with_simple_frame:
    :param None|int with_kaleido: Number of kaleido sides
    :param None|DistanceMetric kaleido_dist_metric: Kaleido center distance metric
    :param bool kaleido_blend_edges: Blend Kaleido with original edge indices
    :param None|float with_wobble: Move entire image around
    :param None|str with_palette: Apply named cosine palette
    :param bool with_glitch: Glitch effect (Bit shit)
    :param bool with_vhs: VHS effect (Shitty tracking)
    :param bool with_crt: Vintage TV effect
    :param bool with_scan_error: Horizontal scan error
    :param float with_snow: Analog broadcast snow
    :param float with_dither: Per-pixel brightness jitter
    :param bool with_frame: Shitty instant camera effect
    :param bool with_nebula: Add clouds
    :param bool with_false_color: Swap colors with basic noise
    :param bool with_watermark: Stylized digital watermark effect
    :param bool with_ticker: With spooky ticker effect
    :param bool with_scratches: Scratched film effect
    :param bool with_fibers: Old-timey paper fibers
    :param bool with_texture: Bumpy canvas
    :param bool with_pre_spatter: Spatter mask (early pass)
    :param bool with_spatter: Spatter mask
    :param bool with_clouds: Cloud cover
    :param None|float with_lens_warp: Lens warp effect
    :param None|float with_tint: Color tint effect alpha amount
    :param None|float with_degauss: CRT degauss effect
    :param bool rgb: Using RGB mode? Hint for some effects.
    :return: Tensor
    """

    tensor = value.normalize(tensor)

    if with_wobble:
        tensor = wobble(tensor, shape, time=time, speed=speed * with_wobble)

    if with_palette:
        tensor = palette(tensor, shape, with_palette, time=time)

    if (with_voronoi and with_voronoi != VoronoiDiagramType.none) or with_dla or with_kaleido:
        multiplier = max(2 * (point_generations - 1), 1)

        tiled_shape = [int(shape[0] / multiplier), int(shape[1] / multiplier), shape[2]]

        if point_freq == 1:
            x, y = point_cloud(1, PointDistribution.square, shape)

        else:
            x, y = point_cloud(point_freq, distrib=point_distrib, shape=tiled_shape, corners=point_corners, generations=point_generations,
                               drift=point_drift, time=time, speed=speed)

        xy = (x, y, len(x))

        input_tensor = value.resample(tensor, tiled_shape)

        if with_voronoi and with_voronoi != VoronoiDiagramType.none:
            input_tensor = voronoi(input_tensor, tiled_shape, alpha=voronoi_alpha, diagram_type=with_voronoi, dist_metric=voronoi_metric,
                                   inverse=voronoi_inverse, nth=voronoi_nth, ridges_hint=ridges_hint, with_refract=voronoi_refract,
                                   xy=xy, refract_y_from_offset=voronoi_refract_y_from_offset)

        if with_dla:
            input_tensor = value.blend(input_tensor, dla(input_tensor, tiled_shape, padding=dla_padding, xy=xy), with_dla)

        if point_generations == 1:
            tensor = input_tensor

        else:
            tensor = expand_tile(input_tensor, tiled_shape, shape)

    # Keep values between 0 and 1 if we're reflecting and refracting, because math?
    # Using refract and reflect together exposes unpleasant edge artifacting along
    # the natural edge where negative and positive offset values meet. It's normally
    # invisible to the human eye, but becomes obvious after extracting derivatives.
    signed_range = refract_signed_range and refract_range != 0 and reflect_range != 0

    if refract_range != 0:
        tensor = refract(tensor, shape, displacement=refract_range, signed_range=signed_range,
                         y_from_offset=refract_y_from_offset)

    if reflect_range != 0:
        tensor = refract(tensor, shape, displacement=reflect_range, from_derivative=True)

    if reindex_range != 0:
        tensor = reindex(tensor, shape, displacement=reindex_range)

    if clut:
        tensor = color_map(tensor, shape, clut=clut, horizontal=clut_horizontal, displacement=clut_range)

    if with_glyph_map:
        tensor = glyph_map(tensor, shape, mask=with_glyph_map, colorize=glyph_map_colorize, zoom=glyph_map_zoom,
                           alpha=glyph_map_alpha, time=time, speed=speed)

    if with_composite:
        tensor = glyph_map(tensor, shape, zoom=composite_zoom, mask=with_composite)

    if warp_range:
        if warp_interp is None:
            warp_interp = spline_order

        warp_freq = freq if warp_freq is None else warp_freq if isinstance(warp_freq, list) else value.freq_for_shape(warp_freq, shape)

        tensor = warp(tensor, shape, warp_freq, displacement=warp_range, octaves=warp_octaves, spline_order=warp_interp,
                      warp_map=warp_map, signed_range=warp_signed_range, time=time, speed=speed)

    if ripple_range:
        ripple_freq = freq if ripple_freq is None else ripple_freq if isinstance(ripple_freq, list) else value.freq_for_shape(ripple_freq, shape)

        tensor = ripple(tensor, shape, ripple_freq, displacement=ripple_range, kink=ripple_kink, time=time, speed=speed)

    if vortex_range:
        tensor = vortex(tensor, shape, displacement=vortex_range, time=time, speed=speed)

    if deriv and deriv != DistanceMetric.none:
        tensor = derivative(tensor, shape, deriv, alpha=deriv_alpha)

    if with_ridge:
        tensor = value.ridge(tensor)

    if posterize_levels:
        tensor = posterize(tensor, posterize_levels)

    if with_worms:
        tensor = worms(tensor, shape, behavior=with_worms, density=worms_density, duration=worms_duration,
                       stride=worms_stride, stride_deviation=worms_stride_deviation, alpha=worms_alpha, kink=worms_kink,
                       drunkenness=worms_drunkenness, drunken_spin=worms_drunken_spin, time=time, speed=speed)

    if with_wormhole:
        tensor = wormhole(tensor, shape, wormhole_kink, wormhole_stride, alpha=wormhole_alpha)

    if with_erosion_worms:
        tensor = erosion_worms(tensor, shape, density=erosion_worms_density, iterations=erosion_worms_iterations,
                               contraction=erosion_worms_contraction, alpha=erosion_worms_alpha, inverse=erosion_worms_inverse,
                               xy_blend=erosion_worms_xy_blend)

    if with_density_map:
        tensor = density_map(tensor, shape)

    if with_kaleido:
        tensor = kaleido(tensor, shape, with_kaleido, dist_metric=kaleido_dist_metric, xy=xy,
                         blend_edges=kaleido_blend_edges)

    if with_sobel and with_sobel != DistanceMetric.none:
        tensor = sobel(tensor, shape, with_sobel, rgb)

    if with_convolve:
        for kernel in with_convolve:
            if isinstance(kernel, str):
                kernel = ValueMask['conv2d_{}'.format(kernel)]

            tensor = convolve(kernel, tensor, shape)

    if with_shadow:
        tensor = shadow(tensor, shape, with_shadow)

    if with_outline and with_outline != DistanceMetric.none:
        tensor = outline(tensor, shape, sobel_metric=with_outline)

    if with_glowing_edges:
        tensor = glowing_edges(tensor, shape, alpha=with_glowing_edges)

    if with_reverb:
        tensor = reverb(tensor, shape, with_reverb, iterations=reverb_iterations, ridges=reverb_ridges)

    if with_aberration:
        tensor = aberration(tensor, shape, displacement=with_aberration, time=time, speed=speed)

    if with_bloom:
        tensor = bloom(tensor, shape, alpha=with_bloom)

    if with_light_leak:
        tensor = light_leak(tensor, shape, with_light_leak, time=time, speed=speed)

    if with_vignette:
        tensor = vignette(tensor, shape, brightness=vignette_brightness, alpha=with_vignette)

    if with_vaseline:
        tensor = vaseline(tensor, shape, alpha=with_vaseline)

    if with_normal_map:
        tensor = normal_map(tensor, shape)

    if post_hue_rotation not in (1.0, 0.0, None) and shape[2] == 3:
        tensor = tf.image.adjust_hue(tensor, post_hue_rotation)

    if post_saturation is not None:
        tensor = tf.image.adjust_saturation(tensor, post_saturation)

    if post_brightness is not None:
        tensor = tf.maximum(tf.minimum(tf.image.adjust_brightness(tensor, post_brightness), 1.0), -1.0)

    if post_contrast is not None:
        tensor = tf.maximum(tf.minimum(tf.image.adjust_contrast(tensor, post_contrast), 1.0), 0.0)

    if with_jpeg_decimate:
        tensor = jpeg_decimate(tensor, shape, iterations=with_jpeg_decimate)

    if with_conv_feedback:
        tensor = conv_feedback(tensor, shape, iterations=with_conv_feedback, alpha=conv_feedback_alpha)

    if with_sort:
        tensor = pixel_sort(tensor, shape, sort_angled, sort_darkest, time=time, speed=speed)

    if with_sketch:
        tensor = sketch(tensor, shape, time=time, speed=speed)

    if with_lowpoly:
        tensor = lowpoly(tensor, shape, distrib=lowpoly_distrib, freq=lowpoly_freq,
                         time=time, speed=speed, dist_metric=lowpoly_metric)

    if with_simple_frame:
        tensor = simple_frame(tensor, shape)

    if angle is not None:
        tensor = rotate(tensor, shape, angle)

    if with_pre_spatter:
        tensor = spatter(tensor, shape, time=time, speed=speed)

    if with_lens_warp:
        tensor = lens_warp(tensor, shape, displacement=with_lens_warp, time=time, speed=speed)

    if with_tint:
        tensor = tint(tensor, shape, alpha=with_tint, time=time, speed=speed)

    if with_nebula:
        tensor = nebula(tensor, shape, time=time, speed=speed)

    if with_false_color:
        tensor = false_color(tensor, shape, time=time, speed=speed)

    if with_glitch:
        tensor = glitch(tensor, shape, time=time, speed=speed)

    if with_dither:
        tensor = dither(tensor, shape, with_dither, time=time, speed=speed)

    if with_snow:
        tensor = snow(tensor, shape, with_snow, time=time, speed=speed)

    if with_scan_error:
        tensor = scanline_error(tensor, shape, time=time, speed=speed)

    if with_vhs:
        tensor = vhs(tensor, shape, time=time, speed=speed)

    if with_crt:
        tensor = crt(tensor, shape, time=time, speed=speed)

    if with_degauss:
        tensor = degauss(tensor, shape, displacement=with_degauss, time=time, speed=speed)

    if with_watermark:
        tensor = watermark(tensor, shape, time=time, speed=speed)

    if with_frame:
        tensor = frame(tensor, shape, time=time, speed=speed)

    if with_grime:
        tensor = grime(tensor, shape, time=time, speed=speed)

    if with_fibers:
        tensor = fibers(tensor, shape, time=time, speed=speed)

    if with_scratches:
        tensor = scratches(tensor, shape, time=time, speed=speed)

    if with_texture:
        tensor = texture(tensor, shape, time=time, speed=speed)

    if with_ticker:
        tensor = spooky_ticker(tensor, shape, time=time, speed=speed)

    if with_stray_hair:
        tensor = stray_hair(tensor, shape, time=time, speed=speed)

    if with_spatter:
        tensor = spatter(tensor, shape, time=time, speed=speed)

    if with_clouds:
        tensor = clouds(tensor, shape, time=time, speed=speed)

    tensor = value.normalize(tensor)

    return tensor


def effect(*args):
    """Function decorator for declaring composable effects."""

    def decorator_fn(func):
        argspec = inspect.getfullargspec(func)

        params = argspec.args
        # All effects respond to "tensor", "shape". Removing these non-keyword args should make params the same length as defaults.
        params.remove("tensor")
        params.remove("shape")

        if params and len(params) != len(argspec.defaults):
            raise ValueError(f'Expected {len(argspec.defaults)} keyword params to "{func.__name__}", but got {len(params)}.')

        # Register effect name and params
        name = args[0] if args else func.__name__
        EFFECTS[name] = dict((params[i], argspec.defaults[i]) for i in range(len(params)))
        EFFECTS[name]["func"] = func

        return func

    return decorator_fn


def _apply_effect(tensor, shape, name, time=0.0, speed=1.0, **kwargs):
    """Apply a composable effect to the given tensor."""

    if name not in EFFECTS:
        raise ValueError(f'"{name}" is not a registered effect name.')

    params = EFFECTS[name].copy()

    for k, v in kwargs.items():
        if k not in params:
            raise ValueError(f'Effect "{name}" does not have a parameter named "{k}"')

        params[k] = v

    if "time" in params:
        params["time"] = time
        params["speed"] = speed

    return params["func"](tensor, shape, **params)


def _conform_kernel_to_tensor(kernel, tensor, shape):
    """Re-shape a convolution kernel to match the given tensor's color dimensions."""

    values, _ = masks.mask_values(kernel)

    length = len(values)

    channels = shape[-1]

    temp = np.repeat(values, channels)

    temp = tf.reshape(temp, (length, length, channels, 1))

    temp = tf.cast(temp, tf.float32)

    temp /= tf.maximum(tf.reduce_max(temp), tf.reduce_min(temp) * -1)

    return temp


def convolve(kernel, tensor, shape, with_normalize=True, alpha=1.0):
    """
    Apply a convolution kernel to an image tensor.

    .. code-block:: python

       image = convolve(ValueMask.conv2d_shadow, image)

    :param ValueMask kernel: See conv2d_* members in ValueMask enum
    :param Tensor tensor: An image tensor.
    :param list[int] shape:
    :param bool with_normalize: Normalize output (True)
    :paral float alpha: Alpha blending amount
    :return: Tensor

    """

    height, width, channels = shape

    kernel_values = _conform_kernel_to_tensor(kernel, tensor, shape)

    # Give the conv kernel some room to play on the edges
    half_height = tf.cast(height / 2, tf.int32)
    half_width = tf.cast(width / 2, tf.int32)

    double_shape = [height * 2, width * 2, channels]

    out = tf.tile(tensor, [2, 2, 1])  # Tile 2x2

    out = value.offset(out, double_shape, half_width, half_height)

    out = tf.nn.depthwise_conv2d([out], kernel_values, [1, 1, 1, 1], "VALID")[0]

    out = tf.image.resize_with_crop_or_pad(out, height, width)

    if with_normalize:
        out = value.normalize(out)

    if kernel == ValueMask.conv2d_edges:
        out = tf.abs(out - .5) * 2

    if alpha == 1.0:
        return out

    return value.blend(tensor, out, alpha)


@effect()
def erosion_worms(tensor, shape, density=50, iterations=50, contraction=1.0, alpha=.25, inverse=False, xy_blend=False):
    """
    WIP hydraulic erosion effect.
    """

    # This will never be as good as
    # https://www.dropbox.com/s/kqv8b3w7o8ucbyi/Beyer%20-%20implementation%20of%20a%20methode%20for%20hydraulic%20erosion.pdf?dl=0

    height, width, channels = shape

    count = int(math.sqrt(height * width) * density)

    x = tf.random.uniform([count]) * (width - 1)
    y = tf.random.uniform([count]) * (height - 1)

    x_dir = tf.random.normal([count])
    y_dir = tf.random.normal([count])

    length = tf.sqrt(x_dir * x_dir + y_dir * y_dir)
    x_dir /= length
    y_dir /= length

    inertia = tf.random.normal([count], mean=0.75, stddev=0.25)

    out = tf.zeros(shape)

    # colors = tf.gather_nd(tensor, tf.cast(tf.stack([y, x], 1), tf.int32))

    values = value_map(convolve(ValueMask.conv2d_blur, tensor, shape), shape, keepdims=True)

    x_index = tf.cast(x, tf.int32)
    y_index = tf.cast(y, tf.int32)
    index = tf.stack([y_index, x_index], 1)
    starting_colors = tf.gather_nd(tensor, index)

    for i in range(iterations):
        x_index = tf.cast(x, tf.int32) % width
        y_index = tf.cast(y, tf.int32) % height
        index = tf.stack([y_index, x_index], 1)

        exposure = 1 - abs(1 - i / (iterations - 1) * 2)  # Makes linear gradient [ 0 .. 1 .. 0 ]
        out += tf.scatter_nd(index, starting_colors * exposure, shape)

        x1_index = (x_index + 1) % width
        y1_index = (y_index + 1) % height
        x1_values = tf.squeeze(tf.gather_nd(values, tf.stack([y_index, x1_index], 1)))
        y1_values = tf.squeeze(tf.gather_nd(values, tf.stack([y1_index, x_index], 1)))
        x1_y1_values = tf.squeeze(tf.gather_nd(values, tf.stack([y1_index, x1_index], 1)))

        u = x - tf.floor(x)
        v = y - tf.floor(y)

        sparse_values = tf.squeeze(tf.gather_nd(values, index))
        g_x = value.blend(y1_values - sparse_values, x1_y1_values - x1_values, u)
        g_y = value.blend(x1_values - sparse_values, x1_y1_values - y1_values, v)

        length = distance(g_x, g_y, DistanceMetric.euclidean) * contraction

        x_dir = value.blend(x_dir, g_x / length, inertia)
        y_dir = value.blend(y_dir, g_y / length, inertia)

        # step
        x = (x + x_dir) % width
        y = (y + y_dir) % height

    out = tf.maximum(tf.minimum(out, 1.0), 0.0)

    if inverse:
        out = 1.0 - out

    if xy_blend:
        tensor = value.blend(shadow(tensor, shape), reindex(tensor, shape, 1), xy_blend * values)

    return value.blend(tensor, out, alpha)


@effect()
def reindex(tensor, shape, displacement=.5):
    """
    Re-color the given tensor, by sampling along one axis at a specified frequency.

    .. image:: images/reindex.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor: An image tensor.
    :param list[int] shape:
    :param float displacement:
    :return: Tensor
    """

    height, width, channels = shape

    reference = value_map(tensor, shape)

    mod = min(height, width)
    x_offset = tf.cast((reference * displacement * mod + reference) % width, tf.int32)
    y_offset = tf.cast((reference * displacement * mod + reference) % height, tf.int32)

    tensor = tf.gather_nd(tensor, tf.stack([y_offset, x_offset], 2))

    return tensor


@effect()
def refract(tensor, shape, displacement=.5, reference_x=None, reference_y=None, warp_freq=None, spline_order=InterpolationType.bicubic,
            from_derivative=False, signed_range=True, time=0.0, speed=1.0, y_from_offset=False):
    """
    Apply displacement from pixel values.

    .. image:: images/refract.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor: An image tensor.
    :param list[int] shape:
    :param float displacement:
    :param Tensor reference_x: An optional horizontal displacement map.
    :param Tensor reference_y: An optional vertical displacement map.
    :param list[int] warp_freq: If given, generate new reference_x and reference_y noise with this base frequency.
    :param int spline_order: Ortho offset spline point count. 0=Constant, 1=Linear, 2=Cosine, 3=Bicubic
    :param bool from_derivative: If True, generate X and Y offsets from noise derivatives.
    :param bool signed_range: Scale displacement values from -1..1 instead of 0..1
    :param bool y_from_offset: If True, derive Y offsets from offsetting the image
    :return: Tensor
    """

    height, width, channels = shape

    x0_index = value.row_index(shape)
    y0_index = value.column_index(shape)

    warp_shape = None

    if warp_freq:
        warp_shape = [height, width, 1]

    if reference_x is None:
        if from_derivative:
            reference_x = convolve(ValueMask.conv2d_deriv_x, tensor, shape, with_normalize=False)

        elif warp_freq:
            reference_x = value.values(freq=warp_freq, shape=warp_shape, distrib=ValueDistribution.periodic_uniform,
                                       time=time, speed=speed, spline_order=spline_order)

        else:
            reference_x = tensor

    if reference_y is None:
        if from_derivative:
            reference_y = convolve(ValueMask.conv2d_deriv_y, tensor, shape, with_normalize=False)

        elif warp_freq:
            reference_y = value.values(freq=warp_freq, shape=warp_shape, distrib=ValueDistribution.periodic_uniform,
                                       time=time, speed=speed, spline_order=spline_order)

        else:
            if y_from_offset:
                # "the old way"
                y0_index += int(height * .5)
                x0_index += int(width * .5)
                reference_y = tf.gather_nd(reference_x, tf.stack([y0_index % height, x0_index % width], 2))
            else:
                reference_y = reference_x
                reference_x = tf.cos(reference_x * math.tau)
                reference_y = tf.sin(reference_y * math.tau)

    quad_directional = signed_range and not from_derivative

    # Use extended range so we can refract in 4 directions (-1..1) instead of 2 (0..1).
    # Doesn't work with derivatives (and isn't needed), because derivatives are signed naturally.
    x_offsets = value_map(reference_x, shape, signed_range=quad_directional, with_normalize=False) * displacement * width
    y_offsets = value_map(reference_y, shape, signed_range=quad_directional, with_normalize=False) * displacement * height
    # If not using extended range (0..1 instead of -1..1), keep the value range consistent.
    if not quad_directional:
        x_offsets *= 2.0
        y_offsets *= 2.0

    # Bilinear interpolation of midpoints
    x0_offsets = (tf.cast(x_offsets, tf.int32) + x0_index) % width
    x1_offsets = (x0_offsets + 1) % width
    y0_offsets = (tf.cast(y_offsets, tf.int32) + y0_index) % height
    y1_offsets = (y0_offsets + 1) % height

    x0_y0 = tf.gather_nd(tensor, tf.stack([y0_offsets, x0_offsets], 2))
    x1_y0 = tf.gather_nd(tensor, tf.stack([y0_offsets, x1_offsets], 2))
    x0_y1 = tf.gather_nd(tensor, tf.stack([y1_offsets, x0_offsets], 2))
    x1_y1 = tf.gather_nd(tensor, tf.stack([y1_offsets, x1_offsets], 2))

    x_fract = tf.reshape(x_offsets - tf.floor(x_offsets), [height, width, 1])
    y_fract = tf.reshape(y_offsets - tf.floor(y_offsets), [height, width, 1])

    x_y0 = value.blend(x0_y0, x1_y0, x_fract)
    x_y1 = value.blend(x0_y1, x1_y1, x_fract)

    return value.blend(x_y0, x_y1, y_fract)


@effect()
def ripple(tensor, shape, freq=2, displacement=1.0, kink=1.0, reference=None, spline_order=InterpolationType.bicubic, time=0.0, speed=1.0):
    """
    Apply displacement from pixel radian values.

    .. image:: images/ripple.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor: An image tensor.
    :param list[int] shape:
    :param list[int] freq: Displacement frequency
    :param float displacement:
    :param float kink:
    :param Tensor reference: An optional displacement map.
    :param int spline_order: Ortho offset spline point count. 0=Constant, 1=Linear, 2=Cosine, 3=Bicubic
    :return: Tensor
    """

    height, width, channels = shape

    x0_index = value.row_index(shape)
    y0_index = value.column_index(shape)

    value_shape = [shape[0], shape[1], 1]

    if reference is None:
        reference = value.values(freq=freq, shape=value_shape, distrib=ValueDistribution.periodic_uniform, spline_order=spline_order)

    # Twist index, borrowed from worms. TODO refactor me?
    index = value_map(reference, shape, with_normalize=False) * math.tau * kink * simplex.random(time, speed=speed)

    reference_x = (tf.cos(index) * displacement * width) % width
    reference_y = (tf.sin(index) * displacement * height) % height

    # Bilinear interpolation of midpoints, borrowed from refract(). TODO refactor me?
    x0_offsets = (tf.cast(reference_x, tf.int32) + x0_index) % width
    x1_offsets = (x0_offsets + 1) % width
    y0_offsets = (tf.cast(reference_y, tf.int32) + y0_index) % height
    y1_offsets = (y0_offsets + 1) % height

    x0_y0 = tf.gather_nd(tensor, tf.stack([y0_offsets, x0_offsets], 2))
    x1_y0 = tf.gather_nd(tensor, tf.stack([y0_offsets, x1_offsets], 2))
    x0_y1 = tf.gather_nd(tensor, tf.stack([y1_offsets, x0_offsets], 2))
    x1_y1 = tf.gather_nd(tensor, tf.stack([y1_offsets, x1_offsets], 2))

    x_fract = tf.reshape(reference_x - tf.floor(reference_x), [height, width, 1])
    y_fract = tf.reshape(reference_y - tf.floor(reference_y), [height, width, 1])

    x_y0 = value.blend(x0_y0, x1_y0, x_fract)
    x_y1 = value.blend(x0_y1, x1_y1, x_fract)

    return value.blend(x_y0, x_y1, y_fract)


@effect()
def color_map(tensor, shape, clut=None, horizontal=False, displacement=.5):
    """
    Apply a color map to an image tensor.

    The color map can be a photo or whatever else.

    .. image:: images/color_map.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :param list[int] shape:
    :param Tensor|str clut: An image tensor or filename (png/jpg only) to use as a color palette
    :param bool horizontal: Scan horizontally
    :param float displacement: Gather distance for clut
    """

    if isinstance(clut, str):
        clut = util.load(clut)

    height, width, channels = shape

    reference = value_map(tensor, shape) * displacement

    x_index = (value.row_index(shape) + tf.cast(reference * (width - 1), tf.int32)) % width

    if horizontal:
        y_index = value.column_index(shape)

    else:
        y_index = (value.column_index(shape) + tf.cast(reference * (height - 1), tf.int32)) % height

    index = tf.stack([y_index, x_index], 2)

    clut = value.resample(tf.image.convert_image_dtype(clut, tf.float32, saturate=True), shape)

    output = tf.gather_nd(clut, index)

    return output


@effect()
def worms(tensor, shape, behavior=1, density=4.0, duration=4.0, stride=1.0, stride_deviation=.05, alpha=.5, kink=1.0,
          drunkenness=0.0, drunken_spin=False, colors=None, time=0.0, speed=1.0):
    """
    Make a furry patch of worms which follow field flow rules.

    .. image:: images/worms.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :param list[int] shape:
    :param int|WormBehavior behavior:
    :param float density: Worm density multiplier (larger == slower)
    :param float duration: Iteration multiplier (larger == slower)
    :param float stride: Mean travel distance per iteration
    :param float stride_deviation: Per-worm travel distance deviation
    :param float alpha: Fade worms (0..1)
    :param float kink: Make your worms twist.
    :param float drunkenness: Randomly fudge angle at each step (1.0 = 360 degrees)
    :param bool drunken_spin: Worms are so drunk they're spinning. Someone hit the brakes!
    :param Tensor colors: Optional starting colors, if not from `tensor`.
    :return: Tensor
    """

    if isinstance(behavior, int):
        behavior = WormBehavior(behavior)

    height, width, channels = shape

    count = int(max(width, height) * density)

    if drunkenness:  # Get nearest power of 2, otherwise fastnoise will probably dump core
        c = 2

        while c < count:
            c *= 2

        count = c

    worms_y = tf.random.uniform([count]) * (height - 1)
    worms_x = tf.random.uniform([count]) * (width - 1)
    worms_stride = tf.random.normal([count], mean=stride, stddev=stride_deviation) * (max(width, height)/1024.0)

    color_source = colors if colors is not None else tensor

    colors = tf.gather_nd(color_source, tf.cast(tf.stack([worms_y, worms_x], 1), tf.int32))

    # For the benefit of drunk or meandering worms
    fastgen = fn.Noise()
    fastgen.frequency = count * .1

    quarter_count = int(count * .25)

    rots = {}

    rots = {
        WormBehavior.obedient: lambda n:
            tf.ones([n]) * random.random() * math.tau,

        WormBehavior.crosshatch: lambda n:
            rots[WormBehavior.obedient](n) + (tf.floor(tf.random.uniform([n]) * 100) % 4) * math.radians(90),

        WormBehavior.unruly: lambda n:
            rots[WormBehavior.obedient](n) + tf.random.uniform([n]) * .25 - .125,

        WormBehavior.chaotic: lambda n:
            tf.random.uniform([n]) * math.tau,

        WormBehavior.random: lambda _:
            tf.reshape(tf.stack([
                rots[WormBehavior.obedient](quarter_count),
                rots[WormBehavior.crosshatch](quarter_count),
                rots[WormBehavior.unruly](quarter_count),
                rots[WormBehavior.chaotic](quarter_count),
            ]), [count]),

        # Chaotic, changing over time
        WormBehavior.meandering: lambda n:
            (value.normalize(tf.stack(fastgen.genAsGrid([count], start=[int(min(shape[0], shape[1]) * time * speed)]))) * 2.0 - 1.0) * math.pi
    }

    worms_rot = rots[behavior](count)

    index = value_map(tensor, shape) * math.tau * kink

    iterations = int(math.sqrt(min(width, height)) * duration)

    out = tf.zeros(shape)

    scatter_shape = tf.shape(tensor)  # Might be different than `shape` due to clut

    if drunken_spin:
        start = int(min(shape[0], shape[1]) * time * speed)  # Just keep spinning in one direction over time

    # Make worms!
    for i in range(iterations):
        if drunkenness:
            if not drunken_spin:
                start = int(min(shape[0], shape[1]) * time * speed + i * speed * 10)  # Wobbling here and there

            worms_rot += (value.normalize(tf.stack(fastgen.genAsGrid([count], start=[start]))) * 2.0 - 1.0) * drunkenness * math.pi

        worm_positions = tf.cast(tf.stack([worms_y % height, worms_x % width], 1), tf.int32)

        exposure = 1 - abs(1 - i / (iterations - 1) * 2)  # Makes linear gradient [ 0 .. 1 .. 0 ]

        out += tf.scatter_nd(worm_positions, colors * exposure, scatter_shape)

        next_position = tf.gather_nd(index, worm_positions) + worms_rot

        worms_y = (worms_y + tf.cos(next_position) * worms_stride) % height
        worms_x = (worms_x + tf.sin(next_position) * worms_stride) % width

    out = tf.image.convert_image_dtype(out, tf.float32, saturate=True)

    return value.blend(tensor, tf.sqrt(value.normalize(out)), alpha)


@effect()
def wormhole(tensor, shape, kink=1.0, input_stride=1.0, alpha=1.0):
    """
    Apply per-pixel field flow. Non-iterative.

    .. image:: images/wormhole.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :param list[int] shape:
    :param float kink: Path twistiness
    :param float input_stride: Maximum pixel offset
    :return: Tensor
    """

    height, width, channels = shape

    values = value_map(tensor, shape, with_normalize=False)

    degrees = values * math.tau * kink
    # stride = values * height * input_stride
    stride = height * input_stride

    x_index = tf.cast(value.row_index(shape), tf.float32)
    y_index = tf.cast(value.column_index(shape), tf.float32)

    x_offset = (tf.cos(degrees) + 1) * stride
    y_offset = (tf.sin(degrees) + 1) * stride

    x = tf.cast(x_index + x_offset, tf.int32) % width
    y = tf.cast(y_index + y_offset, tf.int32) % height

    luminosity = tf.square(tf.reshape(values, [height, width, 1]))

    out = value.normalize(tf.scatter_nd(offset_index(y, height, x, width), tensor * luminosity, tf.shape(tensor)))

    return value.blend(tensor, tf.sqrt(out), alpha)


@effect()
def derivative(tensor, shape, dist_metric=DistanceMetric.euclidean, with_normalize=True, alpha=1.0):
    """
    Extract a derivative from the given noise.

    .. image:: images/derived.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :param list[int] shape:
    :param DistanceMetric|int dist_metric: Derivative distance metric
    :param bool with_normalize:
    :return: Tensor
    """

    x = convolve(ValueMask.conv2d_deriv_x, tensor, shape, with_normalize=False)
    y = convolve(ValueMask.conv2d_deriv_y, tensor, shape, with_normalize=False)

    out = distance(x, y, dist_metric)

    if with_normalize:
        out = value.normalize(out)

    if alpha == 1.0:
        return out

    return value.blend(tensor, out, alpha)


@effect("sobel")
def sobel_operator(tensor, shape, dist_metric=DistanceMetric.euclidean):
    """
    Apply a sobel operator.

    .. image:: images/sobel.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :param list[int] shape:
    :param DistanceMetric|int dist_metric: Sobel distance metric
    :return: Tensor
    """

    tensor = convolve(ValueMask.conv2d_blur, tensor, shape)

    x = convolve(ValueMask.conv2d_sobel_x, tensor, shape, with_normalize=False)
    y = convolve(ValueMask.conv2d_sobel_y, tensor, shape, with_normalize=False)

    out = tf.abs(value.normalize(distance(x, y, dist_metric)) * 2 - 1)

    fudge = -1

    out = value.offset(out, shape, x=fudge, y=fudge)

    return out


@effect()
def normal_map(tensor, shape):
    """
    Generate a tangent-space normal map.

    .. image:: images/normals.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :param list[int] shape:
    :return: Tensor
    """

    height, width, channels = shape

    reference = value_map(tensor, shape, keepdims=True)

    x = value.normalize(1 - convolve(ValueMask.conv2d_sobel_x, reference, [height, width, 1]))
    y = value.normalize(convolve(ValueMask.conv2d_sobel_y, reference, [height, width, 1]))

    z = 1 - tf.abs(value.normalize(tf.sqrt(x * x + y * y)) * 2 - 1) * .5 + .5

    return tf.stack([x[:, :, 0], y[:, :, 0], z[:, :, 0]], 2)


def value_map(tensor, shape, keepdims=False, signed_range=False, with_normalize=True):
    """
    Create a grayscale value map from the given image Tensor by reducing the sum across channels.

    Return value ranges between 0 and 1.

    :param Tensor tensor:
    :param list[int] shape:
    :param bool keepdims: If True, don't collapse the channel dimension.
    :param bool signed_range: If True, use an extended value range between -1 and 1.
    :return: Tensor of shape (height, width), or (height, width, channels) if keepdims was True.
    """

    tensor = tf.reduce_sum(tensor, len(shape) - 1, keepdims=keepdims)

    if with_normalize:
        tensor = value.normalize(tensor, signed_range=signed_range)

    elif signed_range:
        tensor = tensor * 2.0 - 1.0

    return tensor


@effect()
def density_map(tensor, shape):
    """
    Create a binned pixel value density map.

    .. image:: images/density.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :param list[int] shape:
    """

    height, width, channels = shape

    bins = max(height, width)

    # values = value_map(tensor, shape, keepdims=True)
    # values = tf.minimum(tf.maximum(tensor, 0.0), 1.0)  # TODO: Get this to work with HDR data
    values = value.normalize(tensor)

    # https://stackoverflow.com/a/34143927
    binned_values = tf.cast(tf.reshape(values * (bins - 1), [-1]), tf.int32)
    ones = tf.ones_like(binned_values, dtype=tf.int32)
    counts = tf.math.unsorted_segment_sum(ones, binned_values, bins)

    out = tf.gather(counts, tf.cast(values[:, :] * (bins - 1), tf.int32))

    return tf.ones(shape) * value.normalize(tf.cast(out, tf.float32))


@effect()
def jpeg_decimate(tensor, shape, iterations=25):
    """
    Destroy an image with the power of JPEG

    :param Tensor tensor:
    :return: Tensor
    """

    jpegged = tensor

    for i in range(iterations):
        jpegged = tf.image.convert_image_dtype(jpegged, tf.uint8)

        data = tf.image.encode_jpeg(jpegged, quality=random.randint(5, 50), x_density=random.randint(50, 500), y_density=random.randint(50, 500))
        jpegged = tf.image.decode_jpeg(data)

        jpegged = tf.image.convert_image_dtype(jpegged, tf.float32, saturate=True)

    return jpegged


@effect()
def conv_feedback(tensor, shape, iterations=50, alpha=.5):
    """
    Conv2d feedback loop

    :param Tensor tensor:
    :return: Tensor
    """

    iterations = 100

    half_shape = [int(shape[0] * .5), int(shape[1] * .5), shape[2]]

    convolved = value.proportional_downsample(tensor, shape, half_shape)

    for i in range(iterations):
        convolved = convolve(ValueMask.conv2d_blur, convolved, half_shape)
        convolved = convolve(ValueMask.conv2d_sharpen, convolved, half_shape)

    convolved = value.normalize(convolved)

    up = tf.maximum((convolved - .5) * 2, 0.0)

    down = tf.minimum(convolved * 2, 1.0)

    return value.blend(tensor, value.resample(up + (1.0 - down), shape), alpha)


def distance(a, b, metric):
    """
    Compute the distance from a to b, using the specified metric.

    :param Tensor a:
    :param Tensor b:
    :param DistanceMetric|int|str metric: Distance metric
    :return: Tensor
    """

    if isinstance(metric, int):
        metric = DistanceMetric(metric)

    elif isinstance(metric, str):
        metric = DistanceMetric[metric]

    if metric == DistanceMetric.euclidean:
        dist = tf.sqrt(a * a + b * b)

    elif metric == DistanceMetric.manhattan:
        dist = tf.abs(a) + tf.abs(b)

    elif metric == DistanceMetric.chebyshev:
        dist = tf.maximum(tf.abs(a), tf.abs(b))

    elif metric == DistanceMetric.octagram:
        dist = tf.maximum((tf.abs(a) + tf.abs(b)) / math.sqrt(2), tf.maximum(tf.abs(a), tf.abs(b)))

    elif metric == DistanceMetric.triangular:
        dist = tf.maximum(tf.abs(a) - b * .5, b)

    elif metric == DistanceMetric.hexagram:
        dist = tf.maximum(
            tf.maximum(tf.abs(a) - b * .5, b),
            tf.maximum(tf.abs(a) - b * -.5, b * -1)
        )

    else:
        raise ValueError("{0} isn't a distance metric.".format(metric))

    return dist


def blend_layers(control, shape, feather=1.0, *layers):
    layer_count = len(layers)

    control = value.normalize(control)

    control *= layer_count
    control_floor = tf.cast(control, tf.int32)

    x_index = value.row_index(shape)
    y_index = value.column_index(shape)

    layers = tf.stack(list(layers) + [layers[-1]])
    layer_count += 1

    floor_values = control_floor[:, :, 0]

    # I'm not sure why the mod operation is needed, but tensorflow-cpu explodes without it.
    combined_layer_0 = tf.gather_nd(layers, tf.stack([floor_values % layer_count, y_index, x_index], 2))
    combined_layer_1 = tf.gather_nd(layers, tf.stack([(floor_values + 1) % layer_count, y_index, x_index], 2))

    control_floor_fract = control - tf.floor(control)
    control_floor_fract = tf.minimum(tf.maximum(control_floor_fract - (1.0 - feather), 0.0) / feather, 1.0)

    return value.blend(combined_layer_0, combined_layer_1, control_floor_fract)


def center_mask(center, edges, shape, power=2):
    """
    Blend two image tensors from the center to the edges.

    :param Tensor center:
    :param Tensor edges:
    :param list[int] shape:
    :param int power:
    :return: Tensor
    """

    mask = tf.pow(singularity(None, shape, dist_metric=DistanceMetric.chebyshev), power)

    return value.blend(center, edges, mask)


@effect()
def voronoi(tensor, shape, diagram_type=VoronoiDiagramType.range, density=.1, nth=0,
            dist_metric=DistanceMetric.euclidean, alpha=1.0, with_refract=0.0, inverse=False,
            xy=None, ridges_hint=False, image_count=None, refract_y_from_offset=True):
    """
    Create a voronoi diagram, blending with input image Tensor color values.

    .. image:: images/voronoi.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :param list[int] shape:
    :param VoronoiDiagramType|int diagram_type: Diagram type (0=Off, 1=Range, 2=Color Range, 3=Indexed, 4=Color Map, 5=Blended, 6=Flow)
    :param float nth: Plot Nth nearest neighbor, or -Nth farthest
    :param DistanceMetric|int dist_metric: Voronoi distance metric
    :param bool regions: Assign colors to control points (memory intensive)
    :param float alpha: Blend with original tensor (0.0 = Original, 1.0 = Voronoi)
    :param float with_refract: Domain warp input tensor against resulting voronoi
    :param bool inverse: Invert range brightness values (does not affect hue)
    :param (Tensor, Tensor, int) xy: Bring your own x, y, and point count (You shouldn't normally need this)
    :param float ridges_hint: Adjust output colors to match ridged multifractal output (You shouldn't normally need this)
    :return: Tensor
    """

    if isinstance(diagram_type, int):
        diagram_type = VoronoiDiagramType(diagram_type)

    elif isinstance(diagram_type, str):
        diagram_type = VoronoiDiagramType[diagram_type]

    original_shape = shape

    shape = [int(shape[0] * .5), int(shape[1] * .5), shape[2]]  # Gotta upsample later, this one devours memory.

    height, width, channels = shape

    if xy is None:
        point_count = int(min(width, height) * density)

        x = tf.random.uniform([point_count]) * width
        y = tf.random.uniform([point_count]) * height

    else:
        if len(xy) == 2:
            x, y = xy
            point_count = len(x)

        else:
            x, y, point_count = xy

        x = tf.cast(tf.stack(x), tf.float32) / 2.0
        y = tf.cast(tf.stack(y), tf.float32) / 2.0

    value_shape = [height, width, 1]

    x_index = tf.cast(tf.reshape(value.row_index(shape), value_shape), tf.float32)
    y_index = tf.cast(tf.reshape(value.column_index(shape), value_shape), tf.float32)

    is_triangular = dist_metric in (
        DistanceMetric.triangular,
        DistanceMetric.triangular.name,
        DistanceMetric.triangular.value,
        DistanceMetric.hexagram,
        DistanceMetric.hexagram.name,
        DistanceMetric.hexagram.value,
    )

    if diagram_type in VoronoiDiagramType.flow_members():
        # If we're using flow with a perfectly tiled grid, it just disappears. Perturbing the points seems to prevent this from happening.
        x += tf.random.normal(shape=tf.shape(x), stddev=.0001, dtype=tf.float32)
        y += tf.random.normal(shape=tf.shape(y), stddev=.0001, dtype=tf.float32)

    if is_triangular:
        # Keep it visually flipped "horizontal"-side-up
        y_sign = -1.0 if inverse else 1.0

        dist = distance((x_index - x) / width, (y_index - y) * y_sign / height, dist_metric)

    else:
        half_width = int(width * .5)
        half_height = int(height * .5)

        # Wrapping edges! Nearest neighbors might be actually be "wrapped around", on the opposite side of the image.
        # Determine which direction is closer, and use the minimum.

        # Subtracting the list of points from the index results in a new shape
        # [y, x, value] - [point_count] -> [y, x, value, point_count]
        x0_diff = x_index - x - half_width
        x1_diff = x_index - x + half_width
        y0_diff = y_index - y - half_height
        y1_diff = y_index - y + half_height

        x_diff = tf.minimum(tf.abs(x0_diff), tf.abs(x1_diff)) / width
        y_diff = tf.minimum(tf.abs(y0_diff), tf.abs(y1_diff)) / height

        # Not-wrapping edges!
        # x_diff = (x_index - x) / width
        # y_diff = (y_index - y) / height

        dist = distance(x_diff, y_diff, dist_metric)

    ###
    if diagram_type not in VoronoiDiagramType.flow_members():
        dist, indices = tf.nn.top_k(dist, k=point_count)
        index = int((nth + 1) * -1)

    ###

    # Seamless alg offset pixels by half image size. Move results slice back to starting points with `offset`:
    offset_kwargs = {
        'x': 0.0 if is_triangular else half_width,
        'y': 0.0 if is_triangular else half_height,
    }

    if diagram_type in (VoronoiDiagramType.range, VoronoiDiagramType.color_range, VoronoiDiagramType.range_regions):
        range_slice = value.resample(value.offset(tf.expand_dims(tf.sqrt(value.normalize(dist[:, :, index])), -1), shape, **offset_kwargs), original_shape)

        if inverse:
            range_slice = 1.0 - range_slice

    if diagram_type in (VoronoiDiagramType.regions, VoronoiDiagramType.color_regions, VoronoiDiagramType.range_regions):
        regions_slice = value.offset(indices[:, :, index], shape, **offset_kwargs)

    ###
    if diagram_type == VoronoiDiagramType.range:
        range_out = range_slice

    if diagram_type in VoronoiDiagramType.flow_members():
        dist = tf.math.log(dist)

        # Clamp to avoid infinities
        dist = tf.minimum(10, dist)
        dist = tf.maximum(-10, dist)

        dist = tf.expand_dims(dist, -1)

        if diagram_type == VoronoiDiagramType.color_flow:
            colors = tf.gather_nd(tensor, tf.cast(tf.stack([y * 2, x * 2], 1), tf.int32))
            colors = tf.reshape(colors, [1, 1, point_count, shape[2]])
            if ridges_hint:
                colors = tf.abs(colors * 2 - 1)
            # normalize() can make animation twitchy. TODO: figure out a way to do this without normalize
            dist = 1.0 - ((1.0 - value.normalize(dist)) * colors)

        range_out = tf.math.reduce_sum(dist, 2) / point_count

        range_out = value.resample(value.offset(range_out, shape, **offset_kwargs), original_shape)

        if inverse:
            range_out = 1.0 - range_out

    if diagram_type in (VoronoiDiagramType.color_range, VoronoiDiagramType.range_regions):
        # range_out = regions_out * range_slice
        range_out = value.blend(tensor * range_slice, range_slice, range_slice)

    if diagram_type == VoronoiDiagramType.regions:
        regions_out = value.resample(tf.cast(regions_slice, tf.float32), original_shape, spline_order=InterpolationType.constant)

    if diagram_type in (VoronoiDiagramType.color_regions, VoronoiDiagramType.range_regions):
        colors = tf.gather_nd(tensor, tf.cast(tf.stack([y * 2, x * 2], 1), tf.int32))

        if ridges_hint:
            colors = tf.abs(colors * 2 - 1)

        spline_order = 0 if diagram_type == VoronoiDiagramType.color_regions else 3

        regions_out = value.resample(tf.reshape(tf.gather(colors, regions_slice), shape), original_shape, spline_order=spline_order)

    ###
    if diagram_type == VoronoiDiagramType.range_regions:
        out = value.blend(regions_out, range_out, tf.square(range_out))

    elif diagram_type in [VoronoiDiagramType.range, VoronoiDiagramType.color_range] + VoronoiDiagramType.flow_members():
        out = range_out

    elif diagram_type in (VoronoiDiagramType.regions, VoronoiDiagramType.color_regions):
        out = regions_out

    else:
        raise Exception(f"Not sure what to do with diagram type {diagram_type}")

    if diagram_type == VoronoiDiagramType.regions:
        out = tf.expand_dims(out, -1)

    if with_refract != 0.0:
        out = refract(tensor, original_shape, displacement=with_refract, reference_x=out,
                      y_from_offset=refract_y_from_offset)

    if tensor is not None:
        out = value.blend(tensor, out, alpha)

    return out


@effect()
def posterize(tensor, shape, levels=9):
    """
    Reduce the number of color levels per channel.

    .. image:: images/posterize.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :param int levels:
    :return: Tensor
    """

    tensor *= levels

    tensor += (1/levels) * .5

    tensor = tf.floor(tensor)

    tensor /= levels

    return tensor


def inner_tile(tensor, shape, freq):
    """
    """

    if isinstance(freq, int):
        freq = value.freq_for_shape(freq, shape)

    small_shape = [int(shape[0] / freq[0]), int(shape[1] / freq[1]), shape[2]]

    y_index = tf.tile(value.column_index(small_shape) * freq[0], [freq[0], freq[0]])
    x_index = tf.tile(value.row_index(small_shape) * freq[1], [freq[0], freq[0]])

    tiled = tf.gather_nd(tensor, tf.stack([y_index, x_index], 2))

    tiled = value.resample(tiled, shape, spline_order=InterpolationType.linear)

    return tiled


def expand_tile(tensor, input_shape, output_shape, with_offset=True):
    """
    """

    input_width = input_shape[1]
    input_height = input_shape[0]

    if with_offset:
        x_offset = tf.cast(input_shape[1] / 2, tf.int32)
        y_offset = tf.cast(input_shape[0] / 2, tf.int32)

    else:
        x_offset = 0
        y_offset = 0

    x_index = (x_offset + value.row_index(output_shape)) % input_width
    y_index = (y_offset + value.column_index(output_shape)) % input_height

    return tf.gather_nd(tensor, tf.stack([y_index, x_index], 2))


def offset_index(y_index, height, x_index, width):
    """
    Offset X and Y displacement channels from each other, to help with diagonal banding.

    Returns a combined Tensor with shape [height, width, 2]

    :param Tensor y_index: Tensor with shape [height, width, 1], containing Y indices
    :param int height:
    :param Tensor x_index: Tensor with shape [height, width, 1], containing X indices
    :param int width:
    :return: Tensor
    """

    index = tf.stack([
        (y_index + int(height * .5 + random.random() * height * .5)) % height,
        (x_index + int(random.random() * width * .5)) % width,
        ], 2)

    return tf.cast(index, tf.int32)


@effect()
def warp(tensor, shape, freq=2, octaves=5, displacement=1, spline_order=InterpolationType.bicubic, warp_map=None, signed_range=True, time=0.0, speed=1.0):
    """
    Multi-octave warp effect

    .. image:: images/warp.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :param list[int] shape:
    :param list[int] freq:
    :param int octaves:
    :param float displacement:
    :param int spline_order:
    :param str|None warp_map:
    :param bool signed_range:
    :param float time:
    :param float speed:
    """

    for octave in range(1, octaves + 1):
        multiplier = 2 ** octave

        base_freq = [int(f * .5 * multiplier) for f in freq]

        if base_freq[0] >= shape[0] or base_freq[1] >= shape[1]:
            break

        kwargs = {}

        if warp_map is not None:
            if isinstance(warp_map, str):
                warp_map = tf.image.convert_image_dtype(util.load(warp_map), tf.float32)

            kwargs["reference_x"] = warp_map
        else:
            kwargs["warp_freq"] = base_freq

        tensor = refract(tensor, shape, displacement=displacement / multiplier,
                         spline_order=spline_order, signed_range=signed_range, time=time, speed=speed, **kwargs)

    return tensor


def sobel(tensor, shape, dist_metric=1, rgb=False):
    """
    Colorized sobel edges.

    :param Tensor tensor:
    :param list[int] shape:
    :param DistanceMetric|int dist_metric: Sobel distance metric
    :param bool rgb:
    """

    if rgb:
        return sobel_operator(tensor, shape, dist_metric)

    else:
        return outline(tensor, shape, dist_metric, True)


@effect()
def outline(tensor, shape, sobel_metric=1, invert=False):
    """
    Superimpose sobel operator results (cartoon edges)

    :param Tensor tensor:
    :param list[int] shape:
    :param DistanceMetric|int sobel_metric: Sobel distance metric
    """

    height, width, channels = shape

    value_shape = [height, width, 1]

    values = value_map(tensor, shape, keepdims=True)

    edges = sobel_operator(values, value_shape, dist_metric=sobel_metric)

    if invert:
        edges = 1 - edges

    return edges * tensor


@effect()
def glowing_edges(tensor, shape, sobel_metric=2, alpha=1.0):
    """
    """

    height, width, channels = shape

    value_shape = [height, width, 1]

    edges = value_map(tensor, shape, keepdims=True)

    edges = posterize(edges, random.randint(3, 5))

    edges = 1.0 - sobel_operator(edges, value_shape, dist_metric=sobel_metric)

    edges = tf.minimum(edges * 8, 1.0) * tf.minimum(tensor * 1.25, 1.0)

    edges = bloom(edges, shape, alpha=.5)

    edges = value.normalize(edges + convolve(ValueMask.conv2d_blur, edges, shape))

    return value.blend(tensor, 1.0 - ((1.0 - edges) * (1.0 - tensor)), alpha)


def singularity(tensor, shape, diagram_type=VoronoiDiagramType.range, **kwargs):
    """
    Return the range diagram for a single voronoi point, approximately centered.

    :param Tensor tensor:
    :param list[int] shape:
    :param VoronoiDiagramType|int diagram_type:
    :param DistanceMetric|int dist_metric:

    Additional kwargs will be sent to the `voronoi` metric.
    """

    x, y = point_cloud(1, PointDistribution.square, shape)

    return voronoi(tensor, shape, diagram_type=diagram_type, xy=(x, y, 1), **kwargs)


@effect()
def vortex(tensor, shape, displacement=64.0, time=0.0, speed=1.0):
    """
    Vortex tiling effect

    .. image:: images/vortex.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :param list[int] shape:
    :param float displacement:
    """

    value_shape = [shape[0], shape[1], 1]

    displacement_map = singularity(None, value_shape)
    displacement_map = value.normalize(displacement_map)

    x = convolve(ValueMask.conv2d_deriv_x, displacement_map, value_shape, with_normalize=False)
    y = convolve(ValueMask.conv2d_deriv_y, displacement_map, value_shape, with_normalize=False)

    fader = singularity(None, value_shape, dist_metric=DistanceMetric.chebyshev, inverse=True)
    fader = value.normalize(fader)

    x *= fader
    y *= fader

    warped = refract(tensor, shape,
                     displacement=simplex.random(time, speed=speed) * 100 * displacement,
                     reference_x=x, reference_y=y, signed_range=False)

    return warped


@effect()
def aberration(tensor, shape, displacement=.005, time=0.0, speed=1.0):
    """
    Chromatic aberration

    .. image:: images/aberration.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :param list[int] shape:
    :param float displacement:
    """

    height, width, channels = shape

    if channels != 3:  # aye cannit doit
        return tensor

    x_index = value.row_index(shape)
    y_index = value.column_index(shape)

    x_index_float = tf.cast(x_index, tf.float32)

    separated = []

    displacement_pixels = int(width * displacement * simplex.random(time, speed=speed))

    mask = tf.pow(tf.squeeze(singularity(None, [shape[0], shape[1], 1])), 3)

    gradient = value.normalize(x_index_float)

    shift = random.random() * .1 - .05
    tensor = tf.image.adjust_hue(tensor, shift)

    for i in range(channels):
        # Left and right neighbor pixels
        if i == 0:
            # Left (red)
            offset_x_index = tf.minimum(x_index + displacement_pixels, width - 1)

        elif i == 1:
            # Center (green)
            offset_x_index = x_index

        elif i == 2:
            # Right (blue)
            offset_x_index = tf.maximum(x_index - displacement_pixels, 0)

        # return tf.expand_dims(offset_x_index, axis=2)
        offset_x_index = tf.cast(offset_x_index, tf.float32)

        # Left and right image sides
        if i == 0:
            # Left (red)
            offset_x_index = value.blend(offset_x_index, x_index_float, gradient)

        elif i == 2:
            # Right (blue)
            offset_x_index = value.blend(x_index_float, offset_x_index, gradient)

        # Fade effect towards center
        offset_x_index = tf.cast(value.blend_cosine(x_index_float, offset_x_index, mask), tf.int32)

        separated.append(tf.gather_nd(tensor[:, :, i], tf.stack([y_index, offset_x_index], 2)))

    tensor = tf.stack(separated, 2)

    # Restore original colors
    return tf.image.adjust_hue(tensor, -shift)


@effect()
def bloom(tensor, shape, alpha=.5):
    """
    Bloom effect

    Input image must currently be square (sorry).

    :param Tensor tensor:
    :param list[int] shape:
    :param float alpha:
    """

    height, width, channels = shape

    blurred = tf.maximum(tensor * 2.0 - 1.0, 0.0)
    blurred = value.proportional_downsample(blurred, shape, [max(int(height * .01), 1), max(int(width * .01), 1), channels]) * 4.0
    blurred = value.resample(blurred, shape)
    blurred = value.offset(blurred, shape, x=int(shape[1] * -.05), y=int(shape[0] * -.05))

    return value.blend(tensor, 1.0 - (1.0 - tensor) * (1.0 - blurred), alpha)


@effect()
def dla(tensor, shape, padding=2, seed_density=.01, density=.125, xy=None):
    """
    Diffusion-limited aggregation. Slow.

    .. image:: images/dla.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :param list[int] shape:
    :param int padding:
    :param float seed_density:
    :param float density:
    :param None|Tensor xy: Pre-seeded point cloud (optional)
    """

    height, width, channels = shape

    # Nearest-neighbor map for affixed nodes, lets us miss with one lookup instead of eight
    neighborhoods = set()

    # Nearest-neighbor map of neighbor map, lets us skip nodes which are too far away to matter
    expanded_neighborhoods = set()

    # Actual affixed nodes
    clustered = []

    # Not-affixed nodes
    walkers = []

    scale = 1 / padding

    half_width = int(width * scale)
    half_height = int(height * scale)

    if xy is None:
        seed_count = int(half_height * seed_density) or 1
        x, y = point_cloud(int(math.sqrt(seed_count)), distrib=PointDistribution.random, shape=shape)

    else:
        x, y, seed_count = xy

    walkers_count = half_height * half_width * density

    walkers_per_seed = int(walkers_count / seed_count)

    offsets = [-1, 0, 1]

    expanded_range = 8

    expanded_offsets = range(-expanded_range, expanded_range + 1)

    for i in range(seed_count):
        node = (int(y[i] * scale), int(x[i] * scale))

        clustered.append(node)

        for x_offset in offsets:
            for y_offset in offsets:
                neighborhoods.add((node[0] + y_offset, node[1] + x_offset))

        for x_offset in expanded_offsets:
            for y_offset in expanded_offsets:
                expanded_neighborhoods.add((node[0] + y_offset, node[1] + x_offset))

        for i in range(walkers_per_seed):
            # degrees = 360.0 * math.radians(1) * random.random()
            # dist = random.random() * height / math.sqrt(seed_count) * 2.5
            # walkers.append((node[0] + int(math.cos(degrees) * dist), node[1] + int(math.sin(degrees) * dist)))

            walkers.append((int(random.random() * half_height), int(random.random() * half_width)))

    iterations = 2000

    for i in range(iterations):
        remove_walkers = set()

        for walker in walkers:
            if walker in neighborhoods:
                remove_walkers.add(walker)

        # Remove all occurrences
        walkers = [walker for walker in walkers if walker not in remove_walkers]

        for walker in remove_walkers:
            for x_offset in offsets:
                for y_offset in offsets:
                    # tensorflowification - use a conv2d here
                    neighborhoods.add(((walker[0] + y_offset) % half_height, (walker[1] + x_offset) % half_width))

            for x_offset in expanded_offsets:
                for y_offset in expanded_offsets:
                    expanded_neighborhoods.add(((walker[0] + y_offset) % half_height, (walker[1] + x_offset) % half_width))

            clustered.append(walker)

        if not walkers:
            break

        for w in range(len(walkers)):
            walker = walkers[w]

            if walker in expanded_neighborhoods:
                walkers[w] = ((walker[0] + offsets[random.randint(0, len(offsets) - 1)]) % half_height,
                              (walker[1] + offsets[random.randint(0, len(offsets) - 1)]) % half_width)

            else:
                walkers[w] = ((walker[0] + expanded_offsets[random.randint(0, len(expanded_offsets) - 1)]) % half_height,
                              (walker[1] + expanded_offsets[random.randint(0, len(expanded_offsets) - 1)]) % half_width)

    seen = set()
    unique = []

    for c in clustered:
        if c in seen:
            continue

        seen.add(c)

        unique.append(c)

    count = len(unique)

    # hot = tf.ones([count, channels])
    hot = tf.ones([count, channels]) * tf.cast(tf.reshape(tf.stack(list(reversed(range(count)))), [count, 1]), tf.float32)

    out = convolve(ValueMask.conv2d_blur, tf.scatter_nd(tf.stack(unique) * int(1/scale), hot, [height, width, channels]), shape)

    return out * tensor


@effect()
def wobble(tensor, shape, time=0.0, speed=1.0):
    """
    Move the entire image around
    """

    x_offset = tf.cast(simplex.random(time=time, speed=speed) * shape[1], tf.int32)
    y_offset = tf.cast(simplex.random(time=time, speed=speed) * shape[0], tf.int32)

    return value.offset(tensor, shape, x=x_offset, y=y_offset)


@effect()
def reverb(tensor, shape, octaves=2, iterations=1, ridges=True):
    """
    Multi-octave "reverberation" of input image tensor

    :param Tensor tensor:
    :param float[int] shape:
    :param int octaves:
    :param int iterations: Re-reverberate N times. Gratuitous!
    :param bool ridges: abs(tensor * 2 - 1) -- False to not do that.
    """

    height, width, channels = shape

    if ridges:
        reference = 1.0 - tf.abs(tensor * 2 - 1)

    else:
        reference = tensor

    out = reference

    for i in range(iterations):
        for octave in range(1, octaves + 1):
            multiplier = 2 ** octave

            octave_shape = [int(width / multiplier), int(height / multiplier), channels]

            if not all(octave_shape):
                break

            out += expand_tile(value.proportional_downsample(reference, shape, octave_shape), octave_shape, shape) / multiplier

    return value.normalize(out)


@effect()
def light_leak(tensor, shape, alpha=.25, time=0.0, speed=1.0):
    """
    """

    x, y = point_cloud(6, distrib=PointDistribution.grid_members()[random.randint(0, len(PointDistribution.grid_members()) - 1)],
                       drift=.05, shape=shape, time=time, speed=speed)

    leak = voronoi(tensor, shape, diagram_type=VoronoiDiagramType.color_regions, xy=(x, y, len(x)))
    leak = wormhole(leak, shape, kink=1.0, input_stride=.25)

    leak = bloom(leak, shape, 1.0)

    leak = 1 - ((1 - tensor) * (1 - leak))

    leak = center_mask(tensor, leak, shape, 4)

    return vaseline(value.blend(tensor, leak, alpha), shape, alpha)


@effect()
def vignette(tensor, shape, brightness=0.0, alpha=1.0):
    """
    """

    tensor = value.normalize(tensor)

    edges = center_mask(tensor, tf.ones(shape) * brightness, shape)

    return value.blend(tensor, edges, alpha)


@effect()
def vaseline(tensor, shape, alpha=1.0):
    """
    """

    return value.blend(tensor, center_mask(tensor, bloom(tensor, shape, 1.0), shape), alpha)


@effect()
def shadow(tensor, shape, alpha=1.0, reference=None):
    """
    Convolution-based self-shadowing effect.

    .. image:: images/shadow.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :param list[int] shape:
    :param float alpha:
    :param None|Tensor reference: Alternate reference values with shape (height, width)
    """

    height, width, channels = shape

    if reference is None:
        reference = tensor

    reference = value_map(reference, shape, keepdims=True)

    value_shape = [height, width, 1]

    x = convolve(ValueMask.conv2d_sobel_x, reference, value_shape)
    y = convolve(ValueMask.conv2d_sobel_y, reference, value_shape)

    shade = value.normalize(distance(x, y, DistanceMetric.euclidean))

    shade = convolve(ValueMask.conv2d_sharpen, shade, value_shape, alpha=.5)

    # Ramp values to not be so imposing visually
    highlight = tf.math.square(shade)

    # Darken and brighten original pixel values
    shade = (1.0 - ((1.0 - tensor) * (1.0 - highlight))) * shade

    # Limit effect to just the brightness channel
    tensor = tf.image.rgb_to_hsv([tensor])[0]

    tensor = tf.stack([tensor[:, :, 0], tensor[:, :, 1],
                       value.blend(tensor[:, :, 2], tf.image.rgb_to_hsv([shade])[0][:, :, 2], alpha)], 2)

    return tf.image.hsv_to_rgb([tensor])[0]


@effect()
def glyph_map(tensor, shape, mask=None, colorize=True, zoom=1, alpha=1.0, time=0.0, speed=1.0):
    """
    :param Tensor tensor:
    :param list[int] shape:
    :param ValueMask|None mask:
    """

    if mask is None:
        mask = ValueMask.truetype

    elif isinstance(mask, int):
        mask = ValueMask(mask)

    elif isinstance(mask, str):
        mask = ValueMask[mask]

    if mask == ValueMask.truetype:
        glyph_shape = masks.mask_shape(ValueMask.truetype)
        glyphs = load_glyphs(glyph_shape)

    else:
        glyph_shape = masks.mask_shape(mask)

        glyphs = []
        sums = []

        levels = 100
        for i in range(levels):
            # Generate some glyphs.
            glyph, brightness = masks.mask_values(mask, glyph_shape, uv_noise=np.ones(glyph_shape) * i / levels, atlas=masks.get_atlas(mask))

            glyphs.append(glyph)
            sums.append(brightness)

        glyphs = [g for _, g in sorted(zip(sums, glyphs))]

    in_shape = [int(shape[0] / zoom), int(shape[1] / zoom), shape[2]]

    height, width, channels = in_shape

    # Figure out how many glyphs it will take approximately to cover the image
    uv_shape = [int(in_shape[0] / glyph_shape[0]) or 1, int(in_shape[1] / glyph_shape[1] or 1), 1]

    # Generate a value map, multiply by len(glyphs) to create glyph index offsets
    value_shape = [height, width, 1]
    uv_noise = value.proportional_downsample(value_map(tensor, in_shape, keepdims=True), value_shape, uv_shape)

    approx_shape = [glyph_shape[0] * uv_shape[0], glyph_shape[1] * uv_shape[1], 1]

    uv_noise = value.resample(uv_noise, approx_shape, spline_order=InterpolationType.constant)

    x_index = value.row_index(approx_shape) % glyph_shape[1]
    y_index = value.column_index(approx_shape) % glyph_shape[0]

    glyph_count = len(glyphs)
    z_index = tf.cast(uv_noise[:, :, 0] * glyph_count, tf.int32) % glyph_count

    spline_order = InterpolationType.linear if mask == ValueMask.truetype else InterpolationType.constant
    out = value.resample(tf.gather_nd(glyphs, tf.stack([z_index, y_index, x_index], 2)), [shape[0], shape[1], 1], spline_order=spline_order)

    if not colorize:
        return out * tf.ones(shape)

    out *= value.resample(value.proportional_downsample(tensor, shape, [uv_shape[0], uv_shape[1], channels]), shape, spline_order=InterpolationType.constant)

    if alpha == 1.0:
        return out

    return value.blend(tensor, out, alpha)


@effect()
def pixel_sort(tensor, shape, angled=False, darkest=False, time=0.0, speed=1.0):
    """
    Pixel sort effect

    :param Tensor tensor:
    :param list[int] shape:
    :param bool angled: If True, sort along a random angle.
    :param bool darkest: If True, order by darkest instead of brightest
    :return Tensor:
    """

    if angled:
        angle = random.random() * 360.0 if isinstance(angled, bool) else angled

    else:
        angle = False

    tensor = _pixel_sort(tensor, shape, angle, darkest)

    return tensor


def _pixel_sort(tensor, shape, angle, darkest):
    height, width, channels = shape

    if darkest:
        tensor = 1.0 - tensor

    if angle:
        want_length = max(height, width) * 2

        padded_shape = [want_length, want_length, channels]

        padded = tf.image.resize_with_crop_or_pad(tensor, want_length, want_length)

        rotated = tfa.image.rotate(padded, math.radians(angle), 'BILINEAR')

    else:
        padded_shape = shape

        rotated = tensor

    # Find index of brightest pixel
    x_index = tf.expand_dims(tf.argmax(value_map(rotated, padded_shape), axis=1, output_type=tf.int32), -1)

    # Add offset index to row index
    x_index = (value.row_index(padded_shape) - tf.tile(x_index, [1, padded_shape[1]])) % padded_shape[1]

    # Sort pixels
    sorted_channels = [tf.nn.top_k(rotated[:, :, c], padded_shape[1])[0] for c in range(padded_shape[2])]

    # Apply offset
    sorted_channels = tf.gather_nd(tf.stack(sorted_channels, 2), tf.stack([value.column_index(padded_shape), x_index], 2))

    if angle:
        # Rotate back to original orientation
        sorted_channels = tfa.image.rotate(sorted_channels, math.radians(-angle), 'BILINEAR')

        # Crop to original size
        sorted_channels = tf.image.resize_with_crop_or_pad(sorted_channels, height, width)

    # Blend with source image
    tensor = tf.maximum(tensor, sorted_channels)

    if darkest:
        tensor = 1.0 - tensor

    return tensor


@effect()
def rotate(tensor, shape, angle=None):
    """Rotate the image. This breaks seamless edges."""

    height, width, channels = shape

    if angle is None:
        angle = random.random() * 360.0

    want_length = max(height, width) * 2

    padded_shape = [want_length, want_length, channels]

    padded = expand_tile(tensor, shape, padded_shape)

    rotated = tfa.image.rotate(padded, math.radians(angle), 'BILINEAR')

    return tf.image.resize_with_crop_or_pad(rotated, height, width)


@effect()
def sketch(tensor, shape, time=0.0, speed=1.0):
    """
    Pencil sketch effect

    :param Tensor tensor:
    :param list[int] shape:
    :return Tensor:
    """

    value_shape = [shape[0], shape[1], 1]

    values = value_map(tensor, value_shape, keepdims=True)
    values = tf.image.adjust_contrast(values, 2.0)

    values = tf.minimum(values, 1.0)
    values = tf.maximum(values, 0.0)

    outline = 1.0 - derivative(values, value_shape)
    outline = tf.minimum(outline, 1.0 - derivative(1.0 - values, value_shape))
    outline = tf.image.adjust_contrast(outline, .25)
    outline = value.normalize(outline)

    values = vignette(values, value_shape, 1.0, .875)

    crosshatch = 1.0 - worms(1.0 - values, value_shape, behavior=2, density=125, duration=.5, stride=1, stride_deviation=.25, alpha=1.0)
    crosshatch = value.normalize(crosshatch)

    combined = value.blend(crosshatch, outline, .75)
    combined = warp(combined, value_shape, [int(shape[0] * .125) or 1, int(shape[1] * .125) or 1], octaves=1, displacement=.0025, time=time, speed=speed)
    combined *= combined

    return combined * tf.ones(shape)


@effect()
def simple_frame(tensor, shape, brightness=0.0):
    """
    """

    border = singularity(None, shape, dist_metric=DistanceMetric.chebyshev)

    border = value.blend(tf.zeros(shape), border, .55)

    border = posterize(border, 1)

    return value.blend(tensor, tf.ones(shape) * brightness, border)


@effect()
def lowpoly(tensor, shape, distrib=0, freq=10, time=0.0, speed=1.0, dist_metric=DistanceMetric.euclidean):
    """Low-poly art style effect"""

    xy = point_cloud(freq, distrib=distrib, shape=shape, drift=1.0, time=time, speed=speed)

    distance = voronoi(tensor, shape, nth=1, xy=xy, dist_metric=dist_metric)
    color = voronoi(tensor, shape, diagram_type=VoronoiDiagramType.color_regions, xy=xy, dist_metric=dist_metric)

    return value.normalize(value.blend(distance, color, .5))


def square_crop_and_resize(tensor, shape, length=1024):
    """
    Crop and resize an image Tensor into a square with desired side length.

    :param Tensor tensor:
    :param list[int] shape:
    :param int length: Desired side length
    :return Tensor:
    """

    height, width, channels = shape

    have_length = min(height, width)

    if height != width:
        tensor = tf.image.resize_with_crop_or_pad(tensor, have_length, have_length)

    if length != have_length:
        tensor = value.resample(tensor, [length, length, channels])

    return tensor


@effect()
def kaleido(tensor, shape, sides=6, dist_metric=DistanceMetric.euclidean, xy=None, blend_edges=True):
    """
    Adapted from https://github.com/patriciogonzalezvivo/thebookofshaders/blob/master/15/texture-kaleidoscope.frag

    :param Tensor tensor:
    :param list[int] shape:
    :param int sides: Number of sides
    :param DistanceMetric dist_metric:
    :param xy: Optional (x, y) coordinates for points
    :param bool blend_edges: Blend with original edge indices
    """

    height, width, channels = shape

    x_identity = tf.cast(value.row_index(shape), tf.float32)
    y_identity = tf.cast(value.column_index(shape), tf.float32)

    # indices offset to center
    x_index = value.normalize(tf.cast(x_identity, tf.float32)) - .5
    y_index = value.normalize(tf.cast(y_identity, tf.float32)) - .5

    value_shape = [height, width, 1]

    # distance from any pixel to center
    if xy:
        r = voronoi(None, value_shape, dist_metric=dist_metric, xy=xy)

    else:
        r = singularity(None, value_shape, dist_metric=dist_metric)

    r = tf.squeeze(r)

    # cartesian to polar coordinates
    a = tf.math.atan2(y_index, x_index)

    # repeat side according to angle
    # rotate by 90 degrees because vertical symmetry is more pleasing to me
    ma = tf.math.floormod(a + math.radians(90), math.tau / sides)
    ma = tf.math.abs(ma - math.pi / sides)

    # polar to cartesian coordinates
    x_index = r * width * tf.math.sin(ma)
    y_index = r * height * tf.math.cos(ma)

    if blend_edges:
        # fade to original image edges
        fader = value.normalize(singularity(None, value_shape, dist_metric=DistanceMetric.chebyshev))
        fader = tf.squeeze(fader)  # conform to index shape
        fader = tf.math.pow(fader, 5)

        x_index = value.blend(x_index, x_identity, fader)
        y_index = value.blend(y_index, y_identity, fader)

    x_index = tf.cast(x_index, tf.int32)
    y_index = tf.cast(y_index, tf.int32)

    return tf.gather_nd(tensor, tf.stack([y_index % height, x_index % width], 2))


@effect()
def palette(tensor, shape, name=None, time=0.0):
    """
    Another approach to image coloration
    https://iquilezles.org/www/articles/palettes/palettes.htm
    """

    channel_shape = [shape[0], shape[1], 3]

    p = palettes[name]

    offset = p["offset"] * tf.ones(channel_shape)
    amp = p["amp"] * tf.ones(channel_shape)
    freq = p["freq"] * tf.ones(channel_shape)
    phase = p["phase"] * tf.ones(channel_shape) + time

    # Multiply value_map's result x .875, in case the image is just black and white (0 == 1, we don't want a solid color image)
    return offset + amp * tf.math.cos(math.tau * (freq * value_map(tensor, shape, keepdims=True) * .875 + phase))


@effect()
def glitch(tensor, shape, time=0.0, speed=1.0):
    """
    Apply a glitch effect.

    :param Tensor tensor:
    :param list[int] shape:
    :return: Tensor
    """

    height, width, channels = shape

    tensor = value.normalize(tensor)

    base = value.simple_multires(2, shape, time=time, speed=speed, distrib=ValueDistribution.periodic_uniform,
                                 octaves=random.randint(2, 5), spline_order=0)

    base = refract(base, shape, random.random())

    stylized = value.normalize(color_map(base, shape, clut=tensor, horizontal=True, displacement=2.5))

    jpegged = color_map(base, shape, clut=stylized, horizontal=True, displacement=2.5)

    if channels in (1, 3):
        jpegged = jpeg_decimate(jpegged, shape)

    # Offset a single color channel
    separated = [stylized[:, :, i] for i in range(channels)]
    x_index = (value.row_index(shape) + random.randint(1, width)) % width
    index = tf.cast(tf.stack([value.column_index(shape), x_index], 2), tf.int32)

    channel = random.randint(0, channels - 1)
    separated[channel] = value.normalize(tf.gather_nd(separated[channel], index) % random.random())

    stylized = tf.stack(separated, 2)

    combined = value.blend(tf.multiply(stylized, 1.0), jpegged, base)
    combined = value.blend(tensor, combined, tf.maximum(base * 2 - 1, 0))
    combined = value.blend(combined, pixel_sort(combined, shape), 1.0 - base)

    combined = tf.image.adjust_contrast(combined, 1.75)

    return combined


@effect()
def vhs(tensor, shape, time=0.0, speed=1.0):
    """
    Apply a bad VHS tracking effect.

    :param Tensor tensor:
    :param list[int] shape:
    :return: Tensor
    """

    height, width, channels = shape

    # Generate scan noise
    scan_noise = value.values(freq=[int(height * .5) + 1, int(width * .05) + 1], shape=[height, width, 1], time=time,
                              speed=speed, spline_order=1, distrib=ValueDistribution.fastnoise)

    # Create horizontal offsets
    grad = value.values(freq=[int(random.random() * 10) + 5, 1], shape=[height, width, 1], time=time,
                        speed=speed, distrib=ValueDistribution.periodic_uniform)
    grad = tf.maximum(grad - .5, 0)
    grad = tf.minimum(grad * 2, 1)

    x_index = value.row_index(shape)
    x_index -= tf.squeeze(tf.cast(scan_noise * width * tf.square(grad), tf.int32))
    x_index = x_index % width

    tensor = value.blend(tensor, scan_noise, grad)

    identity = tf.stack([value.column_index(shape), x_index], 2)

    tensor = tf.gather_nd(tensor, identity)

    return tensor


@effect()
def lens_warp(tensor, shape, displacement=.0625, time=0.0, speed=1.0):
    """
    """

    value_shape = [shape[0], shape[1], 1]

    # Fake CRT lens shape
    mask = tf.pow(singularity(None, value_shape), 5)  # obscure center pinch

    # Displacement values multiplied by mask to make it wavy towards the edges
    distortion_x = (value.values(2, value_shape, distrib=ValueDistribution.periodic_uniform,
                                 time=time, speed=speed, spline_order=2) * 2.0 - 1.0) * mask

    return refract(tensor, shape, displacement, reference_x=distortion_x)


@effect()
def degauss(tensor, shape, displacement=.0625, time=0.0, speed=1.0):
    """
    """

    channel_shape = [shape[0], shape[1], 1]

    red = lens_warp(tf.expand_dims(tensor[:, :, 0], -1), channel_shape, displacement=displacement, time=time, speed=speed)
    green = lens_warp(tf.expand_dims(tensor[:, :, 1], -1), channel_shape, displacement=displacement, time=time, speed=speed)
    blue = lens_warp(tf.expand_dims(tensor[:, :, 2], -1), channel_shape, displacement=displacement, time=time, speed=speed)

    return tf.stack([tf.squeeze(red), tf.squeeze(green), tf.squeeze(blue)], 2)


@effect()
def crt(tensor, shape, time=0.0, speed=1.0):
    """
    Apply vintage CRT snow and scanlines.

    :param Tensor tensor:
    :param list[int] shape:
    """

    height, width, channels = shape

    value_shape = [height, width, 1]

    # Horizontal scanlines
    scan_noise = tf.tile(value.normalize(value.values(freq=[2, 1], shape=[2, 1, 1], time=time, speed=speed,
                                         distrib=ValueDistribution.periodic_uniform, spline_order=0)),
                         [int(height * .125) or 1, width, 1])

    scan_noise = value.resample(scan_noise, value_shape)

    scan_noise = lens_warp(scan_noise, value_shape, time=time, speed=speed)

    tensor = value.normalize(value.blend(tensor, (tensor + scan_noise) * scan_noise, 0.075))

    if channels == 3:
        tensor = aberration(tensor, shape, .0125 + random.random() * .00625)
        tensor = tf.image.random_hue(tensor, .125)
        tensor = tf.image.adjust_saturation(tensor, 1.25)

    tensor = tf.image.adjust_contrast(tensor, 1.25)

    tensor = vignette(tensor, shape, brightness=0, alpha=random.random() * .175)

    return tensor


@effect()
def scanline_error(tensor, shape, time=0.0, speed=1.0):
    """
    """

    height, width, channels = shape

    value_shape = [height, width, 1]
    error_line = tf.maximum(value.values(freq=[int(height * .75), 1], shape=value_shape, time=time,
                                         speed=speed, distrib=ValueDistribution.fastnoise_exp) - .5, 0)
    error_swerve = tf.maximum(value.values(freq=[int(height * .01), 1], shape=value_shape, time=time,
                                           speed=speed, distrib=ValueDistribution.periodic_exp) - .5, 0)

    error_line *= error_swerve

    error_swerve *= 2

    white_noise = value.values(freq=[int(height * .75), 1], shape=value_shape, time=time,
                               speed=speed, distrib=ValueDistribution.fastnoise)
    white_noise = value.blend(0, white_noise, error_swerve)

    error = error_line + white_noise

    y_index = value.column_index(shape)
    x_index = (value.row_index(shape) - tf.cast(value_map(error, value_shape) * width * .025, tf.int32)) % width

    return tf.minimum(tf.gather_nd(tensor, tf.stack([y_index, x_index], 2)) + error_line * white_noise * 4, 1)


@effect()
def snow(tensor, shape, alpha=0.5, time=0.0, speed=1.0):
    """
    """

    height, width, channels = shape

    static = value.values(freq=[height, width], shape=[height, width, 1], time=time, speed=speed * 100,
                          distrib=ValueDistribution.fastnoise, spline_order=0)

    static_limiter = value.values(freq=[height, width], shape=[height, width, 1], time=time, speed=speed * 100,
                                  distrib=ValueDistribution.fastnoise_exp, spline_order=0) * alpha

    return value.blend(tensor, static, static_limiter)


@effect()
def dither(tensor, shape, alpha=0.5, time=0.0, speed=1.0):
    """
    """

    height, width, channels = shape

    white_noise = value.values(freq=[height, width], shape=[height, width, 1], time=time, speed=speed,
                               distrib=ValueDistribution.fastnoise)

    return value.blend(tensor, white_noise, alpha)


@effect()
def false_color(tensor, shape, horizontal=False, displacement=.5, time=0.0, speed=1.0):
    """
    """

    clut = value.values(freq=2, shape=shape, time=time, speed=speed, distrib=ValueDistribution.periodic_uniform)

    return value.normalize(color_map(tensor, shape, clut=clut, horizontal=horizontal, displacement=displacement))


@effect()
def fibers(tensor, shape, time=0.0, speed=1.0):
    """
    """

    value_shape = [shape[0], shape[1], 1]

    for i in range(4):
        mask = value.values(freq=4, shape=value_shape, time=time, speed=speed, distrib=ValueDistribution.periodic_uniform)

        mask = worms(mask, shape, behavior=WormBehavior.Chaotic, alpha=1, density=.05 + random.random() * .00125,
                     duration=1, kink=random.randint(5, 10), stride=.75, stride_deviation=.125, time=time, speed=speed)

        brightness = value.values(freq=128, shape=shape, time=time, speed=speed, distrib=ValueDistribution.fastnoise)

        tensor = value.blend(tensor, brightness, mask * .5)

    return tensor


@effect()
def scratches(tensor, shape, time=0.0, speed=1.0):
    """
    """

    value_shape = [shape[0], shape[1], 1]

    for i in range(4):
        mask = value.values(freq=random.randint(2, 4), shape=value_shape, time=time, speed=speed,
                            distrib=ValueDistribution.periodic_uniform)

        mask = worms(mask, value_shape, behavior=[1, 3][random.randint(0, 1)], alpha=1, density=.25 + random.random() * .25,
                     duration=2 + random.random() * 2, kink=.125 + random.random() * .125, stride=.75, stride_deviation=.5,
                     time=time, speed=speed)

        mask -= value.values(freq=random.randint(2, 4), shape=value_shape, time=time, speed=speed,
                             distrib=ValueDistribution.periodic_uniform) * 2.0

        mask = tf.maximum(mask, 0.0)

        tensor = tf.maximum(tensor, mask * 8.0)

        tensor = tf.minimum(tensor, 1.0)

    return tensor


@effect()
def stray_hair(tensor, shape, time=0.0, speed=1.0):
    """
    """

    value_shape = [shape[0], shape[1], 1]

    mask = value.values(4, value_shape, time=time, speed=speed, distrib=ValueDistribution.periodic_uniform)

    mask = worms(mask, value_shape, behavior=WormBehavior.unruly, alpha=1, density=.0025 + random.random() * .00125,
                 duration=random.randint(8, 16), kink=random.randint(5, 50), stride=.5, stride_deviation=.25)

    brightness = value.values(freq=32, shape=value_shape, time=time, speed=speed, distrib=ValueDistribution.periodic_uniform)

    return value.blend(tensor, brightness * .333, mask * .666)


@effect()
def grime(tensor, shape, time=0.0, speed=1.0):
    """
    """

    value_shape = [shape[0], shape[1], 1]

    mask = value.simple_multires(freq=5, shape=value_shape, time=time, speed=speed,
                                 distrib=ValueDistribution.periodic_exp, octaves=8)

    mask = refract(mask, value_shape, 1.0, y_from_offset=True)
    mask = derivative(mask, value_shape, DistanceMetric.chebyshev, alpha=0.5)

    dusty = value.blend(tensor, .25, tf.square(mask) * .125)

    specks = value.values(freq=[int(shape[0] * .25), int(shape[1] * .25)], shape=value_shape, time=time,
                          mask=ValueMask.sparse, speed=speed, distrib=ValueDistribution.fastnoise_exp)
    specks = refract(specks, value_shape, .1)

    specks = 1.0 - tf.sqrt(value.normalize(tf.maximum(specks - .5, 0.0)))

    dusty = value.blend(dusty, value.values(freq=[shape[0], shape[1]], shape=value_shape, mask=ValueMask.sparse,
                                            time=time, speed=speed, distrib=ValueDistribution.fastnoise_exp), .125) * specks

    return value.blend(tensor, dusty, mask)


@effect()
def frame(tensor, shape, time=0.0, speed=1.0):
    """
    """

    half_shape = [int(shape[0] * .5), int(shape[1] * .5), shape[2]]
    half_value_shape = [half_shape[0], half_shape[1], 1]

    noise = value.simple_multires(64, half_value_shape, time=time, speed=speed, distrib=ValueDistribution.fastnoise, octaves=8)

    black = tf.zeros(half_value_shape)
    white = tf.ones(half_value_shape)

    mask = singularity(None, half_value_shape, VoronoiDiagramType.range, dist_metric=DistanceMetric.chebyshev, inverse=True)
    mask = value.normalize(mask + noise * .005)
    mask = blend_layers(tf.sqrt(mask), half_value_shape, 0.0125, white, black, black, black)

    faded = value.proportional_downsample(tensor, shape, half_shape)
    faded = tf.image.adjust_brightness(faded, .1)
    faded = tf.image.adjust_contrast(faded, .75)
    faded = light_leak(faded, half_shape, .125)
    faded = vignette(faded, half_shape, 0.05, .75)

    edge_texture = white * .9 + shadow(noise, half_value_shape, 1.0) * .1

    out = value.blend(faded, edge_texture, mask)
    out = aberration(out, half_shape, .00666)
    out = grime(out, half_shape)

    out = tf.image.adjust_saturation(out, .5)
    out = tf.image.random_hue(out, .05)

    out = value.resample(out, shape)

    out = scratches(out, shape)

    out = stray_hair(out, shape)

    return out


@effect()
def texture(tensor, shape, time=0.0, speed=1.0):
    """
    """

    value_shape = [shape[0], shape[1], 1]

    noise = value.simple_multires(64, value_shape, time=time, speed=speed,
                                  distrib=ValueDistribution.fastnoise, octaves=8, ridges=True)

    return tensor * (tf.ones(value_shape) * .95 + shadow(noise, value_shape, 1.0) * .05)


@effect()
def watermark(tensor, shape, time=0.0, speed=1.0):
    """
    """

    value_shape = [int(shape[0] * .5), int(shape[1] * .5), 1]
    value_shape = [shape[0], shape[1], 1]

    mask = value.values(freq=240, shape=value_shape, spline_order=0, distrib=ValueDistribution.ones, mask="alphanum_numeric")

    mask = crt(mask, value_shape)

    mask = warp(mask, value_shape, [2, 4], octaves=1, displacement=.5, time=time, speed=speed)

    mask *= tf.square(value.values(freq=2, shape=value_shape, time=time, speed=speed, distrib=ValueDistribution.periodic_uniform))

    value_shape = [shape[0], shape[1], 1]

    brightness = value.values(freq=16, shape=value_shape, time=time, speed=speed, distrib=ValueDistribution.periodic_uniform)

    return value.blend(tensor, brightness, mask * .125)


@effect()
def spooky_ticker(tensor, shape, time=0.0, speed=1.0):
    """
    """

    if random.random() > .75:
        tensor = on_screen_display(tensor, shape, time=time, speed=speed)

    _masks = [
        ValueMask.arecibo_nucleotide,
        ValueMask.arecibo_num,
        ValueMask.bank_ocr,
        ValueMask.bar_code,
        ValueMask.bar_code_short,
        ValueMask.emoji,
        ValueMask.fat_lcd_hex,
        ValueMask.alphanum_hex,
        ValueMask.iching,
        ValueMask.ideogram,
        ValueMask.invaders,
        ValueMask.lcd,
        ValueMask.letters,
        ValueMask.matrix,
        ValueMask.alphanum_numeric,
        ValueMask.script,
        ValueMask.white_bear,
    ]

    bottom_padding = 2

    rendered_mask = tf.zeros(shape)

    for _ in range(random.randint(1, 3)):
        mask = _masks[random.randint(0, len(_masks) - 1)]
        mask_shape = masks.mask_shape(mask)

        multiplier = 1 if mask != ValueMask.script and (mask_shape[1] == 1 or mask_shape[1] >= 10) else 2

        width = int(shape[1] / multiplier) or 1
        width = mask_shape[1] * int(width / mask_shape[1])  # Make sure the mask divides evenly into width

        freq = [mask_shape[0], width]

        row_shape = [mask_shape[0], width, 1]
        row_mask = value.values(freq=freq, shape=row_shape, corners=True, spline_order=0, distrib=ValueDistribution.ones,
                                mask=mask, time=time, speed=speed)

        if time != 0.0:  # Make the ticker tick!
            row_mask = value.offset(row_mask, row_shape, int(time*width), 0)

        row_mask = value.resample(row_mask, [mask_shape[0] * multiplier, shape[1]], spline_order=1)

        rendered_mask += tf.pad(row_mask, tf.stack([[shape[0] - mask_shape[0] * multiplier - bottom_padding, bottom_padding], [0, 0], [0, 0]]))

        bottom_padding += mask_shape[0] * multiplier + 2

    alpha = .5 + random.random() * .25

    # shadow
    tensor = value.blend(tensor, tensor * 1.0 - value.offset(rendered_mask, shape, -1, -1), alpha * .333)

    return value.blend(tensor, tf.maximum(rendered_mask, tensor), alpha)


@effect()
def on_screen_display(tensor, shape, time=0.0, speed=1.0):
    glyph_count = random.randint(3, 6)

    _masks = [
        ValueMask.bank_ocr,
        ValueMask.alphanum_hex,
        ValueMask.alphanum_numeric,
    ]

    mask = _masks[random.randint(0, len(_masks) - 1)]
    mask_shape = masks.mask_shape(mask)

    width = int(shape[1] / 24)

    width = mask_shape[1] * int(width / mask_shape[1])  # Make sure the mask divides evenly
    height = mask_shape[0] * int(width / mask_shape[1])

    width *= glyph_count

    freq = [mask_shape[0], mask_shape[1] * glyph_count]

    row_mask = value.values(freq=freq, shape=[height, width, shape[2]], corners=True, spline_order=0, distrib=ValueDistribution.ones,
                            mask=mask, time=time, speed=speed)

    rendered_mask = tf.pad(row_mask, tf.stack([[25, shape[0] - height - 25], [shape[1] - width - 25, 25], [0, 0]]))

    alpha = .5 + random.random() * .25

    return value.blend(tensor, tf.maximum(rendered_mask, tensor), alpha)


@effect()
def nebula(tensor, shape, time=0.0, speed=1.0):
    overlay = value.simple_multires(random.randint(2, 4), shape, time=time, speed=speed,
                                    distrib=ValueDistribution.periodic_exp, ridges=True, octaves=6)

    overlay -= value.simple_multires(random.randint(2, 4), shape, time=time, speed=speed,
                                     distrib=ValueDistribution.periodic_uniform, ridges=True, octaves=4)

    overlay = tf.maximum(overlay, 0)

    return tf.maximum(tensor, overlay * .25)


@effect()
def spatter(tensor, shape, time=0.0, speed=1.0):
    """
    """

    value_shape = [shape[0], shape[1], 1]

    # Generate a smear
    smear = value.simple_multires(random.randint(2, 4), value_shape, time=time,
                                  speed=speed, distrib=ValueDistribution.simplex_exp,
                                  ridges=True, octaves=6, spline_order=3)

    smear = warp(smear, value_shape, [random.randint(2, 3), random.randint(1, 3)],
                 octaves=random.randint(1, 2), displacement=1.0 + random.random(),
                 spline_order=3, time=time, speed=speed)

    # Add spatter dots
    spatter = value.simple_multires(random.randint(25, 50), value_shape, time=time,
                                    speed=speed, distrib=ValueDistribution.simplex_exp,
                                    octaves=4, spline_order=InterpolationType.linear)

    spatter = post_process(spatter, shape, None, post_brightness=-.25, post_contrast=4)

    smear = tf.maximum(smear, spatter)

    spatter = value.simple_multires(random.randint(200, 250), value_shape, time=time,
                                    speed=speed, distrib=ValueDistribution.simplex_exp,
                                    octaves=4, spline_order=InterpolationType.linear)

    spatter = post_process(spatter, shape, None, post_brightness=-.25, post_contrast=4)

    smear = tf.maximum(smear, spatter)

    # Remove some of it
    smear = tf.maximum(0.0, smear - value.simple_multires(random.randint(2, 3), value_shape, time=time,
                                                          speed=speed, distrib=ValueDistribution.simplex_exp,
                                                          ridges=True, octaves=3, spline_order=2))

    #
    if shape[2] == 3:
        splash = tf.image.random_hue(tf.ones(shape) * tf.stack([.875, 0.125, 0.125]), .5)

    else:
        splash = tf.zeros(shape)

    return blend_layers(value.normalize(smear), shape, .005, tensor, splash)


@effect()
def clouds(tensor, shape, time=0.0, speed=1.0):
    """Top-down cloud cover effect"""

    pre_shape = [int(shape[0] * .25) or 1, int(shape[1] * .25) or 1, 1]

    control = value.simple_multires(freq=random.randint(2, 4), shape=pre_shape, distrib=ValueDistribution.periodic_uniform,
                                    octaves=8, ridges=True, time=time, speed=speed)

    control = warp(control, pre_shape, freq=3, displacement=.125, octaves=2)

    layer_0 = tf.ones(pre_shape)
    layer_1 = tf.zeros(pre_shape)

    combined = blend_layers(control, pre_shape, 1.0, layer_0, layer_1)

    shadow = value.offset(combined, pre_shape, random.randint(-15, 15), random.randint(-15, 15))
    shadow = tf.minimum(shadow * 2.5, 1.0)

    for _ in range(3):
        shadow = convolve(ValueMask.conv2d_blur, shadow, pre_shape)

    post_shape = [shape[0], shape[1], 1]

    shadow = value.resample(shadow, post_shape)
    combined = value.resample(combined, post_shape)

    tensor = value.blend(tensor, tf.zeros(shape), shadow * .75)
    tensor = value.blend(tensor, tf.ones(shape), combined)

    tensor = shadow(tensor, shape, alpha=.5)

    return tensor


@effect()
def tint(tensor, shape, time=0.0, speed=1.0, alpha=0.5):
    """
    """

    if shape[2] < 3:  # Not a color image
        return tensor

    color = value.values(freq=3, shape=shape, time=time, speed=speed, distrib=ValueDistribution.periodic_uniform, corners=True)

    # Confine hue to a range
    color = tf.stack([(tensor[:, :, 0] * .333 + random.random() * .333 + random.random()) % 1.0,
                      tensor[:, :, 1], tensor[:, :, 2]], 2)

    colorized = tf.stack([color[:, :, 0], color[:, :, 1], tf.image.rgb_to_hsv([tensor])[0][:, :, 2]], 2)

    colorized = tf.image.hsv_to_rgb([colorized])[0]

    return value.blend(tensor, colorized, alpha)
