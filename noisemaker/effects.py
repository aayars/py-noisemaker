"""Low-level effects library for Noisemaker"""

from collections import defaultdict

import math
import os
import random

from PIL import Image

import numpy as np
import tensorflow as tf

from noisemaker.constants import DistanceFunction, PointDistribution, ValueMask, VoronoiDiagramType, WormBehavior
from noisemaker.glyphs import load_glyphs
from noisemaker.points import point_cloud

import noisemaker.masks as masks
import noisemaker.simplex as simplex
import noisemaker.util as util


def post_process(tensor, shape, freq, ridges_hint=False, spline_order=3, reflect_range=0.0, refract_range=0.0, reindex_range=0.0,
                 refract_extend_range=False, clut=None, clut_horizontal=False, clut_range=0.5,
                 with_worms=None, worms_density=4.0, worms_duration=4.0, worms_stride=1.0, worms_stride_deviation=.05,
                 worms_alpha=.5, worms_kink=1.0, with_sobel=None, with_normal_map=False, deriv=None, deriv_alpha=1.0, with_outline=False,
                 with_glowing_edges=False, with_wormhole=False, wormhole_kink=2.5, wormhole_stride=.1, wormhole_alpha=1.0,
                 with_voronoi=0, voronoi_nth=0, voronoi_func=1, voronoi_alpha=1.0, voronoi_refract=0.0, voronoi_inverse=False,
                 posterize_levels=0,
                 with_erosion_worms=False, erosion_worms_density=50, erosion_worms_iterations=50, erosion_worms_contraction=1.0,
                 erosion_worms_alpha=1.0, erosion_worms_inverse=False, erosion_worms_xy_blend=None,
                 warp_range=0.0, warp_octaves=3, warp_interp=None, warp_freq=None,
                 ripple_range=0.0, ripple_freq=None, ripple_kink=1.0,
                 vortex_range=0.0, with_pop=False, with_aberration=None, with_dla=0.0, dla_padding=2,
                 point_freq=5, point_distrib=0, point_corners=False, point_generations=1, point_drift=0.0,
                 with_bloom=None, with_reverb=None, reverb_iterations=1, reverb_ridges=True,
                 with_light_leak=None, with_vignette=None, vignette_brightness=0.0,
                 post_hue_rotation=None, post_saturation=None, post_brightness=None, post_contrast=None,
                 input_dir=None, with_crease=False, with_jpeg_decimate=None, with_conv_feedback=None, conv_feedback_alpha=.5,
                 with_density_map=False,
                 with_glyph_map=None, glyph_map_colorize=True, glyph_map_zoom=1.0, glyph_map_alpha=1.0,
                 with_composite=None, composite_zoom=4.0, with_sort=False, sort_angled=False, sort_darkest=False,
                 with_convolve=None, with_shadow=None, with_sketch=False,
                 with_lowpoly=False, lowpoly_distrib=0, lowpoly_freq=10,
                 angle=None,
                 rgb=False, time=0.0, **_):
    """
    Apply post-processing effects.

    :param Tensor tensor:
    :param list[int] shape:
    :param list[int] freq:
    :param int spline_order: Ortho spline point count (0=Constant, 1=Linear, 2=Cosine, 3=Bicubic)
    :param float reflect_range: Derivative distortion gradient.
    :param float refract_range: Self-distortion gradient.
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
    :param DistanceFunction|int sobel: Sobel operator distance function
    :param DistanceFunction|int outline: Outlines distance function (multiply)
    :param bool with_normal_map: Create a tangent-space normal map
    :param bool with_wormhole: Wormhole effect. What is this?
    :param float wormhole_kink: Wormhole kinkiness, if you're into that.
    :param float wormhole_stride: Wormhole thickness range
    :param float wormhole_alpha: Wormhole alpha blending
    :param VoronoiDiagramType|int with_voronoi: Voronoi diagram type (0=Off, 1=Range, 2=Color Range, 3=Indexed, 4=Color Map, 5=Blended, 6=Flow)
    :param int voronoi_nth: Voronoi Nth nearest
    :param DistanceFunction|int voronoi_func: Voronoi distance function
    :param float voronoi_alpha: Blend with original tensor (0.0 = Original, 1.0 = Voronoi)
    :param float voronoi_refract: Domain warp input tensor against Voronoi
    :param bool voronoi_inverse: Inverse values for Voronoi 'range' types
    :param bool ridges_hint: Ridged multifractal hint for Voronoi
    :param DistanceFunction|int deriv: Derivative distance function
    :param float deriv_alpha: Derivative distance function alpha blending amount
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
    :param float ripple_range: Ripple range
    :param float ripple_freq: Ripple frequency
    :param float ripple_kink: Ripple twistiness
    :param bool with_pop: Pop art filter
    :param float|None with_aberration: Chromatic aberration distance
    :param float|None with_bloom: Bloom alpha
    :param bool with_dla: Diffusion-limited aggregation alpha
    :param int dla_padding: DLA pixel padding
    :param int point_freq: Voronoi and DLA point frequency (freq * freq = count)
    :param PointDistribution|int point_distrib: Voronoi and DLA point cloud distribution
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
    :param None|str input_dir: Input directory containing .png and/or .jpg images, for collage functions.
    :param bool with_crease: Crease at midpoint values
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
    :param None|float angle: Rotation angle
    :param bool rgb: Using RGB mode? Hint for some effects.
    :return: Tensor
    """

    tensor = normalize(tensor)

    if with_voronoi or with_dla:
        multiplier = max(2 * (point_generations - 1), 1)

        tiled_shape = [int(shape[0] / multiplier), int(shape[1] / multiplier), shape[2]]

        if point_freq == 1:
            x, y = point_cloud(1, PointDistribution.square, shape)

        else:
            x, y = point_cloud(point_freq, distrib=point_distrib, shape=tiled_shape, corners=point_corners, generations=point_generations, drift=point_drift, time=time)

        xy = (x, y, len(x))

        input_tensor = resample(tensor, tiled_shape)

        if with_voronoi:
            input_tensor = voronoi(input_tensor, tiled_shape, alpha=voronoi_alpha, diagram_type=with_voronoi, dist_func=voronoi_func, inverse=voronoi_inverse,
                                   nth=voronoi_nth, ridges_hint=ridges_hint, with_refract=voronoi_refract, xy=xy, input_dir=input_dir)

        if with_dla:
            input_tensor = blend(input_tensor, dla(input_tensor, tiled_shape, padding=dla_padding, xy=xy), with_dla)

        if point_generations == 1:
            tensor = input_tensor

        else:
            tensor = expand_tile(input_tensor, tiled_shape, shape)

    # Keep values between 0 and 1 if we're reflecting and refracting, because math?
    # Using refract and reflect together exposes unpleasant edge artifacting along
    # the natural edge where negative and positive offset values meet. It's normally
    # invisible to the human eye, but becomes obvious after extracting derivatives.
    extend_range = refract_extend_range and refract_range != 0 and reflect_range != 0

    if refract_range != 0:
        tensor = refract(tensor, shape, displacement=refract_range, extend_range=extend_range)

    if reflect_range != 0:
        tensor = refract(tensor, shape, displacement=reflect_range, from_derivative=True)

    if reindex_range != 0:
        tensor = reindex(tensor, shape, displacement=reindex_range)

    if clut:
        tensor = color_map(tensor, clut, shape, horizontal=clut_horizontal, displacement=clut_range)

    if with_glyph_map:
        tensor = glyph_map(tensor, shape, mask=with_glyph_map, colorize=glyph_map_colorize, zoom=glyph_map_zoom, alpha=glyph_map_alpha)

    if with_composite:
        tensor = glyph_map(tensor, shape, zoom=composite_zoom, mask=with_composite)

    if warp_range:
        if warp_interp is None:
            warp_interp = spline_order

        warp_freq = freq if warp_freq is None else warp_freq if isinstance(warp_freq, list) else freq_for_shape(warp_freq, shape)

        tensor = warp(tensor, shape, warp_freq, displacement=warp_range, octaves=warp_octaves, spline_order=warp_interp, time=time)

    if ripple_range:
        ripple_freq = freq if ripple_freq is None else ripple_freq if isinstance(ripple_freq, list) else freq_for_shape(ripple_freq, shape)

        tensor = ripple(tensor, shape, ripple_freq, displacement=ripple_range, kink=ripple_kink, time=time)

    if vortex_range:
        tensor = vortex(tensor, shape, displacement=vortex_range, time=time)

    if deriv:
        tensor = derivative(tensor, shape, deriv, alpha=deriv_alpha)

    if with_crease:
        tensor = crease(tensor)

    if posterize_levels:
        tensor = posterize(tensor, posterize_levels)

    if with_worms:
        tensor = worms(tensor, shape, behavior=with_worms, density=worms_density, duration=worms_duration,
                       stride=worms_stride, stride_deviation=worms_stride_deviation, alpha=worms_alpha, kink=worms_kink)

    if with_wormhole:
        tensor = wormhole(tensor, shape, wormhole_kink, wormhole_stride, alpha=wormhole_alpha)

    if with_erosion_worms:
        tensor = erode(tensor, shape, density=erosion_worms_density, iterations=erosion_worms_iterations,
                       contraction=erosion_worms_contraction, alpha=erosion_worms_alpha, inverse=erosion_worms_inverse,
                       xy_blend=erosion_worms_xy_blend)

    if with_density_map:
        tensor = density_map(tensor, shape)

    if with_sobel:
        tensor = sobel(tensor, shape, with_sobel, rgb)

    if with_convolve:
        for kernel in with_convolve:
            if isinstance(kernel, str):
                if kernel not in ValueMask:
                    kernel = 'conv2d_{}'.format(kernel)

                kernel = ValueMask[kernel]

            tensor = convolve(kernel, tensor, shape)

    if with_shadow:
        tensor = shadow(tensor, shape, with_shadow)

    if with_outline:
        tensor = outline(tensor, shape, sobel_func=with_outline)

    if with_glowing_edges:
        tensor = glowing_edges(tensor, shape, alpha=with_glowing_edges)

    if with_reverb:
        tensor = reverb(tensor, shape, with_reverb, iterations=reverb_iterations, ridges=reverb_ridges)

    if with_pop:
        tensor = pop(tensor, shape)

    if with_aberration:
        tensor = aberration(tensor, shape, displacement=with_aberration, time=time)

    if with_bloom:
        tensor = bloom(tensor, shape, alpha=with_bloom)

    if with_light_leak:
        tensor = light_leak(tensor, shape, with_light_leak, time=time)

    if with_vignette:
        tensor = vignette(tensor, shape, brightness=vignette_brightness, alpha=with_vignette)

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
        tensor = pixel_sort(tensor, shape, sort_angled, sort_darkest)

    if with_sketch:
        tensor = sketch(tensor, shape, time=time())

    if with_lowpoly:
        tensor = lowpoly(tensor, shape, distrib=lowpoly_distrib, freq=lowpoly_freq, time=time)

    if angle is not None:
        tensor = rotate(tensor, shape, angle)

    tensor = normalize(tensor)

    return tensor


def _conform_kernel_to_tensor(kernel, tensor, shape):
    """ Re-shape a convolution kernel to match the given tensor's color dimensions. """

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

    out = offset(out, double_shape, half_width, half_height)

    out = tf.nn.depthwise_conv2d([out], kernel_values, [1, 1, 1, 1], "VALID")[0]

    out = tf.image.resize_image_with_crop_or_pad(out, height, width)

    if with_normalize:
        out = normalize(out)

    if kernel == ValueMask.conv2d_edges:
        out = tf.abs(out - .5) * 2

    if alpha == 1.0:
        return out

    return blend(tensor, out, alpha)


def normalize(tensor, extend_range=False):
    """
    Squeeze the given Tensor into a range between 0 and 1.

    :param Tensor tensor: An image tensor.
    :param bool extend_range: Use a range between -1 and 1.
    :return: Tensor
    """

    floor = tf.reduce_min(tensor)
    ceil = tf.reduce_max(tensor)

    values = (tensor - floor) / (ceil - floor)

    if extend_range:
        values = values * 2.0 - 1.0

    return values


def resample(tensor, shape, spline_order=3):
    """
    Resize an image tensor to the specified shape.

    :param Tensor tensor:
    :param list[int] shape:
    :param int spline_order: Spline point count. 0=Constant, 1=Linear, 2=Cosine, 3=Bicubic
    :return: Tensor
    """

    input_shape = tf.shape(tensor)

    # Blown up row and column indices. These map into input tensor, producing a big blocky version.
    resized_row_index = tf.cast(row_index(shape), tf.float32) * (tf.cast(input_shape[1], tf.float32) / shape[1])   # 0, 1, 2, 3, -> 0, 0.5, 1, 1.5A
    resized_col_index = tf.cast(column_index(shape), tf.float32) * (tf.cast(input_shape[0], tf.float32) / shape[0])

    # Map to input indices as int
    resized_row_index_trunc = tf.floor(resized_row_index)
    resized_col_index_trunc = tf.floor(resized_col_index)
    resized_index_trunc = tf.cast(tf.stack([resized_col_index_trunc, resized_row_index_trunc], 2), tf.int32)

    # Resized original
    resized = defaultdict(dict)
    resized[1][1] = tf.gather_nd(tensor, resized_index_trunc)

    if spline_order == 0:
        return resized[1][1]

    # Resized neighbors
    input_rows = defaultdict(dict)
    input_columns = defaultdict(dict)

    input_rows[1] = row_index(input_shape)
    input_columns[1] = column_index(input_shape)

    input_rows[2] = (input_rows[1] + 1) % input_shape[1]
    input_columns[2] = (input_columns[1] + 1) % input_shape[0]

    # Create fractional diffs (how much to blend with each neighbor)
    value_shape = [shape[0], shape[1], 1]
    resized_row_index_fract = tf.reshape(resized_row_index - resized_row_index_trunc, value_shape)  # 0, 0.5, 1, 1.5 -> 0, .5, 0, .5
    resized_col_index_fract = tf.reshape(resized_col_index - resized_col_index_trunc, value_shape)

    for x in range(1, 3):
        for y in range(1, 3):
            if x == 1 and y == 1:
                continue

            resized[y][x] = _gather_scaled_offset(tensor, input_columns[y], input_rows[x], resized_index_trunc)

    if spline_order == 1:
        y1 = blend(resized[1][1], resized[1][2], resized_row_index_fract)
        y2 = blend(resized[2][1], resized[2][2], resized_row_index_fract)

        return blend(y1, y2, resized_col_index_fract)

    if spline_order == 2:
        y1 = blend_cosine(resized[1][1], resized[1][2], resized_row_index_fract)
        y2 = blend_cosine(resized[2][1], resized[2][2], resized_row_index_fract)

        return blend_cosine(y1, y2, resized_col_index_fract)

    if spline_order == 3:
        # Extended neighborhood for bicubic
        points = []

        for y in range(0, 4):
            if y not in input_columns:
                input_columns[y] = (input_columns[1] + (y - 1)) % input_shape[0]

            for x in range(0, 4):
                if x not in input_rows:
                    input_rows[x] = (input_rows[1] + (x - 1)) % input_shape[1]

                resized[y][x] = _gather_scaled_offset(tensor, input_columns[y], input_rows[x], resized_index_trunc)

            points.append(blend_cubic(resized[y][0], resized[y][1], resized[y][2], resized[y][3], resized_row_index_fract))

        args = points + [resized_col_index_fract]

        return blend_cubic(*args)


def _downsample(tensor, shape, new_shape):
    """ Proportional downsample """

    kernel_shape = [int(max(shape[0] / new_shape[0], 1)), int(max(shape[1] / new_shape[1], 1)), shape[2], 1]

    kernel = tf.ones(kernel_shape)

    out = tf.nn.depthwise_conv2d([tensor], kernel, [1, kernel_shape[0], kernel_shape[1], 1], "VALID")[0] / (kernel_shape[0] * kernel_shape[1])

    return resample(out, new_shape)


def _gather_scaled_offset(tensor, input_column_index, input_row_index, output_index):
    """ Helper function for resample(). Apply index offset to input tensor, return output_index values gathered post-offset. """

    return tf.gather_nd(tf.gather_nd(tensor, tf.stack([input_column_index, input_row_index], 2)), output_index)


def crease(tensor):
    """
    Create a "crease" (ridge) at midpoint values. 1 - abs(n * 2 - 1)

    .. image:: images/crease.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor: An image tensor.
    :return: Tensor
    """

    return 1.0 - tf.abs(tensor * 2 - 1)


def erode(tensor, shape, density=50, iterations=50, contraction=1.0, alpha=.25, inverse=False, xy_blend=False):
    """
    WIP hydraulic erosion effect.
    """

    # This will never be as good as
    # https://www.dropbox.com/s/kqv8b3w7o8ucbyi/Beyer%20-%20implementation%20of%20a%20methode%20for%20hydraulic%20erosion.pdf?dl=0

    height, width, channels = shape

    count = int(math.sqrt(height * width) * density)

    x = tf.random_uniform([count]) * (width - 1)
    y = tf.random_uniform([count]) * (height - 1)

    x_dir = tf.random_normal([count])
    y_dir = tf.random_normal([count])

    length = tf.sqrt(x_dir * x_dir + y_dir * y_dir)
    x_dir /= length
    y_dir /= length

    inertia = tf.random_normal([count], mean=0.75, stddev=0.25)

    out = tf.zeros(shape)

    # colors = tf.gather_nd(tensor, tf.cast(tf.stack([y, x], 1), tf.int32))

    values = value_map(convolve(ValueMask.conv2d_blur, tensor, shape), shape, keep_dims=True)

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
        g_x = blend(y1_values - sparse_values, x1_y1_values - x1_values, u)
        g_y = blend(x1_values - sparse_values, x1_y1_values - y1_values, v)

        length = distance(g_x, g_y, 1) * contraction

        x_dir = blend(x_dir, g_x / length, inertia)
        y_dir = blend(y_dir, g_y / length, inertia)

        # step
        x = (x + x_dir) % width
        y = (y + y_dir) % height

    out = tf.maximum(tf.minimum(out, 1.0), 0.0)

    if inverse:
        out = 1.0 - out

    if xy_blend:
        tensor = blend(shadow(tensor, shape), reindex(tensor, shape, 1), xy_blend * values)

    return blend(tensor, out, alpha)


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


def refract(tensor, shape, displacement=.5, reference_x=None, reference_y=None, warp_freq=None, spline_order=3, from_derivative=False, extend_range=True, time=0.0):
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
    :param bool extend_range: Scale displacement values from -1..1 instead of 0..1
    :return: Tensor
    """

    height, width, channels = shape

    x0_index = row_index(shape)
    y0_index = column_index(shape)

    warp_shape = None

    if warp_freq:
        warp_shape = [warp_freq[0], warp_freq[1], 1]

    if reference_x is None:
        if from_derivative:
            reference_x = convolve(ValueMask.conv2d_deriv_x, tensor, shape, with_normalize=False)

        elif warp_freq:
            reference_x = resample(simplex.simplex(warp_shape, time=time, seed=random.randint(1, 65536)), shape, spline_order=spline_order)

        else:
            reference_x = tensor

    if reference_y is None:
        if from_derivative:
            reference_y = convolve(ValueMask.conv2d_deriv_y, tensor, shape, with_normalize=False)

        elif warp_freq:
            reference_y = resample(simplex.simplex(warp_shape, time=time, seed=random.randint(1, 65536)), shape, spline_order=spline_order)

        else:
            y0_index += int(height * .5)
            x0_index += int(width * .5)
            reference_y = tf.gather_nd(reference_x, tf.stack([y0_index % height, x0_index % width], 2))

    quad_directional = extend_range and not from_derivative

    # Use extended range so we can refract in 4 directions (-1..1) instead of 2 (0..1).
    # Doesn't work with derivatives (and isn't needed), because derivatives are signed naturally.
    x_offsets = value_map(reference_x, shape, extend_range=quad_directional) * displacement * width
    y_offsets = value_map(reference_y, shape, extend_range=quad_directional) * displacement * height
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

    x_y0 = blend(x0_y0, x1_y0, x_fract)
    x_y1 = blend(x0_y1, x1_y1, x_fract)

    return blend(x_y0, x_y1, y_fract)


def ripple(tensor, shape, freq, displacement=1.0, kink=1.0, reference=None, spline_order=3, time=0.0):
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

    x0_index = row_index(shape)
    y0_index = column_index(shape)

    value_shape = [shape[0], shape[1], 1]

    if reference is None:
        reference = resample(tf.random_uniform([freq[0], freq[1], 1]), value_shape, spline_order=spline_order)
        # reference = derivative(reference, [shape[0], shape[1], 1], with_normalize=False)

    # Twist index, borrowed from worms. TODO merge me.
    index = value_map(reference, shape) * 360.0 * math.radians(1) * kink * simplex.random(time)

    reference_x = (tf.cos(index) * displacement * width) % width
    reference_y = (tf.sin(index) * displacement * height) % height

    # Bilinear interpolation of midpoints, borrowed from refract(). TODO merge me
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

    x_y0 = blend(x0_y0, x1_y0, x_fract)
    x_y1 = blend(x0_y1, x1_y1, x_fract)

    return blend(x_y0, x_y1, y_fract)


def color_map(tensor, clut, shape, horizontal=False, displacement=.5):
    """
    Apply a color map to an image tensor.

    The color map can be a photo or whatever else.

    .. image:: images/color_map.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :param Tensor|str clut: An image tensor or filename (png/jpg only) to use as a color palette
    :param list[int] shape:
    :param bool horizontal: Scan horizontally
    :param float displacement: Gather distance for clut
    """

    if isinstance(clut, str):
        clut = util.load(clut)

    height, width, channels = shape

    reference = value_map(tensor, shape) * displacement

    x_index = (row_index(shape) + tf.cast(reference * (width - 1), tf.int32)) % width

    if horizontal:
        y_index = column_index(shape)

    else:
        y_index = (column_index(shape) + tf.cast(reference * (height - 1), tf.int32)) % height

    index = tf.stack([y_index, x_index], 2)

    clut = resample(tf.image.convert_image_dtype(clut, tf.float32, saturate=True), shape)

    output = tf.gather_nd(clut, index)

    return output


def worms(tensor, shape, behavior=1, density=4.0, duration=4.0, stride=1.0, stride_deviation=.05, alpha=.5, kink=1.0, colors=None):
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
    :param Tensor colors: Optional starting colors, if not from `tensor`.
    :return: Tensor
    """

    height, width, channels = shape

    count = int(max(width, height) * density)

    worms_y = tf.random_uniform([count]) * (height - 1)
    worms_x = tf.random_uniform([count]) * (width - 1)
    worms_stride = tf.random_normal([count], mean=stride, stddev=stride_deviation) * (max(width, height)/1024.0)

    color_source = colors if colors is not None else tensor

    colors = tf.gather_nd(color_source, tf.cast(tf.stack([worms_y, worms_x], 1), tf.int32))

    if isinstance(behavior, int):
        behavior = WormBehavior(behavior)

    quarter_count = int(count * .25)

    rots = {}

    rots = {
        WormBehavior.obedient: lambda n:
            tf.ones([n]) * random.random() * 360.0,

        WormBehavior.crosshatch: lambda n:
            rots[WormBehavior.obedient](n) + (tf.floor(tf.random_normal([n]) * 100) % 4) * 90,

        WormBehavior.unruly: lambda n:
            rots[WormBehavior.obedient](n) + tf.random_normal([n]) * .25 - .125,

        WormBehavior.chaotic: lambda n:
            tf.random_normal([n]) * 360.0,

        WormBehavior.random: lambda _:
            tf.reshape(tf.stack([
                rots[WormBehavior.obedient](quarter_count),
                rots[WormBehavior.crosshatch](quarter_count),
                rots[WormBehavior.unruly](quarter_count),
                rots[WormBehavior.chaotic](quarter_count),
            ]), [count]),
    }

    worms_rot = rots[behavior](count)

    index = value_map(tensor, shape) * 360.0 * math.radians(1) * kink

    iterations = int(math.sqrt(min(width, height)) * duration)

    out = tf.zeros(shape)

    scatter_shape = tf.shape(tensor)  # Might be different than `shape` due to clut

    # Make worms!
    for i in range(iterations):
        worm_positions = tf.cast(tf.stack([worms_y % height, worms_x % width], 1), tf.int32)

        exposure = 1 - abs(1 - i / (iterations - 1) * 2)  # Makes linear gradient [ 0 .. 1 .. 0 ]

        out += tf.scatter_nd(worm_positions, colors * exposure, scatter_shape)

        next_position = tf.gather_nd(index, worm_positions) + (worms_rot - 45.0)

        worms_y = (worms_y + tf.cos(next_position) * worms_stride) % height
        worms_x = (worms_x + tf.sin(next_position) * worms_stride) % width

    out = tf.image.convert_image_dtype(out, tf.float32, saturate=True)

    return blend(tensor, tf.sqrt(normalize(out)), alpha)


def wormhole(tensor, shape, kink, input_stride, alpha=1.0):
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

    values = value_map(tensor, shape)

    degrees = values * 360.0 * math.radians(1) * kink
    # stride = values * height * input_stride
    stride = height * input_stride

    x_index = tf.cast(row_index(shape), tf.float32)
    y_index = tf.cast(column_index(shape), tf.float32)

    x_offset = (tf.cos(degrees) + 1) * stride
    y_offset = (tf.sin(degrees) + 1) * stride

    x = tf.cast(x_index + x_offset, tf.int32) % width
    y = tf.cast(y_index + y_offset, tf.int32) % height

    luminosity = tf.square(tf.reshape(values, [height, width, 1]))

    out = normalize(tf.scatter_nd(offset_index(y, height, x, width), tensor * luminosity, tf.shape(tensor)))

    return blend(tensor, tf.sqrt(out), alpha)


def wavelet(tensor, shape):
    """
    Convert regular noise into 2-D wavelet noise.

    Completely useless. Maybe useful if Noisemaker supports higher dimensions later.

    .. image:: images/wavelet.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor: An image tensor.
    :param list[int] shape:
    :return: Tensor
    """

    height, width, channels = shape

    return normalize(tensor - resample(_downsample(tensor, shape, [int(height * .5), int(width * .5), channels]), shape))


def derivative(tensor, shape, dist_func=1, with_normalize=True, alpha=1.0):
    """
    Extract a derivative from the given noise.

    .. image:: images/derived.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :param list[int] shape:
    :param DistanceFunction|int dist_func: Derivative distance function
    :param bool with_normalize:
    :return: Tensor
    """

    x = convolve(ValueMask.conv2d_deriv_x, tensor, shape, with_normalize=False)
    y = convolve(ValueMask.conv2d_deriv_y, tensor, shape, with_normalize=False)

    out = distance(x, y, dist_func)

    if with_normalize:
        out = normalize(out)

    if alpha == 1.0:
        return out

    return blend(tensor, out, alpha)


def sobel_operator(tensor, shape, dist_func=1):
    """
    Apply a sobel operator.

    .. image:: images/sobel.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :param list[int] shape:
    :param DistanceFunction|int dist_func: Sobel distance function
    :return: Tensor
    """

    tensor = convolve(ValueMask.conv2d_blur, tensor, shape)

    x = convolve(ValueMask.conv2d_sobel_x, tensor, shape, with_normalize=False)
    y = convolve(ValueMask.conv2d_sobel_y, tensor, shape, with_normalize=False)

    out = tf.abs(normalize(distance(x, y, dist_func)) * 2 - 1)

    fudge = -2

    out = offset(out, shape, x=fudge, y=fudge)

    return out


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

    reference = value_map(tensor, shape, keep_dims=True)

    x = normalize(1 - convolve(ValueMask.conv2d_sobel_x, reference, [height, width, 1]))
    y = normalize(convolve(ValueMask.conv2d_sobel_y, reference, [height, width, 1]))

    z = 1 - tf.abs(normalize(tf.sqrt(x * x + y * y)) * 2 - 1) * .5 + .5

    return tf.stack([x[:, :, 0], y[:, :, 0], z[:, :, 0]], 2)


def value_map(tensor, shape, keep_dims=False, extend_range=False):
    """
    Create a grayscale value map from the given image Tensor by reducing the sum across channels.

    Return value ranges between 0 and 1.

    :param Tensor tensor:
    :param list[int] shape:
    :param bool keep_dims: If True, don't collapse the channel dimension.
    :param bool extend_range: If True, use an extended value range between -1 and 1.
    :return: Tensor of shape (height, width), or (height, width, channels) if keep_dims was True.
    """

    return normalize(tf.reduce_sum(tensor, len(shape) - 1, keep_dims=keep_dims), extend_range=extend_range)


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

    # values = value_map(tensor, shape, keep_dims=True)
    # values = tf.minimum(tf.maximum(tensor, 0.0), 1.0)  # TODO: Get this to work with HDR data
    values = tensor

    # https://stackoverflow.com/a/34143927
    binned_values = tf.cast(tf.reshape(values * (bins - 1), [-1]), tf.int32)
    ones = tf.ones_like(binned_values, dtype=tf.int32)
    counts = tf.unsorted_segment_sum(ones, binned_values, bins)

    out = tf.gather(counts, tf.cast(values[:, :] * (bins - 1), tf.int32))

    return tf.ones(shape) * normalize(tf.cast(out, tf.float32))


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


def conv_feedback(tensor, shape, iterations=50, alpha=.5):
    """
    Conv2d feedback loop

    :param Tensor tensor:
    :return: Tensor
    """

    iterations = 100

    half_shape = [int(shape[0] * .5), int(shape[1] * .5), shape[2]]

    convolved = offset(_downsample(tensor, shape, half_shape), half_shape, x=iterations * -3, y=iterations * -3)

    for i in range(iterations):
        convolved = convolve(ValueMask.conv2d_blur, convolved, half_shape)
        convolved = convolve(ValueMask.conv2d_sharpen, convolved, half_shape)

    convolved = normalize(convolved)

    up = tf.maximum((convolved - .5) * 2, 0.0)

    down = tf.minimum(convolved * 2, 1.0)

    return blend(tensor, resample(up + (1.0 - down), shape), alpha)


def morph(a, b, g, dist_func=DistanceFunction.euclidean, spline_order=1):
    """
    Linear or cosine interpolation using a specified distance function

    :param Tensor a:
    :param Tensor b:
    :param float|Tensor g: Blending gradient a to b (0..1)
    :param DistanceFunction|int|str dist_func: Distance function (1=Euclidean, 2=Manhattan, 3=Chebyshev)
    :param int spline_order: 1=Linear, 2=Cosine
    """

    if spline_order not in (1, 2):
        raise ValueError("Can't interpolate with spline order {0}".format(spline_order))

    elif spline_order == 1:
        component_func = _linear_components

    elif spline_order == 2:
        component_func = _cosine_components

    x, y = component_func(a, b, g)

    return distance(x, y, dist_func)


def distance(a, b, func):
    """
    Compute the distance from a to b, using the specified function.

    :param Tensor a:
    :param Tensor b:
    :param DistanceFunction|int|str func: Distance function (1=Euclidean, 2=Manhattan, 3=Chebyshev)
    :return: Tensor
    """

    if isinstance(func, int):
        func = DistanceFunction(func)

    elif isinstance(func, str):
        func = DistanceFunction[func]

    if func == DistanceFunction.euclidean:
        dist = tf.sqrt(a * a + b * b)

    elif func == DistanceFunction.manhattan:
        dist = tf.abs(a) + tf.abs(b)

    elif func == DistanceFunction.chebyshev:
        dist = tf.maximum(tf.abs(a), tf.abs(b))

    else:
        raise ValueError("{0} isn't a distance function.".format(func))

    return dist


def _linear_components(a, b, g):
    return a * (1 - g), b * g


def blend(a, b, g):
    """
    Blend a and b values with linear interpolation.

    :param Tensor a:
    :param Tensor b:
    :param float|Tensor g: Blending gradient a to b (0..1)
    :return Tensor:
    """

    return sum(_linear_components(a, b, g))


def _cosine_components(a, b, g):
    # This guy is great http://paulbourke.net/miscellaneous/interpolation/

    g2 = (1 - tf.cos(g * math.pi)) / 2

    return a * (1 - g2), b * g2


def blend_cosine(a, b, g):
    """
    Blend a and b values with cosine interpolation.

    :param Tensor a:
    :param Tensor b:
    :param float|Tensor g: Blending gradient a to b (0..1)
    :return Tensor:
    """

    return sum(_cosine_components(a, b, g))


def _cubic_components(a, b, c, d, g):
    # This guy is great http://paulbourke.net/miscellaneous/interpolation/

    g2 = g * g

    a0 = d - c - a + b
    a1 = a - b - a0
    a2 = c - a
    a3 = b

    return a0 * g * g2, a1 * g2, a2 * g + a3


def blend_cubic(a, b, c, d, g):
    """
    Blend b and c values with bi-cubic interpolation.

    :param Tensor a:
    :param Tensor b:
    :param Tensor c:
    :param Tensor d:
    :param float|Tensor g: Blending gradient b to c (0..1)
    :return Tensor:
    """

    return sum(_cubic_components(a, b, c, d, g))


def blend_layers(control, shape, feather=1.0, *layers):
    layer_count = len(layers)

    control = normalize(control)

    control *= layer_count
    control_floor = tf.cast(control, tf.int32)

    x_index = row_index(shape)
    y_index = column_index(shape)

    layers = tf.stack(list(layers) + [layers[-1]])
    layer_count += 1

    floor_values = control_floor[:, :, 0]

    # I'm not sure why the mod operation is needed, but tensorflow-cpu explodes without it.
    combined_layer_0 = tf.gather_nd(layers, tf.stack([floor_values % layer_count, y_index, x_index], 2))
    combined_layer_1 = tf.gather_nd(layers, tf.stack([(floor_values + 1) % layer_count, y_index, x_index], 2))

    control_floor_fract = control - tf.floor(control)
    control_floor_fract = tf.minimum(tf.maximum(control_floor_fract - (1.0 - feather), 0.0) / feather, 1.0)

    return blend(combined_layer_0, combined_layer_1, control_floor_fract)


def center_mask(center, edges, shape):
    """
    Blend two image tensors from the center to the edges.

    :param Tensor center:
    :param Tensor edges:
    :param list[int] shape:
    :return: Tensor
    """

    mask = tf.square(singularity(None, shape, dist_func=DistanceFunction.chebyshev))

    return blend_cosine(center, edges, mask)


def voronoi(tensor, shape, diagram_type=1, density=.1, nth=0, dist_func=1, alpha=1.0, with_refract=0.0, inverse=False, xy=None, ridges_hint=False,
            input_dir=None, image_count=None, collage_images=None):
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
    :param DistanceFunction|int dist_func: Voronoi distance function (1=Euclidean, 2=Manhattan, 3=Chebyshev)
    :param bool regions: Assign colors to control points (memory intensive)
    :param float alpha: Blend with original tensor (0.0 = Original, 1.0 = Voronoi)
    :param float with_refract: Domain warp input tensor against resulting voronoi
    :param bool inverse: Invert range brightness values (does not affect hue)
    :param (Tensor, Tensor, int) xy: Bring your own x, y, and point count (You shouldn't normally need this)
    :param float ridges_hint: Adjust output colors to match ridged multifractal output (You shouldn't normally need this)
    :param str input_dir: Input directory containing .jpg and/or .png images, if using collage mode
    :param None|int image_count: Give an explicit image count for collages (Optional)
    :param None|list[Tensor] collage_images: Give an explicit list of collage image tensors (Optional)
    :return: Tensor
    """

    if isinstance(diagram_type, int):
        diagram_type = VoronoiDiagramType(diagram_type)

    elif isinstance(diagram_type, str):
        diagram_type = VoronoiDiagramType[diagram_type]

    if diagram_type == VoronoiDiagramType.collage and not input_dir and not collage_images:
        raise ValueError("--input-dir containing .jpg/.png images must be specified, when using collage mode.")

    original_shape = shape

    shape = [int(shape[0] * .5), int(shape[1] * .5), shape[2]]  # Gotta upsample later, this one devours memory.

    height, width, channels = shape

    if xy is None:
        point_count = int(min(width, height) * density)

        x = tf.random_uniform([point_count]) * width
        y = tf.random_uniform([point_count]) * height

    else:
        if len(xy) == 2:
            x, y = xy
            point_count = len(x)

        else:
            x, y, point_count = xy

        x = tf.cast(tf.stack(x) / 2, tf.float32)
        y = tf.cast(tf.stack(y) / 2, tf.float32)

    value_shape = [height, width, 1, 1]
    x_index = tf.cast(tf.reshape(row_index(shape), value_shape), tf.float32)
    y_index = tf.cast(tf.reshape(column_index(shape), value_shape), tf.float32)

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

    if diagram_type in VoronoiDiagramType.flow_members():
        # If we're using flow with a perfectly tiled grid, it just disappears. Perturbing the points seems to prevent this from happening.
        x_diff += tf.random_normal(shape=tf.shape(x), stddev=.0001, dtype=tf.float32)
        y_diff += tf.random_normal(shape=tf.shape(y), stddev=.0001, dtype=tf.float32)

    dist = distance(x_diff, y_diff, dist_func)

    ###
    if diagram_type not in VoronoiDiagramType.flow_members():
        dist, indices = tf.nn.top_k(dist, k=point_count)
        index = int((nth + 1) * -1)

    ###

    # Seamless alg offset pixels by half image size. Move results slice back to starting points with `offset`:
    offset_kwargs = {
        'x': half_width,
        'y': half_height,
    }

    if diagram_type in (VoronoiDiagramType.range, VoronoiDiagramType.color_range, VoronoiDiagramType.range_regions):
        range_slice = resample(offset(tf.sqrt(normalize(dist[:, :, :, index])), shape, **offset_kwargs), original_shape)

        if inverse:
            range_slice = 1.0 - range_slice

    if diagram_type in (VoronoiDiagramType.regions, VoronoiDiagramType.color_regions, VoronoiDiagramType.range_regions, VoronoiDiagramType.collage):
        regions_slice = offset(indices[:, :, :, index], shape, **offset_kwargs)

    ###
    if diagram_type == VoronoiDiagramType.range:
        range_out = range_slice

    if diagram_type in VoronoiDiagramType.flow_members():
        range_out = tf.reduce_sum(tf.log(dist), 3)

        range_out = resample(offset(range_out, shape, **offset_kwargs), original_shape)

        if diagram_type == VoronoiDiagramType.flow:
            range_out = normalize(range_out)

        else:
            range_out = density_map(range_out, original_shape)

        if inverse:
            range_out = 1.0 - range_out

    if diagram_type in (VoronoiDiagramType.color_range, VoronoiDiagramType.range_regions):
        range_out = blend(tensor * range_slice, range_slice, range_slice)

    if diagram_type == VoronoiDiagramType.regions:
        regions_out = resample(tf.cast(regions_slice, tf.float32), original_shape, spline_order=0)

    if diagram_type in (VoronoiDiagramType.color_regions, VoronoiDiagramType.range_regions):
        colors = tf.gather_nd(tensor, tf.cast(tf.stack([y * 2, x * 2], 1), tf.int32))

        if ridges_hint:
            colors = tf.abs(colors * 2 - 1)

        spline_order = 0 if diagram_type == VoronoiDiagramType.color_regions else 3

        regions_out = resample(tf.reshape(tf.gather(colors, regions_slice), shape), original_shape, spline_order=spline_order)

    if diagram_type == VoronoiDiagramType.collage:
        filenames = [f for f in os.listdir(input_dir) if f.endswith(".png") or f.endswith(".jpg")]

        freq = int(max(math.sqrt(point_count), 1))

        collage_count = image_count or freq
        collage_height = int(max(shape[0] / freq, 1))
        collage_width = int(max(shape[1] / freq, 1))
        collage_shape = [collage_height, collage_width, shape[2]]

        if not collage_images:
            collage_images = []

            for i in range(collage_count):
                index = i if image_count else random.randint(0, len(filenames) - 1)

                collage_input = tf.image.convert_image_dtype(util.load(os.path.join(input_dir, filenames[index])), dtype=tf.float32)
                collage_images.append(_downsample(resample(collage_input, shape), shape, collage_shape))

        out = tf.gather_nd(collage_images,
                           tf.stack([regions_slice[:, :, 0] % collage_count, column_index(shape) % collage_height, row_index(shape) % collage_width], 2))

        out = resample(out, original_shape)

    ###
    if diagram_type == VoronoiDiagramType.range_regions:
        out = blend(regions_out, range_out, tf.square(range_out))

    elif diagram_type in [VoronoiDiagramType.range, VoronoiDiagramType.color_range] + VoronoiDiagramType.flow_members():
        out = range_out

    elif diagram_type in (VoronoiDiagramType.regions, VoronoiDiagramType.color_regions):
        out = regions_out

    if with_refract != 0.0:
        out = refract(tensor, original_shape, displacement=with_refract, reference_x=out)

    if tensor is not None:
        out = blend(tensor, out, alpha)

    return out


def posterize(tensor, levels):
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
        freq = freq_for_shape(freq, shape)

    small_shape = [int(shape[0] / freq[0]), int(shape[1] / freq[1]), shape[2]]

    y_index = tf.tile(column_index(small_shape) * freq[0], [freq[0], freq[0]])
    x_index = tf.tile(row_index(small_shape) * freq[1], [freq[0], freq[0]])

    tiled = tf.gather_nd(tensor, tf.stack([y_index, x_index], 2))

    tiled = resample(tiled, shape, spline_order=1)

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

    x_index = (x_offset + row_index(output_shape)) % input_width
    y_index = (y_offset + column_index(output_shape)) % input_height

    return tf.gather_nd(tensor, tf.stack([y_index, x_index], 2))


def row_index(shape):
    """
    Generate an X index for the given tensor.

    .. code-block:: python

      [
        [ 0, 1, 2, ... width-1 ],
        [ 0, 1, 2, ... width-1 ],
        ... (x height)
      ]

    .. image:: images/row_index.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param list[int] shape:
    :return: Tensor of shape (height, width)
    """

    height = shape[0]
    width = shape[1]

    row_identity = tf.cumsum(tf.ones([width], dtype=tf.int32), exclusive=True)
    row_identity = tf.reshape(tf.tile(row_identity, [height]), [height, width])

    return row_identity


def column_index(shape):
    """
    Generate a Y index for the given tensor.

    .. code-block:: python

      [
        [ 0, 0, 0, ... ],
        [ 1, 1, 1, ... ],
        [ n, n, n, ... ],
        ...
        [ height-1, height-1, height-1, ... ]
      ]

    .. image:: images/column_index.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param list[int] shape:
    :return: Tensor of shape (height, width)
    """

    height = shape[0]
    width = shape[1]

    column_identity = tf.ones([width], dtype=tf.int32)
    column_identity = tf.tile(column_identity, [height])
    column_identity = tf.reshape(column_identity, [height, width])
    column_identity = tf.cumsum(column_identity, exclusive=True)

    return column_identity


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


def freq_for_shape(freq, shape):
    """
    Given a base frequency as int, generate noise frequencies for each spatial dimension.

    :param int freq: Base frequency
    :param list[int] shape: List of spatial dimensions, e.g. [height, width]
    """

    height = shape[0]
    width = shape[1]

    if height == width:
        return [freq, freq]

    elif height < width:
        return [freq, int(freq * width / height)]

    else:
        return [int(freq * height / width), freq]


def warp(tensor, shape, freq, octaves=5, displacement=1, spline_order=3, time=0.0):
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
    """

    for octave in range(1, octaves + 1):
        multiplier = 2 ** octave

        base_freq = [int(f * .5 * multiplier) for f in freq]

        if base_freq[0] >= shape[0] or base_freq[1] >= shape[1]:
            break

        tensor = refract(tensor, shape, displacement=displacement / multiplier, warp_freq=base_freq, spline_order=spline_order, time=time)

    return tensor


def sobel(tensor, shape, dist_func=1, rgb=False):
    """
    Colorized sobel edges.

    :param Tensor tensor:
    :param list[int] shape:
    :param DistanceFunction|int dist_func: Sobel distance function
    :param bool rgb:
    """

    if rgb:
        return sobel_operator(tensor, shape, dist_func)

    else:
        return outline(tensor, shape, dist_func, True)


def outline(tensor, shape, sobel_func=1, invert=False):
    """
    Superimpose sobel operator results (cartoon edges)

    :param Tensor tensor:
    :param list[int] shape:
    :param DistanceFunction|int sobel_func: Sobel distance function
    """

    height, width, channels = shape

    value_shape = [height, width, 1]

    values = value_map(tensor, shape, keep_dims=True)

    edges = sobel_operator(values, value_shape, dist_func=sobel_func)

    if invert:
        edges = 1 - edges

    return edges * tensor


def glowing_edges(tensor, shape, sobel_func=2, alpha=1.0):
    """
    """

    height, width, channels = shape

    value_shape = [height, width, 1]

    edges = value_map(tensor, shape, keep_dims=True)

    edges = posterize(edges, random.randint(3, 5))

    edges = 1.0 - sobel_operator(edges, value_shape, dist_func=sobel_func)

    edges = tf.minimum(edges * 8, 1.0) * tf.minimum(tensor * 1.25, 1.0)

    edges = bloom(edges, shape, alpha=.5)

    edges = normalize(edges + convolve(ValueMask.conv2d_blur, edges, shape))

    return blend(tensor, 1.0 - ((1.0 - edges) * (1.0 - tensor)), alpha)


def singularity(tensor, shape, diagram_type=1, **kwargs):
    """
    Return the range diagram for a single voronoi point, approximately centered.

    :param Tensor tensor:
    :param list[int] shape:
    :param VoronoiDiagramType|int diagram_type:
    :param DistanceFunction|int dist_func:

    Additional kwargs will be sent to the `voronoi` function.
    """

    x, y = point_cloud(1, PointDistribution.square, shape)

    return voronoi(tensor, shape, diagram_type=diagram_type, xy=(x, y, 1), **kwargs)


def vortex(tensor, shape, displacement=64.0, time=0.0):
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

    x = convolve(ValueMask.conv2d_deriv_x, displacement_map, value_shape, with_normalize=False)
    y = convolve(ValueMask.conv2d_deriv_y, displacement_map, value_shape, with_normalize=False)

    fader = singularity(None, value_shape, dist_func=DistanceFunction.chebyshev, inverse=True)

    x *= fader
    y *= fader

    warped = refract(tensor, shape, displacement=displacement * simplex.random(time), reference_x=x, reference_y=y)

    return warped


def aberration(tensor, shape, displacement=.005, time=0.0):
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

    x_index = row_index(shape)
    y_index = column_index(shape)

    x_index_float = tf.cast(x_index, tf.float32)

    separated = []

    displacement_pixels = int(width * displacement * simplex.random(time))

    mask = tf.pow(tf.squeeze(singularity(None, [shape[0], shape[1], 1])), 3)

    gradient = normalize(x_index_float)

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
            offset_x_index = blend(offset_x_index, x_index_float, gradient)

        elif i == 2:
            # Right (blue)
            offset_x_index = blend(x_index_float, offset_x_index, gradient)

        # Fade effect towards center
        offset_x_index = tf.cast(blend_cosine(x_index_float, offset_x_index, mask), tf.int32)

        separated.append(tf.gather_nd(tensor[:, :, i], tf.stack([y_index, offset_x_index], 2)))

    tensor = tf.stack(separated, 2)

    # Restore original colors
    return tf.image.adjust_hue(tensor, -shift)


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
    blurred = _downsample(blurred, shape, [max(int(height * .01), 1), max(int(width * .01), 1), channels]) * 4.0
    blurred = resample(blurred, shape)
    blurred = offset(blurred, shape, x=int(shape[1] * -.005), y=int(shape[0] * -.005))

    return blend(tensor, 1.0 - (1.0 - tensor) * (1.0 - blurred), alpha)


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


def pop(tensor, shape):
    """
    Pop art filter

    :param Tensor tensor:
    :param list[int] shape:
    """

    images = []

    freq = random.randint(1, 3) * 2

    ref = _downsample(resample(tensor, shape), shape, [int(shape[0] / (freq * 2)), int(shape[1] / (freq * 2)), shape[2]])

    for i in range(freq * freq):
        image = posterize(ref, random.randint(3, 6))
        image = image % tf.random_normal([3], mean=.5, stddev=.25)
        images.append(image)

    x, y = point_cloud(freq, distrib=PointDistribution.square, shape=shape, corners=True)

    out = voronoi(None, shape, diagram_type=VoronoiDiagramType.collage, xy=(x, y, len(x)), nth=random.randint(0, 3), collage_images=images, image_count=4)

    return outline(out, shape, sobel_func=1)


def offset(tensor, shape, x=0, y=0):
    """
    """

    x_index = row_index(shape)
    y_index = column_index(shape)

    return tf.gather_nd(tensor, tf.stack([(y_index + y) % shape[0], (x_index + x) % shape[1]], 2))


def reverb(tensor, shape, octaves, iterations=1, ridges=True):
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

            out += expand_tile(_downsample(reference, shape, octave_shape), octave_shape, shape) / multiplier

    return normalize(out)


def light_leak(tensor, shape, alpha=.25, time=0.0):
    """
    """

    x, y = point_cloud(6, distrib=PointDistribution.grid_members()[random.randint(0, len(PointDistribution.grid_members()) - 1)], shape=shape, time=time)

    leak = voronoi(tensor, shape, diagram_type=VoronoiDiagramType.color_regions, xy=(x, y, len(x)))
    leak = wormhole(leak, shape, kink=1.0, input_stride=.25)

    leak = bloom(leak, shape, 1.0)

    for _ in range(3):
        leak = convolve(ValueMask.conv2d_blur, leak, shape)

    leak = 1 - ((1 - tensor) * (1 - leak))

    leak = center_mask(tensor, leak, shape)
    leak = center_mask(tensor, leak, shape)

    return blend(tensor, leak, alpha)


def vignette(tensor, shape, brightness=0.0, alpha=1.0):
    """
    """

    for _ in range(3):
        edges = convolve(ValueMask.conv2d_blur, tensor, shape)

    edges = center_mask(edges, tf.ones(shape) * brightness, shape)
    edges = center_mask(tensor, edges, shape)

    return blend(tensor, edges, alpha)


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
        reference = value_map(tensor, shape, keep_dims=True)

    else:
        reference = value_map(reference, shape, keep_dims=True)

    value_shape = [height, width, 1]

    grad = random.random()

    x = convolve(ValueMask.conv2d_sobel_x, reference, value_shape, with_normalize=True)
    y = convolve(ValueMask.conv2d_sobel_y, reference, value_shape, with_normalize=True)

    shade = normalize(morph(x, y, grad, dist_func=DistanceFunction.manhattan)) * 2.0 - 1.0

    down = tf.sqrt(tf.minimum(shade + 1.0, 1.0))
    up = tf.square(tf.maximum(shade * .5, 0.0))

    return blend(tensor, tensor * down * (1.0 - (1.0 - up) * (1.0 - tensor)), alpha)


def glyph_map(tensor, shape, mask=None, colorize=True, zoom=1, alpha=1.0):
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
            glyph, brightness = masks.mask_values(mask, glyph_shape, uv_noise=np.ones(glyph_shape) * i / levels)

            glyphs.append(glyph)
            sums.append(brightness)

        glyphs = [g for _, g in sorted(zip(sums, glyphs))]

    in_shape = [int(shape[0] / zoom), int(shape[1] / zoom), shape[2]]

    height, width, channels = in_shape

    # Figure out how many glyphs it will take approximately to cover the image
    uv_shape = [int(in_shape[0] / glyph_shape[0]) or 1, int(in_shape[1] / glyph_shape[1] or 1), 1]

    # Generate a value map, multiply by len(glyphs) to create glyph index offsets
    value_shape = [height, width, 1]
    uv_noise = _downsample(value_map(tensor, in_shape, keep_dims=True), value_shape, uv_shape)

    approx_shape = [glyph_shape[0] * uv_shape[0], glyph_shape[1] * uv_shape[1], 1]

    uv_noise = resample(uv_noise, approx_shape, spline_order=0)

    x_index = row_index(approx_shape) % glyph_shape[1]
    y_index = column_index(approx_shape) % glyph_shape[0]

    glyph_count = len(glyphs)
    z_index = tf.cast(uv_noise[:, :, 0] * glyph_count, tf.int32) % glyph_count

    resample_order = 1 if mask == ValueMask.truetype else 0

    out = resample(tf.gather_nd(glyphs, tf.stack([z_index, y_index, x_index], 2)), [shape[0], shape[1], 1], spline_order=resample_order)

    if not colorize:
        return out * tf.ones(shape)

    out *= resample(_downsample(tensor, shape, [uv_shape[0], uv_shape[1], channels]), shape, spline_order=0)

    if alpha == 1.0:
        return out

    return blend(tensor, out, alpha)


def pixel_sort(tensor, shape, angled=False, darkest=False):
    """
    Pixel sort effect

    :param Tensor tensor:
    :param list[int] shape:
    :param bool angled: If True, sort along a random angle.
    :param bool darkest: If True, order by darkest instead of brightest
    :return Tensor:
    """

    if angled:
        angle = random.randint(0, 360) if isinstance(angled, bool) else angled

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

        padded = tf.image.resize_image_with_crop_or_pad(tensor, want_length, want_length)

        rotated = tf.contrib.image.rotate(padded, angle, 'BILINEAR')

    else:
        padded_shape = shape

        rotated = tensor

    # Find index of brightest pixel
    x_index = tf.expand_dims(tf.argmax(value_map(rotated, padded_shape), axis=1, output_type=tf.int32), -1)

    # Add offset index to row index
    x_index = (row_index(padded_shape) - tf.tile(x_index, [1, padded_shape[1]])) % padded_shape[1]

    # Sort pixels
    sorted_channels = [tf.nn.top_k(rotated[:, :, c], padded_shape[1])[0] for c in range(padded_shape[2])]

    # Apply offset
    sorted_channels = tf.gather_nd(tf.stack(sorted_channels, 2), tf.stack([column_index(padded_shape), x_index], 2))

    if angle:
        # Rotate back to original orientation
        sorted_channels = tf.contrib.image.rotate(sorted_channels, -angle, 'BILINEAR')

        # Crop to original size
        sorted_channels = tf.image.resize_image_with_crop_or_pad(sorted_channels, height, width)
 
    # Blend with source image
    tensor = tf.maximum(tensor, sorted_channels)

    if darkest:
        tensor = 1.0 - tensor

    return tensor


def rotate(tensor, shape, angle=None):
    """Rotate the image. This breaks seamless edges."""

    height, width, channels = shape

    if angle is None:
        angle = random.random() * 360.0

    want_length = max(height, width) * 2

    padded_shape = [want_length, want_length, channels]

    padded = expand_tile(tensor, shape, padded_shape)

    rotated = tf.contrib.image.rotate(padded, angle, 'BILINEAR')

    return tf.image.resize_image_with_crop_or_pad(rotated, height, width)


def sketch(tensor, shape, time=0.0):
    """
    Pencil sketch effect

    :param Tensor tensor:
    :param list[int] shape:
    :return Tensor:
    """

    value_shape = [shape[0], shape[1], 1]

    values = value_map(tensor, value_shape, keep_dims=True)
    values = tf.image.adjust_contrast(values, 2.0)

    values = tf.minimum(values, 1.0)
    values = tf.maximum(values, 0.0)

    outline = 1.0 - derivative(values, value_shape)
    outline = tf.minimum(outline, 1.0 - derivative(1.0 - values, value_shape))
    outline = normalize(outline)

    values = vignette(values, value_shape, 1.0, .875)

    crosshatch = 1.0 - worms(1.0 - values, value_shape, behavior=2, density=125, duration=.5, stride=1, stride_deviation=.25, alpha=1.0)
    crosshatch = normalize(crosshatch)

    combined = blend(crosshatch, outline, .625)
    combined = warp(combined, value_shape, [int(shape[0] * .125) or 1, int(shape[1] * .125) or 1], octaves=1, displacement=.0025, time=time)
    combined *= combined

    return combined * tf.ones(shape)


def lowpoly(tensor, shape, distrib=0, freq=10, time=0.0):
    """Low-poly art style effect"""

    xy = point_cloud(freq, distrib=distrib, shape=shape, drift=1.0, time=time)

    distance = voronoi(tensor, shape, nth=1, xy=xy)
    color = voronoi(tensor, shape, diagram_type=VoronoiDiagramType.color_regions, xy=xy)

    return normalize(blend(distance, color, .5))


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
        tensor = tf.image.resize_image_with_crop_or_pad(tensor, have_length, have_length)

    if length != have_length:
        tensor = resample(tensor, [length, length, channels])

    return tensor


def shape_from_file(filename):
    """
    Get image dimensions from a file, using PIL, to avoid adding to the TensorFlow graph.
    """

    image = Image.open(filename)

    input_width, input_height = image.size

    return [input_height, input_width, len(image.getbands())]
