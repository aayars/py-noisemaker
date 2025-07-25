"""Low-level effects library for Noisemaker"""

import math
import random

import numpy as np
import tensorflow as tf

from noisemaker.constants import (
    ColorSpace,
    DistanceMetric,
    InterpolationType,
    PointDistribution,
    ValueDistribution,
    ValueMask,
    VoronoiDiagramType,
    WormBehavior
)
from noisemaker.effects_registry import effect
from noisemaker.glyphs import load_glyphs
from noisemaker.palettes import PALETTES as palettes
from noisemaker.points import point_cloud

import noisemaker.masks as masks
import noisemaker.simplex as simplex
import noisemaker.util as util
import noisemaker.value as value


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


@effect()
def erosion_worms(tensor, shape, density=50, iterations=50, contraction=1.0, quantize=False, alpha=.25, inverse=False, xy_blend=False, time=0.0, speed=1.0):
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

    values = value.value_map(value.convolve(kernel=ValueMask.conv2d_blur, tensor=tensor, shape=shape), shape, keepdims=True)

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

        if quantize:
            g_x = tf.floor(g_x)
            g_y = tf.floor(g_y)

        length = value.distance(g_x, g_y, DistanceMetric.euclidean) * contraction

        x_dir = value.blend(x_dir, g_x / length, inertia)
        y_dir = value.blend(y_dir, g_y / length, inertia)

        # step
        x = (x + x_dir) % width
        y = (y + y_dir) % height

    out = value.clamp01(out)

    if inverse:
        out = 1.0 - out

    if xy_blend:
        tensor = value.blend(shadow(tensor, shape), reindex(tensor, shape, 1), xy_blend * values)

    return value.blend(tensor, out, alpha)


@effect()
def reindex(tensor, shape, displacement=.5, time=0.0, speed=1.0):
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

    reference = value.value_map(tensor, shape)

    mod = min(height, width)
    x_offset = tf.cast((reference * displacement * mod + reference) % width, tf.int32)
    y_offset = tf.cast((reference * displacement * mod + reference) % height, tf.int32)

    tensor = tf.gather_nd(tensor, tf.stack([y_offset, x_offset], 2))

    return tensor


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

    value_shape = value.value_shape(shape)

    if reference is None:
        reference = value.values(freq=freq, shape=value_shape, spline_order=spline_order)

    # Twist index, borrowed from worms. TODO refactor me?
    index = value.value_map(reference, shape, with_normalize=False) * math.tau * kink * simplex.random(time, speed=speed)

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
def color_map(tensor, shape, clut=None, horizontal=False, displacement=.5, time=0.0, speed=1.0):
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

    reference = value.value_map(tensor, shape) * displacement

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
          drunkenness=0.0, quantize=False, colors=None, time=0.0, speed=1.0):
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
    :param bool quantize: Quantize rotations to 45 degree increments
    :param Tensor colors: Optional starting colors, if not from `tensor`.
    :return: Tensor
    """

    behavior = value.coerce_enum(behavior, WormBehavior)

    height, width, channels = shape

    count = int(max(width, height) * density)

    worms_y = tf.random.uniform([count]) * (height - 1)
    worms_x = tf.random.uniform([count]) * (width - 1)
    worms_stride = tf.random.normal([count], mean=stride, stddev=stride_deviation) * (max(width, height)/1024.0)

    color_source = colors if colors is not None else tensor

    colors = tf.gather_nd(color_source, tf.cast(tf.stack([worms_y, worms_x], 1), tf.int32))

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
        WormBehavior.meandering: lambda n: value.periodic_value(time * speed, tf.random.uniform([count]))
    }

    worms_rot = rots[behavior](count)

    index = value.value_map(tensor, shape) * math.tau * kink

    iterations = int(math.sqrt(min(width, height)) * duration)

    out = tf.zeros(shape)

    scatter_shape = tf.shape(tensor)  # Might be different than `shape` due to clut

    # Make worms!
    for i in range(iterations):
        if drunkenness:
            start = int(min(shape[0], shape[1]) * time * speed + i * speed * 10)

            worms_rot += (value.periodic_value(start, tf.random.uniform([count])) * 2.0 - 1.0) * drunkenness * math.pi

        worm_positions = tf.cast(tf.stack([worms_y % height, worms_x % width], 1), tf.int32)

        exposure = 1 - abs(1 - i / (iterations - 1) * 2)  # Makes linear gradient [ 0 .. 1 .. 0 ]

        out += tf.scatter_nd(worm_positions, colors * exposure, scatter_shape)

        next_position = tf.gather_nd(index, worm_positions) + worms_rot

        if quantize:
            next_position = tf.math.round(next_position)

        worms_y = (worms_y + tf.cos(next_position) * worms_stride) % height
        worms_x = (worms_x + tf.sin(next_position) * worms_stride) % width

    out = tf.image.convert_image_dtype(out, tf.float32, saturate=True)

    return value.blend(tensor, tf.sqrt(value.normalize(out)), alpha)


@effect()
def wormhole(tensor, shape, kink=1.0, input_stride=1.0, alpha=1.0, time=0.0, speed=1.0):
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

    values = value.value_map(tensor, shape, with_normalize=False)

    degrees = values * math.tau * kink
    stride = 1024 * input_stride

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
def derivative(tensor, shape, dist_metric=DistanceMetric.euclidean, with_normalize=True, alpha=1.0, time=0.0, speed=1.0):
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

    x = value.convolve(kernel=ValueMask.conv2d_deriv_x, tensor=tensor, shape=shape, with_normalize=False)
    y = value.convolve(kernel=ValueMask.conv2d_deriv_y, tensor=tensor, shape=shape, with_normalize=False)

    out = value.distance(x, y, dist_metric)

    if with_normalize:
        out = value.normalize(out)

    if alpha == 1.0:
        return out

    return value.blend(tensor, out, alpha)


@effect("sobel")
def sobel_operator(tensor, shape, dist_metric=DistanceMetric.euclidean, time=0.0, speed=1.0):
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

    tensor = value.convolve(kernel=ValueMask.conv2d_blur, tensor=tensor, shape=shape)

    x = value.convolve(kernel=ValueMask.conv2d_sobel_x, tensor=tensor, shape=shape, with_normalize=False)
    y = value.convolve(kernel=ValueMask.conv2d_sobel_y, tensor=tensor, shape=shape, with_normalize=False)

    out = tf.abs(value.normalize(value.distance(x, y, dist_metric)) * 2 - 1)

    fudge = -1

    out = value.offset(out, shape, x=fudge, y=fudge)

    return out


@effect()
def normal_map(tensor, shape, time=0.0, speed=1.0):
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

    reference = value.value_map(tensor, shape, keepdims=True)

    value_shape = value.value_shape(shape)

    x = value.normalize(1 - value.convolve(kernel=ValueMask.conv2d_sobel_x, tensor=reference, shape=value_shape))
    y = value.normalize(value.convolve(kernel=ValueMask.conv2d_sobel_y, tensor=reference, shape=value_shape))

    z = 1 - tf.abs(value.normalize(tf.sqrt(x * x + y * y)) * 2 - 1) * .5 + .5

    return tf.stack([x[:, :, 0], y[:, :, 0], z[:, :, 0]], 2)


@effect()
def density_map(tensor, shape, time=0.0, speed=1.0):
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

    # values = value.value_map(tensor, shape, keepdims=True)
    # values = tf.minimum(tf.maximum(tensor, 0.0), 1.0)  # TODO: Get this to work with HDR data
    values = value.normalize(tensor)

    # https://stackoverflow.com/a/34143927
    binned_values = tf.cast(tf.reshape(values * (bins - 1), [-1]), tf.int32)
    ones = tf.ones_like(binned_values, dtype=tf.int32)
    counts = tf.math.unsorted_segment_sum(ones, binned_values, bins)

    out = tf.gather(counts, tf.cast(values[:, :] * (bins - 1), tf.int32))

    return tf.ones(shape) * value.normalize(tf.cast(out, tf.float32))


@effect()
def jpeg_decimate(tensor, shape, iterations=25, time=0.0, speed=1.0):
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
def conv_feedback(tensor, shape, iterations=50, alpha=.5, time=0.0, speed=1.0):
    """
    Conv2d feedback loop

    :param Tensor tensor:
    :return: Tensor
    """

    iterations = 100

    half_shape = [int(shape[0] * .5), int(shape[1] * .5), shape[2]]

    convolved = value.proportional_downsample(tensor, shape, half_shape)

    for i in range(iterations):
        convolved = value.convolve(kernel=ValueMask.conv2d_blur, tensor=convolved, shape=half_shape)
        convolved = value.convolve(kernel=ValueMask.conv2d_sharpen, tensor=convolved, shape=half_shape)

    convolved = value.normalize(convolved)

    up = tf.maximum((convolved - .5) * 2, 0.0)

    down = tf.minimum(convolved * 2, 1.0)

    return value.blend(tensor, value.resample(up + (1.0 - down), shape), alpha)


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


def center_mask(center, edges, shape, dist_metric=DistanceMetric.chebyshev, power=2):
    """
    Blend two image tensors from the center to the edges.

    :param Tensor center:
    :param Tensor edges:
    :param list[int] shape:
    :param int power:
    :return: Tensor
    """

    mask = tf.pow(value.singularity(None, shape, dist_metric=dist_metric), power)

    return value.blend(center, edges, mask)


@effect()
def posterize(tensor, shape, levels=9, time=0.0, speed=1.0):
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

    if levels == 0:
        return tensor

    if shape[-1] == 3:
       tensor = util.from_srgb(tensor)

    tensor *= levels

    tensor += (1/levels) * .5

    tensor = tf.floor(tensor)

    tensor /= levels

    if shape[-1] == 3:
       tensor = util.from_linear_rgb(tensor)

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

    if isinstance(freq, int):
        freq = value.freq_for_shape(freq, shape)

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

        tensor = value.refract(tensor, shape, displacement=displacement / multiplier,
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
def outline(tensor, shape, sobel_metric=1, invert=False, time=0.0, speed=1.0):
    """
    Superimpose sobel operator results (cartoon edges)

    :param Tensor tensor:
    :param list[int] shape:
    :param DistanceMetric|int sobel_metric: Sobel distance metric
    """

    height, width, channels = shape

    value_shape = value.value_shape(shape)

    values = value.value_map(tensor, shape, keepdims=True)

    edges = sobel_operator(values, value_shape, dist_metric=sobel_metric)

    if invert:
        edges = 1.0 - edges

    return edges * tensor


@effect()
def glowing_edges(tensor, shape, sobel_metric=2, alpha=1.0, time=0.0, speed=1.0):
    """
    """

    height, width, channels = shape

    value_shape = value.value_shape(shape)

    edges = value.value_map(tensor, shape, keepdims=True)

    edges = posterize(edges, value_shape, random.randint(3, 5))

    edges = 1.0 - sobel_operator(edges, value_shape, dist_metric=sobel_metric)

    edges = tf.minimum(edges * 8, 1.0) * tf.minimum(tensor * 1.25, 1.0)

    edges = bloom(edges, shape, alpha=.5)

    edges = value.normalize(edges + value.convolve(kernel=ValueMask.conv2d_blur, tensor=edges, shape=shape))

    return value.blend(tensor, 1.0 - ((1.0 - edges) * (1.0 - tensor)), alpha)


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

    value_shape = value.value_shape(shape)

    displacement_map = value.singularity(None, value_shape)
    displacement_map = value.normalize(displacement_map)

    x = value.convolve(kernel=ValueMask.conv2d_deriv_x, tensor=displacement_map, shape=value_shape, with_normalize=False)
    y = value.convolve(kernel=ValueMask.conv2d_deriv_y, tensor=displacement_map, shape=value_shape, with_normalize=False)

    fader = value.singularity(None, value_shape, dist_metric=DistanceMetric.chebyshev, inverse=True)
    fader = value.normalize(fader)

    x *= fader
    y *= fader

    warped = value.refract(tensor, shape,
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

    mask = tf.pow(tf.squeeze(value.singularity(None, [shape[0], shape[1], 1])), 3)

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
def bloom(tensor, shape, alpha=.5, time=0.0, speed=1.0):
    """
    Bloom effect

    Input image must currently be square (sorry).

    :param Tensor tensor:
    :param list[int] shape:
    :param float alpha:
    """

    height, width, channels = shape

    blurred = value.clamp01(tensor * 2.0 - 1.0)
    blurred = value.proportional_downsample(blurred, shape, [max(int(height / 100), 1), max(int(width / 100), 1), channels]) * 4.0
    blurred = value.resample(blurred, shape)

    blurred = value.offset(blurred, shape, x=int(tf.cast(width, tf.float32) * -.05), y=int(tf.cast(shape[0], tf.float32) * -.05))

    blurred = tf.image.adjust_brightness(blurred, .25)
    blurred = tf.image.adjust_contrast(blurred, 1.5)

    return value.blend(value.clamp01(tensor), value.clamp01((tensor + blurred) * .5), alpha)


@effect()
def dla(tensor, shape, padding=2, seed_density=.01, density=.125, xy=None, alpha=1.0, time=0.0, speed=1.0):
    """
    Diffusion-limited aggregation. Renders with respect to the `time` param (0..1)

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
        seed_count = int(math.sqrt(int(half_height * seed_density) or 1))
        x, y = point_cloud(seed_count, distrib=PointDistribution.random, shape=shape, time=time, speed=speed)

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
            walkers.append((int(random.random() * half_height), int(random.random() * half_width)))

    iterations = int(math.sqrt(walkers_count) * time * time)

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

    out = value.convolve(kernel=ValueMask.conv2d_blur, tensor=tf.scatter_nd(tf.stack(unique) * int(1/scale), hot, [height, width, channels]), shape=shape)

    return value.blend(tensor, out * tensor, alpha)


@effect()
def wobble(tensor, shape, time=0.0, speed=1.0):
    """
    Move the entire image around
    """

    x_offset = tf.cast(simplex.random(time=time, speed=speed * .5) * shape[1], tf.int32)
    y_offset = tf.cast(simplex.random(time=time, speed=speed * .5) * shape[0], tf.int32)

    return value.offset(tensor, shape, x=x_offset, y=y_offset)


@effect()
def reverb(tensor, shape, octaves=2, iterations=1, ridges=True, time=0.0, speed=1.0):
    """
    Multi-octave "reverberation" of input image tensor

    :param Tensor tensor:
    :param float[int] shape:
    :param int octaves:
    :param int iterations: Re-reverberate N times. Gratuitous!
    :param bool ridges: abs(tensor * 2 - 1) -- False to not do that.
    """

    if not octaves:
        return tensor

    height, width, channels = shape

    if ridges:
        reference = 1.0 - tf.abs(tensor * 2 - 1)

    else:
        reference = tensor

    out = reference

    for i in range(iterations):
        for octave in range(1, octaves + 1):
            multiplier = 2 ** octave

            octave_shape = [int(height / multiplier) or 1, int(width / multiplier) or 1, channels]

            if not all(octave_shape):
                break

            layer = value.proportional_downsample(reference, shape, octave_shape)

            out += expand_tile(layer, octave_shape, shape) / multiplier

    return value.normalize(out)


@effect()
def light_leak(tensor, shape, alpha=.25, time=0.0, speed=1.0):
    """
    """

    x, y = point_cloud(6, distrib=PointDistribution.grid_members()[random.randint(0, len(PointDistribution.grid_members()) - 1)],
                       drift=.05, shape=shape, time=time, speed=speed)

    leak = value.voronoi(tensor, shape, diagram_type=VoronoiDiagramType.color_regions, xy=(x, y, len(x)))
    leak = wormhole(leak, shape, kink=1.0, input_stride=.25)

    leak = bloom(leak, shape, 1.0)

    leak = 1 - ((1 - tensor) * (1 - leak))

    leak = center_mask(tensor, leak, shape, 4)

    return vaseline(value.blend(tensor, leak, alpha), shape, alpha)


@effect()
def vignette(tensor, shape, brightness=0.0, alpha=1.0, time=0.0, speed=1.0):
    """
    """

    tensor = value.normalize(tensor)

    edges = center_mask(tensor, tf.ones(shape) * brightness, shape, dist_metric=DistanceMetric.euclidean)

    return value.blend(tensor, edges, alpha)


@effect()
def vaseline(tensor, shape, alpha=1.0, time=0.0, speed=1.0):
    """
    """

    return value.blend(tensor, center_mask(tensor, bloom(tensor, shape, 1.0), shape), alpha)


@effect()
def shadow(tensor, shape, alpha=1.0, reference=None, time=0.0, speed=1.0):
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

    reference = value.value_map(reference, shape, keepdims=True)

    value_shape = value.value_shape(shape)

    x = value.convolve(kernel=ValueMask.conv2d_sobel_x, tensor=reference, shape=value_shape)
    y = value.convolve(kernel=ValueMask.conv2d_sobel_y, tensor=reference, shape=value_shape)

    shade = value.normalize(value.distance(x, y, DistanceMetric.euclidean))

    shade = value.convolve(kernel=ValueMask.conv2d_sharpen, tensor=shade, shape=value_shape, alpha=.5)

    # Ramp values to not be so imposing visually
    highlight = tf.math.square(shade)

    # Darken and brighten original pixel values
    shade = (1.0 - ((1.0 - tensor) * (1.0 - highlight))) * shade

    if channels == 1:
        tensor = value.blend(tensor, shade, alpha)
    elif channels == 2:
        tensor = tf.stack([value.blend(tensor[:, :, 0], shade, alpha), tensor[:, :, 1]], 2)
    elif channels in (3, 4):
        if channels == 4:
            a = tensor[:, :, 0]
            tensor = tf.stack([tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]], 2)

        # Limit effect to just the brightness channel
        tensor = tf.image.rgb_to_hsv([tensor])[0]

        tensor = tf.stack([tensor[:, :, 0], tensor[:, :, 1],
                           value.blend(tensor[:, :, 2], tf.image.rgb_to_hsv([shade])[0][:, :, 2], alpha)], 2)

        tensor = tf.image.hsv_to_rgb([tensor])[0]

        if channels == 4:
            tensor = tf.stack([tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2], a], 2)

    return tensor


@effect()
def glyph_map(tensor, shape, mask=None, colorize=True, zoom=1, alpha=1.0,
              spline_order=InterpolationType.constant, time=0.0, speed=1.0):
    """
    :param Tensor tensor:
    :param list[int] shape:
    :param ValueMask|None mask:
    """

    if mask is None:
        mask = ValueMask.truetype

    mask = value.coerce_enum(mask, ValueMask)

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
    value_shape = value.value_shape(shape)
    uv_noise = value.proportional_downsample(value.value_map(tensor, in_shape, keepdims=True), value_shape, uv_shape)

    approx_shape = [glyph_shape[0] * uv_shape[0], glyph_shape[1] * uv_shape[1], 1]

    uv_noise = value.resample(uv_noise, approx_shape, spline_order=spline_order)

    x_index = value.row_index(approx_shape) % glyph_shape[1]
    y_index = value.column_index(approx_shape) % glyph_shape[0]

    glyph_count = len(glyphs)
    z_index = tf.cast(uv_noise[:, :, 0] * glyph_count, tf.int32) % glyph_count

    spline_order = InterpolationType.cosine if mask == ValueMask.truetype else spline_order
    out = value.resample(tf.gather_nd(glyphs, tf.stack([z_index, y_index, x_index], 2)), [shape[0], shape[1], 1], spline_order=spline_order)

    if not colorize:
        return out * tf.ones(shape)

    out *= value.resample(value.proportional_downsample(tensor, shape, [uv_shape[0], uv_shape[1], channels]), shape, spline_order=spline_order)

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

    want_length = max(height, width) * 2

    padded_shape = [want_length, want_length, channels]

    padded = tf.image.resize_with_crop_or_pad(tensor, want_length, want_length)

    rotated = rotate2D(padded, padded_shape, math.radians(angle))

    # Find index of brightest pixel
    x_index = tf.expand_dims(tf.argmax(value.value_map(rotated, padded_shape), axis=1, output_type=tf.int32), -1)

    # Add offset index to row index
    x_index = (value.row_index(padded_shape) - tf.tile(x_index, [1, padded_shape[1]])) % padded_shape[1]

    # Sort pixels
    sorted_channels = [tf.nn.top_k(rotated[:, :, c], padded_shape[1])[0] for c in range(padded_shape[2])]

    # Apply offset
    sorted_channels = tf.gather_nd(tf.stack(sorted_channels, 2), tf.stack([value.column_index(padded_shape), x_index], 2))

    # Rotate back to original orientation
    sorted_channels = rotate2D(sorted_channels, padded_shape, math.radians(-angle))

    # Crop to original size
    sorted_channels = tf.image.resize_with_crop_or_pad(sorted_channels, height, width)

    # Blend with source image
    tensor = tf.maximum(tensor, sorted_channels)

    if darkest:
        tensor = 1.0 - tensor

    return tensor


@effect()
def rotate(tensor, shape, angle=None, time=0.0, speed=1.0):
    """Rotate the image. This breaks seamless edges."""

    height, width, channels = shape

    if angle is None:
        angle = random.random() * 360.0

    want_length = max(height, width) * 2

    padded_shape = [want_length, want_length, channels]

    padded = expand_tile(tensor, shape, padded_shape)

    rotated = rotate2D(padded, padded_shape, math.radians(angle))

    return tf.image.resize_with_crop_or_pad(rotated, height, width)


def rotate2D(tensor, shape, angle):
    """
    """

    x_index = tf.cast(value.row_index(shape), tf.float32) / shape[1] - 0.5
    y_index = tf.cast(value.column_index(shape), tf.float32) / shape[0] - 0.5

    _x_index = tf.cos(angle) * x_index + tf.sin(angle) * y_index + 0.5

    _y_index = -tf.sin(angle) * x_index + tf.cos(angle) * y_index + 0.5

    x_index = tf.cast(_x_index * shape[1], tf.int32) % shape[1]
    y_index = tf.cast(_y_index * shape[0], tf.int32) % shape[0]

    return tf.gather_nd(tensor, tf.stack([y_index, x_index], 2))


@effect()
def sketch(tensor, shape, time=0.0, speed=1.0):
    """
    Pencil sketch effect

    :param Tensor tensor:
    :param list[int] shape:
    :return Tensor:
    """

    value_shape = value.value_shape(shape)

    values = value.value_map(tensor, value_shape, keepdims=True)
    values = tf.image.adjust_contrast(values, 2.0)

    values = value.clamp01(values)

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
def simple_frame(tensor, shape, brightness=0.0, time=0.0, speed=1.0):
    """
    """

    border = value.singularity(None, shape, dist_metric=DistanceMetric.chebyshev)

    border = value.blend(tf.zeros(shape), border, .55)

    border = posterize(border, shape, 1)

    return value.blend(tensor, tf.ones(shape) * brightness, border)


@effect()
def lowpoly(tensor, shape, distrib=PointDistribution.random, freq=10, time=0.0, speed=1.0, dist_metric=DistanceMetric.euclidean):
    """Low-poly art style effect"""

    xy = point_cloud(freq, distrib=distrib, shape=shape, drift=1.0, time=time, speed=speed)

    distance = value.voronoi(tensor, shape, nth=1, xy=xy, dist_metric=dist_metric)
    color = value.voronoi(tensor, shape, diagram_type=VoronoiDiagramType.color_regions, xy=xy, dist_metric=dist_metric)

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
def kaleido(tensor, shape, sides=6, sdf_sides=5, xy=None, blend_edges=True, time=0.0, speed=1.0,
            point_freq=1, point_generations=1, point_distrib=PointDistribution.random, point_drift=0.0, point_corners=False):
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

    value_shape = value.value_shape(shape)

    if sdf_sides < 3:
        dist_metric = DistanceMetric.euclidean
    else:
        dist_metric = DistanceMetric.sdf

    # distance from any pixel to center
    r = value.voronoi(None, value_shape, dist_metric=dist_metric, sdf_sides=sdf_sides,
                      xy=xy, point_freq=point_freq, point_generations=point_generations,
                      point_distrib=point_distrib, point_drift=point_drift, point_corners=point_corners)

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
        fader = value.normalize(value.singularity(None, value_shape, dist_metric=DistanceMetric.chebyshev))
        fader = tf.squeeze(fader)  # conform to index shape
        fader = tf.math.pow(fader, 5)

        x_index = value.blend(x_index, x_identity, fader)
        y_index = value.blend(y_index, y_identity, fader)

    x_index = tf.cast(x_index, tf.int32)
    y_index = tf.cast(y_index, tf.int32)

    return tf.gather_nd(tensor, tf.stack([y_index % height, x_index % width], 2))


@effect()
def palette(tensor, shape, name=None, alpha=1.0, time=0.0, speed=1.0):
    """
    Another approach to image coloration
    https://iquilezles.org/www/articles/palettes/palettes.htm
    """

    if not name:
        return tensor

    # Can't apply if mode is grayscale
    if shape[2] in (1, 2):
        return tensor

    # Preserve the alpha channel
    alpha_channel = None
    if shape[2] == 4:
        alpha_channel = tensor[:, :, 3]
        tensor = tf.stack([tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]], 2)

    rgb_shape = [shape[0], shape[1], 3]

    p = palettes[name]

    offset = p["offset"] * tf.ones(rgb_shape)
    amp = p["amp"] * tf.ones(rgb_shape)
    freq = p["freq"] * tf.ones(rgb_shape)
    phase = p["phase"] * tf.ones(rgb_shape) + time

    # Multiply value_map's result x .875, in case the image is just black and white (0 == 1, we don't want a solid color image)
    colored = offset + amp * tf.math.cos(math.tau * (freq * value.value_map(
        tensor, shape, keepdims=True, with_normalize=False
    ) * .875 + .0625 + phase))

    tensor = value.blend_cosine(tensor, colored, alpha)

    # Re-insert the alpha channel
    if shape[2] == 4:
        tensor = tf.stack([tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2], alpha_channel], 2)

    return tensor


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

    base = value.simple_multires(2, shape, time=time, speed=speed, 
                                 octaves=random.randint(2, 5), spline_order=0)

    base = value.refract(base, shape, random.random())

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
    scan_noise = value.values(freq=[int(height * .5) + 1, int(width * .25) + 1], shape=[height, width, 1], time=time,
                              speed=speed * 100, spline_order=1)

    # Create horizontal offsets
    grad = value.values(freq=[int(random.random() * 10) + 5, 1], shape=[height, width, 1], time=time, speed=speed)
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

    value_shape = value.value_shape(shape)

    # Fake CRT lens shape
    mask = tf.pow(value.singularity(None, value_shape), 5)  # obscure center pinch

    # Displacement values multiplied by mask to make it wavy towards the edges
    distortion_x = (value.values(2, value_shape,
                                 time=time, speed=speed, spline_order=2) * 2.0 - 1.0) * mask

    return value.refract(tensor, shape, displacement, reference_x=distortion_x)


@effect()
def lens_distortion(tensor, shape, displacement=1.0, time=0.0, speed=1.0):
    """
    """

    x_index = tf.cast(value.row_index(shape), tf.float32) / shape[1]
    y_index = tf.cast(value.column_index(shape), tf.float32) / shape[0]

    x_dist = x_index - .5
    y_dist = y_index - .5

    center_dist = 1.0 - value.normalize(value.distance(x_dist, y_dist))

    if displacement < 0.0:
        zoom = displacement * -.25
    else:
        zoom = 0.0

    x_offset = tf.cast(((x_index - x_dist * zoom) - x_dist * center_dist * center_dist * displacement) * shape[1], tf.int32) % shape[1]
    y_offset = tf.cast(((y_index - y_dist * zoom) - y_dist * center_dist * center_dist * displacement) * shape[0], tf.int32) % shape[0]

    return tf.gather_nd(tensor, tf.stack([y_offset, x_offset], 2))


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
    Apply vintage CRT scanlines.

    :param Tensor tensor:
    :param list[int] shape:
    """

    height, width, channels = shape

    value_shape = value.value_shape(shape)

    # Horizontal scanlines
    scan_noise = tf.tile(value.normalize(value.values(freq=[2, 1], shape=[2, 1, 1], time=time, speed=speed * .1, spline_order=0)),
                         [int(height * .125) or 1, width, 1])

    scan_noise = value.resample(scan_noise, value_shape)

    scan_noise = lens_warp(scan_noise, value_shape, time=time, speed=speed)

    tensor = value.clamp01(value.blend(tensor, (tensor + scan_noise) * scan_noise, 0.05))

    if channels == 3:
        tensor = aberration(tensor, shape, .0125 + random.random() * .00625)
        tensor = tf.image.random_hue(tensor, .125)
        tensor = tf.image.adjust_saturation(tensor, 1.125)

    tensor = vignette(tensor, shape, brightness=0, alpha=random.random() * .175)
    tensor = tf.image.adjust_contrast(tensor, 1.25)

    return tensor


@effect()
def scanline_error(tensor, shape, time=0.0, speed=1.0):
    """
    """

    height, width, channels = shape

    value_shape = value.value_shape(shape)

    error_freq = [int(value_shape[0] * .5) or 1, int(value_shape[1] * .5) or 1]

    error_line = tf.maximum(value.values(freq=error_freq, shape=value_shape, time=time,
                                         speed=speed * 10, distrib=ValueDistribution.exp) - .5, 0)
    error_swerve = tf.maximum(value.values(freq=[int(height * .01), 1], shape=value_shape, time=time,
                                           speed=speed, distrib=ValueDistribution.exp) - .5, 0)

    error_line *= error_swerve

    error_swerve *= 2

    white_noise = value.values(freq=error_freq, shape=value_shape, time=time, speed=speed * 100)
    white_noise = value.blend(0, white_noise, error_swerve)

    error = error_line + white_noise

    y_index = value.column_index(shape)
    x_index = (value.row_index(shape) - tf.cast(value.value_map(error, value_shape) * width * .025, tf.int32)) % width

    return tf.minimum(tf.gather_nd(tensor, tf.stack([y_index, x_index], 2)) + error_line * white_noise * 4, 1)


@effect()
def snow(tensor, shape, alpha=0.25, time=0.0, speed=1.0):
    """
    """

    height, width, channels = shape

    value_shape = value.value_shape(shape)

    static = value.values(freq=[height, width], shape=value_shape, time=time, speed=speed * 100,
                          spline_order=0)

    static_limiter = value.values(freq=[height, width], shape=value_shape, time=time, speed=speed * 100,
                                  distrib=ValueDistribution.exp, spline_order=0) * alpha

    return value.blend(tensor, static, static_limiter)


@effect()
def grain(tensor, shape, alpha=0.25, time=0.0, speed=1.0):
    """
    """

    height, width, channels = shape

    white_noise = value.values(freq=[height, width], shape=[height, width, 1], time=time, speed=speed * 100)

    return value.blend(tensor, white_noise, alpha)


@effect()
def false_color(tensor, shape, horizontal=False, displacement=.5, time=0.0, speed=1.0):
    """
    """

    clut = value.values(freq=2, shape=shape, time=time, speed=speed)

    return value.normalize(color_map(tensor, shape, clut=clut, horizontal=horizontal, displacement=displacement))


@effect()
def fibers(tensor, shape, time=0.0, speed=1.0):
    """
    """

    value_shape = value.value_shape(shape)

    for i in range(4):
        mask = value.values(freq=4, shape=value_shape, time=time, speed=speed)

        mask = worms(mask, shape, behavior=WormBehavior.chaotic, alpha=1, density=.05 + random.random() * .00125,
                     duration=1, kink=random.randint(5, 10), stride=.75, stride_deviation=.125, time=time, speed=speed)

        brightness = value.values(freq=128, shape=shape, time=time, speed=speed)

        tensor = value.blend(tensor, brightness, mask * .5)

    return tensor


@effect()
def scratches(tensor, shape, time=0.0, speed=1.0):
    """
    """

    value_shape = value.value_shape(shape)

    for i in range(4):
        mask = value.values(freq=random.randint(2, 4), shape=value_shape, time=time, speed=speed)

        mask = worms(mask, value_shape, behavior=[1, 3][random.randint(0, 1)], alpha=1, density=.25 + random.random() * .25,
                     duration=2 + random.random() * 2, kink=.125 + random.random() * .125, stride=.75, stride_deviation=.5,
                     time=time, speed=speed)

        mask -= value.values(freq=random.randint(2, 4), shape=value_shape, time=time, speed=speed) * 2.0

        mask = tf.maximum(mask, 0.0)

        tensor = tf.maximum(tensor, mask * 8.0)

        tensor = tf.minimum(tensor, 1.0)

    return tensor


@effect()
def stray_hair(tensor, shape, time=0.0, speed=1.0):
    """
    """

    value_shape = value.value_shape(shape)

    mask = value.values(4, value_shape, time=time, speed=speed)

    mask = worms(mask, value_shape, behavior=WormBehavior.unruly, alpha=1, density=.0025 + random.random() * .00125,
                 duration=random.randint(8, 16), kink=random.randint(5, 50), stride=.5, stride_deviation=.25)

    brightness = value.values(freq=32, shape=value_shape, time=time, speed=speed)

    return value.blend(tensor, brightness * .333, mask * .666)


@effect()
def grime(tensor, shape, time=0.0, speed=1.0):
    """
    """

    value_shape = value.value_shape(shape)

    mask = value.simple_multires(freq=5, shape=value_shape, time=time, speed=speed,
                                 octaves=8)

    mask = value.refract(mask, value_shape, 1.0, y_from_offset=True)
    mask = derivative(mask, value_shape, DistanceMetric.chebyshev, alpha=0.125)

    dusty = value.blend(tensor, .25, tf.square(mask) * .075)

    specks = value.values(freq=[int(shape[0] * .25), int(shape[1] * .25)], shape=value_shape, time=time,
                          mask=ValueMask.dropout, speed=speed, distrib=ValueDistribution.exp)
    specks = value.refract(specks, value_shape, .25)

    specks = 1.0 - tf.sqrt(value.normalize(tf.maximum(specks - .625, 0.0)))

    dusty = value.blend(dusty, value.values(freq=[shape[0], shape[1]], shape=value_shape, mask=ValueMask.sparse,
                                            time=time, speed=speed, distrib=ValueDistribution.exp), .075) * specks

    return value.blend(tensor, dusty, mask * .75)


@effect()
def frame(tensor, shape, time=0.0, speed=1.0):
    """
    """

    half_shape = [int(shape[0] * .5), int(shape[1] * .5), shape[2]]
    half_value_shape = value.value_shape(half_shape)

    noise = value.simple_multires(64, half_value_shape, time=time, speed=speed, octaves=8)

    black = tf.zeros(half_value_shape)
    white = tf.ones(half_value_shape)

    mask = value.singularity(None, half_value_shape, VoronoiDiagramType.range, dist_metric=DistanceMetric.chebyshev, inverse=True)
    mask = value.normalize(mask + noise * .005)
    mask = blend_layers(tf.sqrt(mask), half_value_shape, 0.0125, white, black, black, black)

    faded = value.proportional_downsample(tensor, shape, half_shape)
    faded = tf.image.adjust_brightness(faded, .1)
    faded = tf.image.adjust_contrast(faded, .75)
    faded = light_leak(faded, half_shape, .125)
    faded = vignette(faded, half_shape, 0.05, .75)

    edge_texture = white * .9 + shadow(noise, half_value_shape, alpha=1.0) * .1

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

    value_shape = value.value_shape(shape)

    noise = value.simple_multires(64, value_shape, time=time, speed=speed,
                                  octaves=8, ridges=True)

    return tensor * (tf.ones(value_shape) * .9 + shadow(noise, value_shape, 1.0) * .1)


@effect()
def watermark(tensor, shape, time=0.0, speed=1.0):
    """
    """

    value_shape = value.value_shape(shape)

    mask = value.values(freq=240, shape=value_shape, spline_order=0, distrib=ValueDistribution.ones, mask="alphanum_numeric")

    mask = crt(mask, value_shape)

    mask = warp(mask, value_shape, [2, 4], octaves=1, displacement=.5, time=time, speed=speed)

    mask *= tf.square(value.values(freq=2, shape=value_shape, time=time, speed=speed))

    brightness = value.values(freq=16, shape=value_shape, time=time, speed=speed)

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
    value_shape = value.value_shape(shape)

    overlay = value.simple_multires([random.randint(3, 4), 1], value_shape, time=time, speed=speed,
                                    distrib=ValueDistribution.exp, ridges=True, octaves=6)

    overlay -= value.simple_multires([random.randint(2, 4), 1], value_shape, time=time, speed=speed,
                                     ridges=True, octaves=4)

    overlay *= .125

    overlay = rotate(overlay, value_shape, angle=random.randint(-15, 15), time=time, speed=speed)

    tensor *= 1.0 - overlay

    tensor += tint(tf.maximum(overlay * tf.ones(shape), 0), shape, alpha=1.0, time=time, speed=1.0)

    return tensor


@effect()
def spatter(tensor, shape, color=True, time=0.0, speed=1.0):
    """
    """

    value_shape = value.value_shape(shape)

    # Generate a smear
    smear = value.simple_multires(random.randint(3, 6), value_shape, time=time,
                                  speed=speed, distrib=ValueDistribution.exp,
                                  octaves=6, spline_order=3)

    smear = warp(smear, value_shape, [random.randint(2, 3), random.randint(1, 3)],
                 octaves=random.randint(1, 2), displacement=1.0 + random.random(),
                 spline_order=3, time=time, speed=speed)

    # Add spatter dots
    spatter = value.simple_multires(random.randint(32, 64), value_shape, time=time,
                                    speed=speed, distrib=ValueDistribution.exp,
                                    octaves=4, spline_order=InterpolationType.linear)

    spatter = adjust_brightness(spatter, shape, -1.0)
    spatter = adjust_contrast(spatter, shape, 4.0)

    smear = tf.maximum(smear, spatter)

    spatter = value.simple_multires(random.randint(150, 200), value_shape, time=time,
                                    speed=speed, distrib=ValueDistribution.exp,
                                    octaves=4, spline_order=InterpolationType.linear)

    spatter = adjust_brightness(spatter, shape, -1.25)
    spatter = adjust_contrast(spatter, shape, 4.0)

    smear = tf.maximum(smear, spatter)

    # Remove some of it
    smear = tf.maximum(0.0, smear - value.simple_multires(random.randint(2, 3), value_shape, time=time,
                                                          speed=speed, distrib=ValueDistribution.exp,
                                                          ridges=True, octaves=3, spline_order=2))

    #
    if color and shape[2] == 3:
        if color is True:
            splash = tf.image.random_hue(tf.ones(shape) * tf.stack([.875, 0.125, 0.125]), .5)

        else:  # Pass in [r, g, b]
            splash = tf.ones(shape) * tf.stack(color)

    else:
        splash = tf.zeros(shape)

    return blend_layers(value.normalize(smear), shape, .005, tensor, splash * tensor)


@effect()
def clouds(tensor, shape, time=0.0, speed=1.0):
    """Top-down cloud cover effect"""

    pre_shape = [int(shape[0] * .25) or 1, int(shape[1] * .25) or 1, 1]

    control = value.simple_multires(freq=random.randint(2, 4), shape=pre_shape,
                                    octaves=8, ridges=True, time=time, speed=speed)

    control = warp(control, pre_shape, freq=3, displacement=.125, octaves=2)

    layer_0 = tf.ones(pre_shape)
    layer_1 = tf.zeros(pre_shape)

    combined = blend_layers(control, pre_shape, 1.0, layer_0, layer_1)

    shaded = value.offset(combined, pre_shape, random.randint(-15, 15), random.randint(-15, 15))
    shaded = tf.minimum(shaded * 2.5, 1.0)

    for _ in range(3):
        shaded = value.convolve(kernel=ValueMask.conv2d_blur, tensor=shaded, shape=pre_shape)

    post_shape = [shape[0], shape[1], 1]

    shaded = value.resample(shaded, post_shape)
    combined = value.resample(combined, post_shape)

    tensor = value.blend(tensor, tf.zeros(shape), shaded * .75)
    tensor = value.blend(tensor, tf.ones(shape), combined)

    tensor = shadow(tensor, shape, alpha=.5)

    return tensor


@effect()
def tint(tensor, shape, time=0.0, speed=1.0, alpha=0.5):
    """
    """

    if shape[2] < 3:  # Not a color image
        return tensor

    color = value.values(freq=3, shape=shape, time=time, speed=speed, corners=True)

    # Confine hue to a range
    color = tf.stack([(tensor[:, :, 0] * .333 + random.random() * .333 + random.random()) % 1.0,
                      tensor[:, :, 1], tensor[:, :, 2]], 2)

    alpha_chan = None
    if shape[2] == 4:
        alpha_chan = tensor[:, :, 3]
        tensor = tf.stack([tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]], 2)

    colorized = tf.stack([color[:, :, 0], color[:, :, 1], tf.image.rgb_to_hsv([tensor])[0][:, :, 2]], 2)

    colorized = tf.image.hsv_to_rgb([colorized])[0]

    out = value.blend(tensor, colorized, alpha)

    if shape[2] == 4:
        out = tf.stack([out[:, :, 0], out[:, :, 1], out[:, :, 2], alpha_chan], 2)

    return out


@effect()
def adjust_hue(tensor, shape, amount=.25, time=0.0, speed=1.0):
    if amount not in (1.0, 0.0, None) and shape[2] == 3:
        tensor = tf.image.adjust_hue(tensor, amount)

    return tensor


@effect()
def adjust_saturation(tensor, shape, amount=.75, time=0.0, speed=1.0):
    if shape[2] == 3:
        tensor = tf.image.adjust_saturation(tensor, amount)

    return tensor


@effect()
def adjust_brightness(tensor, shape, amount=.125, time=0.0, speed=1.0):
    return tf.maximum(tf.minimum(tf.image.adjust_brightness(tensor, amount), 1.0), -1.0)


@effect()
def adjust_contrast(tensor, shape, amount=1.25, time=0.0, speed=1.0):
    return value.clamp01(tf.image.adjust_contrast(tensor, amount))


@effect()
def normalize(tensor, shape, time=0.0, speed=1.0):
    return value.normalize(tensor)


@effect()
def ridge(tensor, shape, time=0.0, speed=1.0):
    return value.ridge(tensor)


@effect()
def sine(tensor, shape, amount=1.0, time=0.0, speed=1.0, rgb=False):
    channels = shape[2]

    if channels == 1:
        return value.normalized_sine(tensor * amount)

    elif channels == 2:
        return tf.stack([value.normalized_sine(tensor[:, :, 0] * amount), tensor[:, :, 1]], 2)

    elif channels == 3:
        if rgb:
            return value.normalized_sine(tensor * amount)

        return tf.stack([tensor[:, :, 0], tensor[:, :, 1], value.normalized_sine(tensor[:, :, 2] * amount)], 2)

    elif channels == 4:
        if rgb:
            temp = value.normalized_sine(tf.stack([tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]], 2) * amount)

            return tf.stack([temp[:, :, 0], temp[:, :, 1], temp[:, :, 2], tensor[:, :, 3]], 2)

        return tf.stack([tensor[:, :, 0], tensor[:, :, 1], value.normalized_sine(tensor[:, :, 2] * amount), tensor[:, :, 3]], 2)


@effect()
def value_refract(tensor, shape, freq=4, distrib=ValueDistribution.center_circle, displacement=.125, time=0.0, speed=1.0):
    """
    """

    blend_values = value.values(freq=freq, shape=value.value_shape(shape), distrib=distrib, time=time, speed=speed)

    return value.refract(tensor, shape, time=time, speed=speed, reference_x=blend_values, displacement=displacement)


@effect()
def blur(tensor, shape, amount=10.0, spline_order=InterpolationType.bicubic, time=0.0, speed=1.0):
    ""
    ""

    tensor = value.proportional_downsample(tensor, shape, [max(int(shape[0] / amount), 1), max(int(shape[1] / amount), 1), shape[2]]) * 4.0

    tensor = value.resample(tensor, shape, spline_order=spline_order)

    return tensor
