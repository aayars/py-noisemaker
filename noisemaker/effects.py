"""Low-level effects library for Noisemaker"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import tensorflow as tf

import noisemaker.masks as masks
import noisemaker.rng as rng
import noisemaker.simplex as simplex
import noisemaker.util as util
import noisemaker.value as value
from noisemaker.constants import (
    DistanceMetric,
    InterpolationType,
    PointDistribution,
    ValueDistribution,
    ValueMask,
    VoronoiDiagramType,
    WormBehavior,
)
from noisemaker.effects_registry import effect
from noisemaker.glyphs import load_glyphs
from noisemaker.palettes import PALETTES as palettes
from noisemaker.points import point_cloud


def _conform_kernel_to_tensor(kernel: ValueMask, tensor: tf.Tensor, shape: list[int]) -> tf.Tensor:
    """
    Re-shape a convolution kernel to match the given tensor's color dimensions.

    Args:
        kernel: Convolution kernel mask
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]

    Returns:
        Processed tensor
    """

    values, _ = masks.mask_values(kernel)

    length = len(values)

    channels = shape[-1]

    temp = np.repeat(values, channels)

    temp = tf.reshape(temp, (length, length, channels, 1))

    temp = tf.cast(temp, tf.float32)

    temp /= tf.maximum(tf.reduce_max(temp), tf.reduce_min(temp) * -1)

    return temp


@effect()
def erosion_worms(
    tensor: tf.Tensor,
    shape: list[int],
    density: float = 50,
    iterations: int = 50,
    contraction: float = 1.0,
    quantize: bool = False,
    alpha: float = 0.25,
    inverse: bool = False,
    xy_blend: bool = False,
    time: float = 0.0,
    speed: float = 1.0,
) -> tf.Tensor:
    """
    WIP hydraulic erosion effect.

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        density: Feature density
        iterations: Number of iterations to perform
        contraction: Contraction amount
        quantize: Quantize output colors
        alpha: Blending alpha value (0.0-1.0)
        inverse: Invert the effect
        xy_blend: XY blend amount
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    # This will never be as good as
    # https://www.dropbox.com/s/kqv8b3w7o8ucbyi/Beyer%20-%20implementation%20of%20a%20methode%20for%20hydraulic%20erosion.pdf?dl=0

    height, width, channels = shape

    count = int(math.sqrt(height * width) * density)

    x = rng.uniform([count]) * (width - 1)
    y = rng.uniform([count]) * (height - 1)

    x_dir = rng.normal([count])
    y_dir = rng.normal([count])

    length = tf.sqrt(x_dir * x_dir + y_dir * y_dir)
    x_dir /= length
    y_dir /= length

    inertia = rng.normal([count], mean=0.75, stddev=0.25)

    out = tf.zeros(shape, dtype=tf.float32)

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
def reindex(tensor: tf.Tensor, shape: list[int], displacement: float = 0.5, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Re-color the given tensor, by sampling along one axis at a specified frequency.

    .. image:: images/reindex.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        displacement: Displacement amount
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Processed tensor
    """

    height, width, channels = shape

    reference = value.value_map(tensor, shape)

    mod = min(height, width)
    x_offset = tf.cast((reference * displacement * mod + reference) % width, tf.int32)
    y_offset = tf.cast((reference * displacement * mod + reference) % height, tf.int32)

    tensor = tf.gather_nd(tensor, tf.stack([y_offset, x_offset], 2))

    return tensor


@effect()
def ripple(
    tensor: tf.Tensor,
    shape: list[int],
    freq: int | list[int] = 2,
    displacement: float = 1.0,
    kink: float = 1.0,
    reference: tf.Tensor | None = None,
    spline_order: int | InterpolationType = InterpolationType.bicubic,
    time: float = 0.0,
    speed: float = 1.0,
) -> tf.Tensor:
    """
    Apply displacement from pixel radian values.

    .. image:: images/ripple.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        freq: Noise frequency
        displacement: Displacement amount
        kink: Displacement kink amount
        reference: Reference tensor for comparison
        spline_order: Interpolation type for resampling
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
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
def color_map(
    tensor: tf.Tensor,
    shape: list[int],
    clut: tf.Tensor | str | None = None,
    horizontal: bool = False,
    displacement: float = 0.5,
    time: float = 0.0,
    speed: float = 1.0,
) -> tf.Tensor:
    """
    Apply a color map to an image tensor.
    The color map can be a photo or whatever else.

    .. image:: images/color_map.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        clut: Color lookup table
        horizontal: Apply horizontally (vs vertically)
        displacement: Displacement amount
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
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
def worms(
    tensor: tf.Tensor,
    shape: list[int],
    behavior: int | WormBehavior = 1,
    density: float = 4.0,
    duration: float = 4.0,
    stride: float = 1.0,
    stride_deviation: float = 0.05,
    alpha: float = 0.5,
    kink: float = 1.0,
    drunkenness: float = 0.0,
    quantize: bool = False,
    colors: tf.Tensor | None = None,
    time: float = 0.0,
    speed: float = 1.0,
) -> tf.Tensor:
    """
    Make a furry patch of worms which follow field flow rules.

    .. image:: images/worms.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        behavior: Worm behavior mode
        density: Feature density
        duration: Effect duration
        stride: Movement stride length
        stride_deviation: Stride randomization amount
        alpha: Blending alpha value (0.0-1.0)
        kink: Displacement kink amount
        drunkenness: Random walk amount
        quantize: Quantize output colors
        colors: Optional color tensor for rendering
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    behavior = value.coerce_enum(behavior, WormBehavior)

    height, width, channels = shape

    count = int(max(width, height) * density)

    worms_y = rng.uniform([count]) * (height - 1)  # RNG[1]
    worms_x = rng.uniform([count]) * (width - 1)  # RNG[2]
    worms_stride = rng.normal([count], mean=stride, stddev=stride_deviation) * (max(width, height) / 1024.0)  # RNG[3]

    color_source = colors if colors is not None else tensor

    colors = tf.gather_nd(color_source, tf.cast(tf.stack([worms_y, worms_x], 1), tf.int32))

    quarter_count = int(count * 0.25)

    rots: dict[WormBehavior, Any] = {}

    rots = {
        WormBehavior.obedient: lambda n: tf.ones([n], dtype=tf.float32) * rng.random() * math.tau,  # RNG[4]
        WormBehavior.crosshatch: lambda n: rots[WormBehavior.obedient](n) + (tf.floor(rng.uniform([n]) * 100) % 4) * math.radians(90),  # RNG[5]
        WormBehavior.unruly: lambda n: rots[WormBehavior.obedient](n) + rng.uniform([n]) * 0.25 - 0.125,  # RNG[6]
        WormBehavior.chaotic: lambda n: rng.uniform([n]) * math.tau,  # RNG[7]
        WormBehavior.random: lambda _: tf.reshape(
            tf.stack(
                [
                    rots[WormBehavior.obedient](quarter_count),
                    rots[WormBehavior.crosshatch](quarter_count),
                    rots[WormBehavior.unruly](quarter_count),
                    rots[WormBehavior.chaotic](quarter_count),
                ]
            ),
            [count],
        ),
        # Chaotic, changing over time
        WormBehavior.meandering: lambda n: value.periodic_value(time * speed, rng.uniform([count])),  # RNG[8]
    }

    # Ensure behavior is WormBehavior enum
    if isinstance(behavior, int):
        behavior = WormBehavior(behavior)

    worms_rot = rots[behavior](count)

    index = value.value_map(tensor, shape) * math.tau * kink

    iterations = int(math.sqrt(min(width, height)) * duration)

    out = tf.zeros(shape, dtype=tf.float32)

    scatter_shape = tf.shape(tensor)  # Might be different than `shape` due to clut

    # Make worms!
    for i in range(iterations):
        if drunkenness:
            start = int(min(shape[0], shape[1]) * time * speed + i * speed * 10)

            worms_rot += (value.periodic_value(start, rng.uniform([count])) * 2.0 - 1.0) * drunkenness * math.pi

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
def wormhole(
    tensor: tf.Tensor, shape: list[int], kink: float = 1.0, input_stride: float = 1.0, alpha: float = 1.0, time: float = 0.0, speed: float = 1.0
) -> tf.Tensor:
    """
    Apply per-pixel field flow. Non-iterative.

    .. image:: images/wormhole.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        kink: Displacement kink amount
        input_stride: Input sampling stride
        alpha: Blending alpha value (0.0-1.0)
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
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
def derivative(
    tensor: tf.Tensor,
    shape: list[int],
    dist_metric: int | DistanceMetric = DistanceMetric.euclidean,
    with_normalize: bool = True,
    alpha: float = 1.0,
    time: float = 0.0,
    speed: float = 1.0,
) -> tf.Tensor:
    """
    Extract a derivative from the given noise.

    .. image:: images/derived.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        dist_metric: Distance metric to use
        with_normalize: Normalize the output
        alpha: Blending alpha value (0.0-1.0)
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
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
def sobel_operator(
    tensor: tf.Tensor, shape: list[int], dist_metric: int | DistanceMetric = DistanceMetric.euclidean, time: float = 0.0, speed: float = 1.0
) -> tf.Tensor:
    """
    Apply a sobel operator.

    .. image:: images/sobel.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        dist_metric: Distance metric to use
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    tensor = value.convolve(kernel=ValueMask.conv2d_blur, tensor=tensor, shape=shape)

    x = value.convolve(kernel=ValueMask.conv2d_sobel_x, tensor=tensor, shape=shape, with_normalize=False)
    y = value.convolve(kernel=ValueMask.conv2d_sobel_y, tensor=tensor, shape=shape, with_normalize=False)

    out = tf.abs(value.normalize(value.distance(x, y, dist_metric)) * 2 - 1)

    fudge = -1

    out = value.offset(out, shape, x=fudge, y=fudge)

    return out


@effect()
def normal_map(tensor: tf.Tensor, shape: list[int], time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Generate a tangent-space normal map.

    .. image:: images/normals.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    reference = value.value_map(tensor, shape, keepdims=True)

    value_shape = value.value_shape(shape)

    x = value.normalize(1 - value.convolve(kernel=ValueMask.conv2d_sobel_x, tensor=reference, shape=value_shape))
    y = value.normalize(value.convolve(kernel=ValueMask.conv2d_sobel_y, tensor=reference, shape=value_shape))

    z = 1 - tf.abs(value.normalize(tf.sqrt(x * x + y * y)) * 2 - 1) * 0.5 + 0.5

    return tf.stack([x[:, :, 0], y[:, :, 0], z[:, :, 0]], 2)


@effect()
def density_map(tensor: tf.Tensor, shape: list[int], time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Create a binned pixel value density map.

    .. image:: images/density.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
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

    return tf.ones(shape, dtype=tf.float32) * value.normalize(tf.cast(out, tf.float32))


@effect()
def jpeg_decimate(tensor: tf.Tensor, shape: list[int], iterations: int = 25, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Destroy an image with the power of JPEG

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        iterations: Number of iterations to perform
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    jpegged = tensor

    for i in range(iterations):
        jpegged = tf.image.convert_image_dtype(jpegged, tf.uint8)

        data = tf.image.encode_jpeg(jpegged, quality=rng.random_int(5, 50), x_density=rng.random_int(50, 500), y_density=rng.random_int(50, 500))
        jpegged = tf.image.decode_jpeg(data)

        jpegged = tf.image.convert_image_dtype(jpegged, tf.float32, saturate=True)

    return jpegged


@effect()
def conv_feedback(tensor: tf.Tensor, shape: list[int], iterations: int = 50, alpha: float = 0.5, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Conv2d feedback loop

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        iterations: Number of iterations to perform
        alpha: Blending alpha value (0.0-1.0)
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    iterations = 100

    half_shape = [int(shape[0] * 0.5), int(shape[1] * 0.5), shape[2]]

    convolved = value.proportional_downsample(tensor, shape, half_shape)

    for i in range(iterations):
        convolved = value.convolve(kernel=ValueMask.conv2d_blur, tensor=convolved, shape=half_shape)
        convolved = value.convolve(kernel=ValueMask.conv2d_sharpen, tensor=convolved, shape=half_shape)

    convolved = value.normalize(convolved)

    up = tf.maximum((convolved - 0.5) * 2, 0.0)

    down = tf.minimum(convolved * 2, 1.0)

    return value.blend(tensor, value.resample(up + (1.0 - down), shape), alpha)


def blend_layers(control: list[tuple[tf.Tensor, float]], shape: list[int], feather: float = 1.0, *layers: Any) -> tf.Tensor:
    """
    Blend multiple image layers based on a control tensor.

    Args:
        control: Control tensor for blending
        shape: Shape of the tensor [height, width, channels]
        feather: Feathering amount for blending transitions
        *layers: Variable number of layer tensors to blend

    Returns:
        Modified tensor
    """
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


def center_mask(
    center: tf.Tensor, edges: tf.Tensor, shape: list[int], dist_metric: int | DistanceMetric = DistanceMetric.chebyshev, power: float = 2
) -> tf.Tensor:
    """
    Blend two image tensors from the center to the edges.

    Args:
        center: Center point coordinates
        edges: Edge handling mode
        shape: Shape of the tensor [height, width, channels]
        dist_metric: Distance metric to use
        power: Power curve exponent

    Returns:
        Processed tensor
    """

    mask = tf.pow(value.singularity(None, shape, dist_metric=dist_metric), power)

    return value.blend(center, edges, mask)


@effect()
def posterize(tensor: tf.Tensor, shape: list[int], levels: int = 9, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Reduce the number of color levels per channel.

    .. image:: images/posterize.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        levels: Number of posterization levels
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    if levels == 0:
        return tensor

    if shape[-1] == 3:
        tensor = util.from_srgb(tensor)

    tensor *= levels

    tensor += (1 / levels) * 0.5

    tensor = tf.floor(tensor)

    tensor /= levels

    if shape[-1] == 3:
        tensor = util.from_linear_rgb(tensor)

    return tensor


def inner_tile(tensor: tf.Tensor, shape: list[int], freq: int | list[int]) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        freq: Noise frequency

    Returns:
        Modified tensor
    """

    if isinstance(freq, int):
        freq = value.freq_for_shape(freq, shape)

    # At this point freq is definitely list[int]
    assert isinstance(freq, list)
    freq_list: list[int] = freq

    small_shape = [int(shape[0] / freq_list[0]), int(shape[1] / freq_list[1]), shape[2]]

    y_index = tf.tile(value.column_index(small_shape) * freq_list[0], [freq_list[0], freq_list[0]])
    x_index = tf.tile(value.row_index(small_shape) * freq_list[1], [freq_list[0], freq_list[0]])

    tiled = tf.gather_nd(tensor, tf.stack([y_index, x_index], 2))

    tiled = value.resample(tiled, shape, spline_order=InterpolationType.linear)

    return tiled


def expand_tile(tensor: tf.Tensor, input_shape: list[int], output_shape: list[int], with_offset: bool = True) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        input_shape: Shape of input tensor
        output_shape: Shape of output tensor
        with_offset: Apply offset to output

    Returns:
        Modified tensor
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


def offset_index(y_index: tf.Tensor, height: int, x_index: tf.Tensor, width: int) -> tf.Tensor:
    """
    Offset X and Y displacement channels from each other, to help with diagonal banding.
    Returns a combined Tensor with shape [height, width, 2]

    Args:
        y_index: Y coordinate indices
        height: Height dimension
        x_index: X coordinate indices
        width: Width dimension

    Returns:
        Processed tensor
    """

    index = tf.stack(
        [
            (y_index + int(height * 0.5 + rng.random() * height * 0.5)) % height,
            (x_index + int(rng.random() * width * 0.5)) % width,
        ],
        2,
    )

    return tf.cast(index, tf.int32)


@effect()
def warp(
    tensor: tf.Tensor,
    shape: list[int],
    freq: int | list[int] = 2,
    octaves: int = 5,
    displacement: float = 1,
    spline_order: int | InterpolationType = InterpolationType.bicubic,
    warp_map: tf.Tensor | None = None,
    signed_range: bool = True,
    time: float = 0.0,
    speed: float = 1.0,
) -> tf.Tensor:
    """
    Multi-octave warp effect

    .. image:: images/warp.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        freq: Noise frequency
        octaves: Number of octave layers
        displacement: Displacement amount
        spline_order: Interpolation type for resampling
        warp_map: Optional warp displacement map
        signed_range: Use signed range (-1 to 1)
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    if isinstance(freq, int):
        freq = value.freq_for_shape(freq, shape)

    for octave in range(1, octaves + 1):
        multiplier = 2**octave

        freq_list = freq if isinstance(freq, list) else [freq, freq]
        base_freq = [int(f * 0.5 * multiplier) for f in freq_list]

        if base_freq[0] >= shape[0] or base_freq[1] >= shape[1]:
            break

        kwargs = {}

        if warp_map is not None:
            if isinstance(warp_map, str):
                warp_map = tf.image.convert_image_dtype(util.load(warp_map), tf.float32)

            kwargs["reference_x"] = warp_map
        else:
            kwargs["warp_freq"] = base_freq

        tensor = value.refract(
            tensor, shape, displacement=displacement / multiplier, spline_order=spline_order, signed_range=signed_range, time=time, speed=speed, **kwargs
        )

    return tensor


def sobel(tensor: tf.Tensor, shape: list[int], dist_metric: int | DistanceMetric = 1, rgb: bool = False) -> tf.Tensor:
    """
    Colorized sobel edges.

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        dist_metric: Distance metric to use
        rgb: Treat as RGB (vs grayscale)

    Returns:
        Modified tensor
    """

    if rgb:
        return sobel_operator(tensor, shape, dist_metric)

    else:
        return outline(tensor, shape, dist_metric, True)


@effect()
def outline(
    tensor: tf.Tensor, shape: list[int], sobel_metric: int | DistanceMetric = 1, invert: bool = False, time: float = 0.0, speed: float = 1.0
) -> tf.Tensor:
    """
    Superimpose sobel operator results (cartoon edges)

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        sobel_metric: Distance metric for Sobel operator
        invert: Invert the effect
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    height, width, channels = shape

    value_shape = value.value_shape(shape)

    values = value.value_map(tensor, shape, keepdims=True)

    edges = sobel_operator(values, value_shape, dist_metric=sobel_metric)

    if invert:
        edges = 1.0 - edges

    return edges * tensor


@effect()
def glowing_edges(
    tensor: tf.Tensor, shape: list[int], sobel_metric: int | DistanceMetric = 2, alpha: float = 1.0, time: float = 0.0, speed: float = 1.0
) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        sobel_metric: Distance metric for Sobel operator
        alpha: Blending alpha value (0.0-1.0)
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    height, width, channels = shape

    value_shape = value.value_shape(shape)

    edges = value.value_map(tensor, shape, keepdims=True)

    edges = posterize(edges, value_shape, rng.random_int(3, 5))

    edges = 1.0 - sobel_operator(edges, value_shape, dist_metric=sobel_metric)

    edges = tf.minimum(edges * 8, 1.0) * tf.minimum(tensor * 1.25, 1.0)

    edges = bloom(edges, shape, alpha=0.5)

    edges = value.normalize(edges + value.convolve(kernel=ValueMask.conv2d_blur, tensor=edges, shape=shape))

    return value.blend(tensor, 1.0 - ((1.0 - edges) * (1.0 - tensor)), alpha)


@effect()
def vortex(tensor: tf.Tensor, shape: list[int], displacement: float = 64.0, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Vortex tiling effect

    .. image:: images/vortex.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        displacement: Displacement amount
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
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

    warped = value.refract(tensor, shape, displacement=simplex.random(time, speed=speed) * 100 * displacement, reference_x=x, reference_y=y, signed_range=False)

    return warped


@effect()
def aberration(tensor: tf.Tensor, shape: list[int], displacement: float = 0.005, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Chromatic aberration

    .. image:: images/aberration.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        displacement: Displacement amount
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
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

    shift = rng.random() * 0.1 - 0.05
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
def bloom(tensor: tf.Tensor, shape: list[int], alpha: float = 0.5, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Bloom effect
    Input image must currently be square (sorry).

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        alpha: Blending alpha value (0.0-1.0)
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    height, width, channels = shape

    blurred = value.clamp01(tensor * 2.0 - 1.0)
    blurred = value.proportional_downsample(blurred, shape, [max(int(height / 100), 1), max(int(width / 100), 1), channels]) * 4.0
    blurred = value.resample(blurred, shape)

    blurred = value.offset(blurred, shape, x=int(tf.cast(width, tf.float32) * -0.05), y=int(tf.cast(shape[0], tf.float32) * -0.05))

    # Mirror the JavaScript bloom implementation exactly: brightness is a straight
    # addition followed by clamping to ``[-1, 1]`` before the contrast stretch is
    # applied.  Using the TensorFlow helpers here introduces small numerical
    # differences, so we implement the arithmetic directly to stay bit-for-bit in
    # sync with the reference.
    blurred = tf.clip_by_value(blurred + 0.25, -1.0, 1.0)

    mean = tf.reduce_mean(blurred, axis=[0, 1], keepdims=True)
    blurred = (blurred - mean) * 1.5 + mean
    blurred = value.clamp01(blurred)

    return value.blend(value.clamp01(tensor), value.clamp01((tensor + blurred) * 0.5), alpha)


@effect()
def dla(
    tensor: tf.Tensor,
    shape: list[int],
    padding: int = 2,
    seed_density: float = 0.01,
    density: float = 0.125,
    xy: tuple[tf.Tensor, tf.Tensor, int] | None = None,
    alpha: float = 1.0,
    time: float = 0.0,
    speed: float = 1.0,
) -> tf.Tensor:
    """
    Diffusion-limited aggregation. Renders with respect to the `time` param (0..1)

    .. image:: images/dla.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        padding: Edge padding amount
        seed_density: Density of seed points
        density: Feature density
        xy: Optional XY coordinates for point cloud
        alpha: Blending alpha value (0.0-1.0)
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
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
        result = point_cloud(seed_count, distrib=PointDistribution.random, shape=shape, time=time, speed=speed)
        if result is None:
            raise ValueError("point_cloud returned None")
        x, y = result

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
            walkers.append((int(rng.random() * half_height), int(rng.random() * half_width)))

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
                walkers[w] = (
                    (walker[0] + offsets[rng.random_int(0, len(offsets) - 1)]) % half_height,
                    (walker[1] + offsets[rng.random_int(0, len(offsets) - 1)]) % half_width,
                )

            else:
                walkers[w] = (
                    (walker[0] + expanded_offsets[rng.random_int(0, len(expanded_offsets) - 1)]) % half_height,
                    (walker[1] + expanded_offsets[rng.random_int(0, len(expanded_offsets) - 1)]) % half_width,
                )

    seen = set()
    unique = []

    for c in clustered:
        if c in seen:
            continue

        seen.add(c)

        unique.append(c)

    count = len(unique)

    # hot = tf.ones([count, channels])
    hot = tf.ones([count, channels], dtype=tf.float32) * tf.cast(tf.reshape(tf.stack(list(reversed(range(count)))), [count, 1]), tf.float32)

    out = value.convolve(kernel=ValueMask.conv2d_blur, tensor=tf.scatter_nd(tf.stack(unique) * int(1 / scale), hot, [height, width, channels]), shape=shape)

    return value.blend(tensor, out * tensor, alpha)


@effect()
def wobble(tensor: tf.Tensor, shape: list[int], time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Move the entire image around

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    x_offset = tf.cast(simplex.random(time=time, speed=speed * 0.5) * shape[1], tf.int32)
    y_offset = tf.cast(simplex.random(time=time, speed=speed * 0.5) * shape[0], tf.int32)

    return value.offset(tensor, shape, x=x_offset, y=y_offset)


@effect()
def reverb(tensor: tf.Tensor, shape: list[int], octaves: int = 2, iterations: int = 1, ridges: bool = True, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Multi-octave "reverberation" of input image tensor

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        octaves: Number of octave layers
        iterations: Number of iterations to perform
        ridges: Apply ridge transformation
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
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
            multiplier = 2**octave

            octave_shape = [int(height / multiplier) or 1, int(width / multiplier) or 1, channels]

            if not all(octave_shape):
                break

            layer = value.proportional_downsample(reference, shape, octave_shape)

            out += expand_tile(layer, octave_shape, shape) / multiplier

    return value.normalize(out)


@effect()
def light_leak(tensor: tf.Tensor, shape: list[int], alpha: float = 0.25, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        alpha: Blending alpha value (0.0-1.0)
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    result = point_cloud(
        6,
        distrib=PointDistribution.grid_members()[rng.random_int(0, len(PointDistribution.grid_members()) - 1)],
        drift=0.05,
        shape=shape,
        time=time,
        speed=speed,
    )
    if result is None:
        raise ValueError("point_cloud returned None")
    x, y = result

    leak = value.voronoi(tensor, shape, diagram_type=VoronoiDiagramType.color_regions, xy=(x, y, len(x)))
    leak = wormhole(leak, shape, kink=1.0, input_stride=0.25)

    leak = bloom(leak, shape, 1.0)

    leak = 1 - ((1 - tensor) * (1 - leak))

    leak = center_mask(tensor, leak, shape, 4)

    return vaseline(value.blend(tensor, leak, alpha), shape, alpha)


@effect()
def vignette(tensor: tf.Tensor, shape: list[int], brightness: float = 0.0, alpha: float = 1.0, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        brightness: Brightness adjustment
        alpha: Blending alpha value (0.0-1.0)
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    tensor = value.normalize(tensor)

    edges = center_mask(tensor, tf.ones(shape, dtype=tf.float32) * brightness, shape, dist_metric=DistanceMetric.euclidean)

    return value.blend(tensor, edges, alpha)


@effect()
def vaseline(tensor: tf.Tensor, shape: list[int], alpha: float = 1.0, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        alpha: Blending alpha value (0.0-1.0)
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    return value.blend(tensor, center_mask(tensor, bloom(tensor, shape, 1.0), shape), alpha)


@effect()
def shadow(tensor: tf.Tensor, shape: list[int], alpha: float = 1.0, reference: tf.Tensor | None = None, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Convolution-based self-shadowing effect.

    .. image:: images/shadow.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        alpha: Blending alpha value (0.0-1.0)
        reference: Reference tensor for comparison
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    height, width, channels = shape

    if reference is None:
        reference = tensor

    reference = value.value_map(reference, shape, keepdims=True)

    value_shape = value.value_shape(shape)

    x = value.convolve(kernel=ValueMask.conv2d_sobel_x, tensor=reference, shape=value_shape)
    y = value.convolve(kernel=ValueMask.conv2d_sobel_y, tensor=reference, shape=value_shape)

    shade = value.normalize(value.distance(x, y, DistanceMetric.euclidean))

    shade = value.convolve(kernel=ValueMask.conv2d_sharpen, tensor=shade, shape=value_shape, alpha=0.5)

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

        tensor = tf.stack([tensor[:, :, 0], tensor[:, :, 1], value.blend(tensor[:, :, 2], tf.image.rgb_to_hsv([shade])[0][:, :, 2], alpha)], 2)

        tensor = tf.image.hsv_to_rgb([tensor])[0]

        if channels == 4:
            tensor = tf.stack([tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2], a], 2)

    return tensor


@effect()
def glyph_map(
    tensor: tf.Tensor,
    shape: list[int],
    mask: ValueMask | None = None,
    colorize: bool = True,
    zoom: int = 1,
    alpha: float = 1.0,
    spline_order: InterpolationType = InterpolationType.constant,
    time: float = 0.0,
    speed: float = 1.0,
) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        mask: Value mask to apply
        colorize: Apply colorization
        zoom: Zoom factor
        alpha: Blending alpha value (0.0-1.0)
        spline_order: Interpolation type for resampling
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
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
        return out * tf.ones(shape, dtype=tf.float32)

    out *= value.resample(value.proportional_downsample(tensor, shape, [uv_shape[0], uv_shape[1], channels]), shape, spline_order=spline_order)

    if alpha == 1.0:
        return out

    return value.blend(tensor, out, alpha)


@effect()
def pixel_sort(tensor: tf.Tensor, shape: list[int], angled: bool | float = False, darkest: bool = False, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Pixel sort effect

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        angled: Use angled sorting
        darkest: Sort from darkest pixels
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    if angled:
        angle = rng.random() * 360.0 if isinstance(angled, bool) else angled

    else:
        angle = False

    tensor = _pixel_sort(tensor, shape, angle, darkest)

    return tensor


def _pixel_sort(tensor: tf.Tensor, shape: list[int], angle: float | None, darkest: bool) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        angle: Rotation angle in radians
        darkest: Sort from darkest pixels

    Returns:
        Processed tensor
    """
    height, width, channels = shape

    if darkest:
        tensor = 1.0 - tensor

    # Handle None angle
    if angle is None:
        angle = 0.0

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
def rotate(tensor: tf.Tensor, shape: list[int], angle: float | None = None, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Rotate the image. This breaks seamless edges.

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        angle: Rotation angle in radians
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    height, width, channels = shape

    if angle is None:
        angle = rng.random() * 360.0

    want_length = max(height, width) * 2

    padded_shape = [want_length, want_length, channels]

    padded = expand_tile(tensor, shape, padded_shape)

    rotated = rotate2D(padded, padded_shape, math.radians(angle))

    return tf.image.resize_with_crop_or_pad(rotated, height, width)


def rotate2D(tensor: tf.Tensor, shape: list[int], angle: float | None) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        angle: Rotation angle in radians

    Returns:
        Modified tensor
    """

    x_index = tf.cast(value.row_index(shape), tf.float32) / shape[1] - 0.5
    y_index = tf.cast(value.column_index(shape), tf.float32) / shape[0] - 0.5

    _x_index = tf.cos(angle) * x_index + tf.sin(angle) * y_index + 0.5

    _y_index = -tf.sin(angle) * x_index + tf.cos(angle) * y_index + 0.5

    x_index = tf.cast(_x_index * shape[1], tf.int32) % shape[1]
    y_index = tf.cast(_y_index * shape[0], tf.int32) % shape[0]

    return tf.gather_nd(tensor, tf.stack([y_index, x_index], 2))


@effect()
def sketch(tensor: tf.Tensor, shape: list[int], time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Pencil sketch effect

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    value_shape = value.value_shape(shape)

    values = value.value_map(tensor, value_shape, keepdims=True)
    values = tf.image.adjust_contrast(values, 2.0)

    values = value.clamp01(values)

    outline = 1.0 - derivative(values, value_shape)
    outline = tf.minimum(outline, 1.0 - derivative(1.0 - values, value_shape))
    outline = tf.image.adjust_contrast(outline, 0.25)
    outline = value.normalize(outline)

    values = vignette(values, value_shape, 1.0, 0.875)

    crosshatch = 1.0 - worms(1.0 - values, value_shape, behavior=2, density=125, duration=0.5, stride=1, stride_deviation=0.25, alpha=1.0)
    crosshatch = value.normalize(crosshatch)

    combined = value.blend(crosshatch, outline, 0.75)
    combined = warp(combined, value_shape, [int(shape[0] * 0.125) or 1, int(shape[1] * 0.125) or 1], octaves=1, displacement=0.0025, time=time, speed=speed)
    combined *= combined

    return combined * tf.ones(shape, dtype=tf.float32)


@effect()
def simple_frame(tensor: tf.Tensor, shape: list[int], brightness: float = 0.0, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        brightness: Brightness adjustment
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    border = value.singularity(None, shape, dist_metric=DistanceMetric.chebyshev)

    border = value.blend(tf.zeros(shape, dtype=tf.float32), border, 0.55)

    border = posterize(border, shape, 1)

    return value.blend(tensor, tf.ones(shape, dtype=tf.float32) * brightness, border)


@effect()
def lowpoly(
    tensor: tf.Tensor,
    shape: list[int],
    distrib: int | PointDistribution | ValueDistribution = PointDistribution.random,
    freq: int | list[int] = 10,
    time: float = 0.0,
    speed: float = 1.0,
    dist_metric: int | DistanceMetric = DistanceMetric.euclidean,
) -> tf.Tensor:
    """
    Low-poly art style effect

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        distrib: Distrib
        freq: Noise frequency
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier
        dist_metric: Distance metric to use

    Returns:
        Modified tensor
    """

    # Convert freq to int if it's a list
    if isinstance(freq, list):
        freq = freq[0]

    xy = point_cloud(freq, distrib=distrib, shape=shape, drift=1.0, time=time, speed=speed)  # type: ignore[arg-type]

    distance = value.voronoi(tensor, shape, nth=1, xy=xy, dist_metric=dist_metric)
    color = value.voronoi(tensor, shape, diagram_type=VoronoiDiagramType.color_regions, xy=xy, dist_metric=dist_metric)

    return value.normalize(value.blend(distance, color, 0.5))


def square_crop_and_resize(tensor: tf.Tensor, shape: list[int], length: int = 1024) -> tf.Tensor:
    """
    Crop and resize an image Tensor into a square with desired side length.

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        length: Length

    Returns:
        Modified tensor
    """

    height, width, channels = shape

    have_length = min(height, width)

    if height != width:
        tensor = tf.image.resize_with_crop_or_pad(tensor, have_length, have_length)

    if length != have_length:
        tensor = value.resample(tensor, [length, length, channels])

    return tensor


@effect()
def kaleido(
    tensor: tf.Tensor,
    shape: list[int],
    sides: int = 6,
    sdf_sides: int = 5,
    xy: tf.Tensor | None = None,
    blend_edges: bool = True,
    time: float = 0.0,
    speed: float = 1.0,
    point_freq: int = 1,
    point_generations: int = 1,
    point_distrib: PointDistribution = PointDistribution.random,
    point_drift: float = 0.0,
    point_corners: bool = False,
) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        sides: Number of polygon sides
        sdf_sides: SDF polygon sides for distance metric
        xy: Optional XY coordinates for point cloud
        blend_edges: Blend with original edges
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier
        point_freq: Point frequency for Voronoi
        point_generations: Point generation count
        point_distrib: Point distribution method
        point_drift: Point drift amount
        point_corners: Include corner points

    Returns:
        Modified tensor
    """

    height, width, channels = shape

    x_identity = tf.cast(value.row_index(shape), tf.float32)
    y_identity = tf.cast(value.column_index(shape), tf.float32)

    # indices offset to center
    x_index = value.normalize(tf.cast(x_identity, tf.float32)) - 0.5
    y_index = value.normalize(tf.cast(y_identity, tf.float32)) - 0.5

    value_shape = value.value_shape(shape)

    if sdf_sides < 3:
        dist_metric = DistanceMetric.euclidean
    else:
        dist_metric = DistanceMetric.sdf

    # distance from any pixel to center
    r = value.voronoi(
        None,
        value_shape,
        dist_metric=dist_metric,
        sdf_sides=sdf_sides,
        xy=xy,
        point_freq=point_freq,
        point_generations=point_generations,
        point_distrib=point_distrib,
        point_drift=point_drift,
        point_corners=point_corners,
    )

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
def palette(tensor: tf.Tensor, shape: list[int], name: str | None = None, alpha: float = 1.0, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Another approach to image coloration
    https://iquilezles.org/www/articles/palettes/palettes.htm

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        name: Name
        alpha: Blending alpha value (0.0-1.0)
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
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

    p: Any = palettes[name]

    offset = p["offset"] * tf.ones(rgb_shape, dtype=tf.float32)
    amp = p["amp"] * tf.ones(rgb_shape, dtype=tf.float32)
    freq = p["freq"] * tf.ones(rgb_shape, dtype=tf.float32)
    phase = p["phase"] * tf.ones(rgb_shape, dtype=tf.float32) + time

    # Multiply value_map's result x .875, in case the image is just black and white (0 == 1, we don't want a solid color image)
    colored = offset + amp * tf.math.cos(math.tau * (freq * value.value_map(tensor, shape, keepdims=True, with_normalize=False) * 0.875 + 0.0625 + phase))

    tensor = value.blend_cosine(tensor, colored, alpha)

    # Re-insert the alpha channel
    if shape[2] == 4:
        tensor = tf.stack([tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2], alpha_channel], 2)

    return tensor


@effect()
@effect()
def vhs(tensor: tf.Tensor, shape: list[int], time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Apply a bad VHS tracking effect.

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    height, width, channels = shape

    # Generate scan noise
    scan_noise = value.values(
        freq=int(height * 0.5) + 1,
        shape=[height, width, 1],
        time=time,
        speed=speed * 100,
    )

    # Create horizontal offsets
    grad = value.values(
        freq=[5, 1],
        shape=[height, width, 1],
        time=time,
        speed=speed,
    )
    grad = tf.maximum(grad - 0.5, 0)
    grad = tf.minimum(grad * 2, 1)

    x_index = value.row_index(shape)
    x_index -= tf.squeeze(tf.cast(scan_noise * width * tf.square(grad), tf.int32))
    x_index = x_index % width

    tensor = value.blend(tensor, scan_noise, grad)

    identity = tf.stack([value.column_index(shape), x_index], 2)

    tensor = tf.gather_nd(tensor, identity)

    return tensor


@effect()
def lens_warp(tensor: tf.Tensor, shape: list[int], displacement: float = 0.0625, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        displacement: Displacement amount
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    value_shape = value.value_shape(shape)

    # Fake CRT lens shape
    mask = tf.pow(value.singularity(None, value_shape), 5)  # obscure center pinch

    # Displacement values multiplied by mask to make it wavy towards the edges
    distortion_x = (value.values(2, value_shape, time=time, speed=speed, spline_order=2) * 2.0 - 1.0) * mask

    return value.refract(tensor, shape, displacement, reference_x=distortion_x)


@effect()
def lens_distortion(tensor: tf.Tensor, shape: list[int], displacement: float = 1.0, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        displacement: Displacement amount
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    x_index = tf.cast(value.row_index(shape), tf.float32) / shape[1]
    y_index = tf.cast(value.column_index(shape), tf.float32) / shape[0]

    x_dist = x_index - 0.5
    y_dist = y_index - 0.5

    center_dist = 1.0 - value.normalize(value.distance(x_dist, y_dist))

    if displacement < 0.0:
        zoom = displacement * -0.25
    else:
        zoom = 0.0

    x_offset = tf.cast(((x_index - x_dist * zoom) - x_dist * center_dist * center_dist * displacement) * shape[1], tf.int32) % shape[1]
    y_offset = tf.cast(((y_index - y_dist * zoom) - y_dist * center_dist * center_dist * displacement) * shape[0], tf.int32) % shape[0]

    return tf.gather_nd(tensor, tf.stack([y_offset, x_offset], 2))


@effect()
def degauss(tensor: tf.Tensor, shape: list[int], displacement: float = 0.0625, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        displacement: Displacement amount
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    channel_shape = [shape[0], shape[1], 1]

    red = lens_warp(tf.expand_dims(tensor[:, :, 0], -1), channel_shape, displacement=displacement, time=time, speed=speed)
    green = lens_warp(tf.expand_dims(tensor[:, :, 1], -1), channel_shape, displacement=displacement, time=time, speed=speed)
    blue = lens_warp(tf.expand_dims(tensor[:, :, 2], -1), channel_shape, displacement=displacement, time=time, speed=speed)

    return tf.stack([tf.squeeze(red), tf.squeeze(green), tf.squeeze(blue)], 2)


@effect()
def crt(tensor: tf.Tensor, shape: list[int], time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Apply vintage CRT scanlines.

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    height, width, channels = shape

    value_shape = value.value_shape(shape)

    # Horizontal scanlines
    scan_noise = value.normalize(value.values(freq=[2, 1], shape=[2, 1, 1], time=time, speed=speed * 0.1, spline_order=0))

    tile_h = max(1, int(height * 0.125))
    scan_noise = expand_tile(scan_noise, [2, 1, 1], [tile_h * 2, width, 1], with_offset=False)

    scan_noise = value.resample(scan_noise, value_shape)

    scan_noise = lens_warp(scan_noise, value_shape, time=time, speed=speed)

    tensor = value.clamp01(value.blend(tensor, (tensor + scan_noise) * scan_noise, 0.05))

    if channels == 3:
        tensor = aberration(tensor, shape, 0.0125 + rng.random() * 0.00625)
        tensor = adjust_hue(tensor, shape, rng.random() * 0.25 - 0.125)
        tensor = tf.image.adjust_saturation(tensor, 1.125)

    tensor = vignette(tensor, shape, brightness=0, alpha=rng.random() * 0.175)
    mean = tf.reduce_mean(tensor)
    tensor = (tensor - mean) * 1.25 + mean

    return tensor


@effect()
def scanline_error(tensor: tf.Tensor, shape: list[int], time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    height, width, channels = shape

    value_shape = value.value_shape(shape)

    error_freq = [int(value_shape[0] * 0.5) or 1, int(value_shape[1] * 0.5) or 1]

    error_line = tf.maximum(value.values(freq=error_freq, shape=value_shape, time=time, speed=speed * 10, distrib=ValueDistribution.exp) - 0.5, 0)
    error_swerve = tf.maximum(value.values(freq=[int(height * 0.01), 1], shape=value_shape, time=time, speed=speed, distrib=ValueDistribution.exp) - 0.5, 0)

    error_line *= error_swerve

    error_swerve *= 2

    white_noise = value.values(freq=error_freq, shape=value_shape, time=time, speed=speed * 100)
    white_noise = value.blend(0, white_noise, error_swerve)

    error = error_line + white_noise

    y_index = value.column_index(shape)
    x_index = (value.row_index(shape) - tf.cast(value.value_map(error, value_shape) * width * 0.025, tf.int32)) % width

    return tf.minimum(tf.gather_nd(tensor, tf.stack([y_index, x_index], 2)) + error_line * white_noise * 4, 1)


@effect()
def snow(tensor: tf.Tensor, shape: list[int], alpha: float = 0.25, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        alpha: Blending alpha value (0.0-1.0)
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    height, width, channels = shape

    value_shape = value.value_shape(shape)

    static = value.values(freq=[height, width], shape=value_shape, time=time, speed=speed * 100, spline_order=0)

    static_limiter = value.values(freq=[height, width], shape=value_shape, time=time, speed=speed * 100, distrib=ValueDistribution.exp, spline_order=0) * alpha

    return value.blend(tensor, static, static_limiter)


@effect()
def grain(tensor: tf.Tensor, shape: list[int], alpha: float = 0.25, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        alpha: Blending alpha value (0.0-1.0)
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    height, width, channels = shape

    white_noise = value.values(freq=[height, width], shape=[height, width, 1], time=time, speed=speed * 100)

    return value.blend(tensor, white_noise, alpha)


@effect()
def false_color(tensor: tf.Tensor, shape: list[int], horizontal: bool = False, displacement: float = 0.5, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        horizontal: Apply horizontally (vs vertically)
        displacement: Displacement amount
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    clut = value.values(freq=2, shape=shape, time=time, speed=speed)

    return value.normalize(color_map(tensor, shape, clut=clut, horizontal=horizontal, displacement=displacement))


@effect()
def fibers(tensor: tf.Tensor, shape: list[int], time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    value_shape = value.value_shape(shape)

    for i in range(4):
        mask = value.values(freq=4, shape=value_shape, time=time, speed=speed)

        mask = worms(
            mask,
            shape,
            behavior=WormBehavior.chaotic,
            alpha=1,
            density=0.05 + rng.random() * 0.00125,
            duration=1,
            kink=rng.random_int(5, 10),
            stride=0.75,
            stride_deviation=0.125,
            time=time,
            speed=speed,
        )

        brightness = value.values(freq=128, shape=shape, time=time, speed=speed)

        tensor = value.blend(tensor, brightness, mask * 0.5)

    return tensor


@effect()
def scratches(tensor: tf.Tensor, shape: list[int], time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    value_shape = value.value_shape(shape)

    for i in range(4):
        mask = value.values(freq=rng.random_int(2, 4), shape=value_shape, time=time, speed=speed)

        mask = worms(
            mask,
            value_shape,
            behavior=[1, 3][rng.random_int(0, 1)],
            alpha=1,
            density=0.25 + rng.random() * 0.25,
            duration=2 + rng.random() * 2,
            kink=0.125 + rng.random() * 0.125,
            stride=0.75,
            stride_deviation=0.5,
            time=time,
            speed=speed,
        )

        mask -= value.values(freq=rng.random_int(2, 4), shape=value_shape, time=time, speed=speed) * 2.0

        mask = tf.maximum(mask, 0.0)

        tensor = tf.maximum(tensor, mask * 8.0)

        tensor = tf.minimum(tensor, 1.0)

    return tensor


@effect()
def stray_hair(tensor: tf.Tensor, shape: list[int], time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    value_shape = value.value_shape(shape)

    mask = value.values(4, value_shape, time=time, speed=speed)

    mask = worms(
        mask,
        value_shape,
        behavior=WormBehavior.unruly,
        alpha=1,
        density=0.0025 + rng.random() * 0.00125,
        duration=rng.random_int(8, 16),
        kink=rng.random_int(5, 50),
        stride=0.5,
        stride_deviation=0.25,
    )

    brightness = value.values(freq=32, shape=value_shape, time=time, speed=speed)

    return value.blend(tensor, brightness * 0.333, mask * 0.666)


@effect()
def grime(tensor: tf.Tensor, shape: list[int], time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    value_shape = value.value_shape(shape)

    mask = value.simple_multires(freq=5, shape=value_shape, time=time, speed=speed, octaves=8)

    mask = value.refract(mask, value_shape, 1.0, y_from_offset=True)
    mask = derivative(mask, value_shape, DistanceMetric.chebyshev, alpha=0.125)

    dusty = value.blend(tensor, 0.25, tf.square(mask) * 0.075)

    specks = value.values(
        freq=[int(shape[0] * 0.25), int(shape[1] * 0.25)], shape=value_shape, time=time, mask=ValueMask.dropout, speed=speed, distrib=ValueDistribution.exp
    )
    specks = value.refract(specks, value_shape, 0.25)

    specks = 1.0 - tf.sqrt(value.normalize(tf.maximum(specks - 0.625, 0.0)))

    dusty = (
        value.blend(
            dusty,
            value.values(freq=[shape[0], shape[1]], shape=value_shape, mask=ValueMask.sparse, time=time, speed=speed, distrib=ValueDistribution.exp),
            0.075,
        )
        * specks
    )

    return value.blend(tensor, dusty, mask * 0.75)


@effect()
def frame(tensor: tf.Tensor, shape: list[int], time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    half_shape = [int(shape[0] * 0.5), int(shape[1] * 0.5), shape[2]]
    half_value_shape = value.value_shape(half_shape)

    noise = value.simple_multires(64, half_value_shape, time=time, speed=speed, octaves=8)

    black = tf.zeros(half_value_shape, dtype=tf.float32)
    white = tf.ones(half_value_shape, dtype=tf.float32)

    mask = value.singularity(None, half_value_shape, VoronoiDiagramType.range, dist_metric=DistanceMetric.chebyshev, inverse=True)
    mask = value.normalize(mask + noise * 0.005)
    mask = blend_layers(tf.sqrt(mask), half_value_shape, 0.0125, white, black, black, black)

    faded = value.proportional_downsample(tensor, shape, half_shape)
    faded = tf.image.adjust_brightness(faded, 0.1)
    faded = tf.image.adjust_contrast(faded, 0.75)
    faded = light_leak(faded, half_shape, 0.125)
    faded = vignette(faded, half_shape, 0.05, 0.75)

    edge_texture = white * 0.9 + shadow(noise, half_value_shape, alpha=1.0) * 0.1

    out = value.blend(faded, edge_texture, mask)
    out = aberration(out, half_shape, 0.00666)
    out = grime(out, half_shape)

    out = tf.image.adjust_saturation(out, 0.5)
    out = tf.image.random_hue(out, 0.05, seed=rng.random_int(0, 0xFFFFFFFF))

    out = value.resample(out, shape)

    out = scratches(out, shape)

    out = stray_hair(out, shape)

    return out


@effect()
def texture(tensor: tf.Tensor, shape: list[int], time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    value_shape = value.value_shape(shape)

    noise = value.simple_multires(64, value_shape, time=time, speed=speed, octaves=8, ridges=True)

    return tensor * (tf.ones(value_shape, dtype=tf.float32) * 0.9 + shadow(noise, value_shape, 1.0) * 0.1)


@effect()
def watermark(tensor: tf.Tensor, shape: list[int], time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    value_shape = value.value_shape(shape)

    mask = value.values(freq=240, shape=value_shape, spline_order=0, distrib=ValueDistribution.ones, mask=ValueMask.alphanum_numeric)

    mask = crt(mask, value_shape)

    mask = warp(mask, value_shape, [2, 4], octaves=1, displacement=0.5, time=time, speed=speed)

    mask *= tf.square(value.values(freq=2, shape=value_shape, time=time, speed=speed))

    brightness = value.values(freq=16, shape=value_shape, time=time, speed=speed)

    return value.blend(tensor, brightness, mask * 0.125)


@effect()
def spooky_ticker(tensor: tf.Tensor, shape: list[int], time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    if rng.random() > 0.75:
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

    rendered_mask = tf.zeros(shape, dtype=tf.float32)

    for _ in range(rng.random_int(1, 3)):
        mask = _masks[rng.random_int(0, len(_masks) - 1)]
        mask_shape = masks.mask_shape(mask)

        multiplier = 1 if mask != ValueMask.script and (mask_shape[1] == 1 or mask_shape[1] >= 10) else 2

        width = int(shape[1] / multiplier) or 1
        width = mask_shape[1] * int(width / mask_shape[1])  # Make sure the mask divides evenly into width

        freq = [mask_shape[0], width]

        row_shape = [mask_shape[0], width, 1]
        row_mask = value.values(freq=freq, shape=row_shape, corners=True, spline_order=0, distrib=ValueDistribution.ones, mask=mask, time=time, speed=speed)

        if time != 0.0:  # Make the ticker tick!
            row_mask = value.offset(row_mask, row_shape, int(time * width), 0)

        row_mask = value.resample(row_mask, [mask_shape[0] * multiplier, shape[1]], spline_order=1)

        rendered_mask += tf.pad(row_mask, tf.stack([[shape[0] - mask_shape[0] * multiplier - bottom_padding, bottom_padding], [0, 0], [0, 0]]))

        bottom_padding += mask_shape[0] * multiplier + 2

    alpha = 0.5 + rng.random() * 0.25

    # shadow
    tensor = value.blend(tensor, tensor * 1.0 - value.offset(rendered_mask, shape, -1, -1), alpha * 0.333)

    return value.blend(tensor, tf.maximum(rendered_mask, tensor), alpha)


@effect()
def on_screen_display(tensor: tf.Tensor, shape: list[int], time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """
    glyph_count = rng.random_int(3, 6)

    _masks = [
        ValueMask.bank_ocr,
        ValueMask.alphanum_hex,
        ValueMask.alphanum_numeric,
    ]

    mask = _masks[rng.random_int(0, len(_masks) - 1)]
    mask_shape = masks.mask_shape(mask)

    width = int(shape[1] / 24)

    width = mask_shape[1] * int(width / mask_shape[1])  # Make sure the mask divides evenly
    height = mask_shape[0] * int(width / mask_shape[1])

    width *= glyph_count

    freq = [mask_shape[0], mask_shape[1] * glyph_count]

    row_mask = value.values(
        freq=freq, shape=[height, width, shape[2]], corners=True, spline_order=0, distrib=ValueDistribution.ones, mask=mask, time=time, speed=speed
    )

    rendered_mask = tf.pad(row_mask, tf.stack([[25, shape[0] - height - 25], [shape[1] - width - 25, 25], [0, 0]]))

    alpha = 0.5 + rng.random() * 0.25

    return value.blend(tensor, tf.maximum(rendered_mask, tensor), alpha)


@effect()
def nebula(tensor: tf.Tensor, shape: list[int], time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """
    value_shape = value.value_shape(shape)

    overlay = value.simple_multires([rng.random_int(3, 4), 1], value_shape, time=time, speed=speed, distrib=ValueDistribution.exp, ridges=True, octaves=6)

    overlay -= value.simple_multires([rng.random_int(2, 4), 1], value_shape, time=time, speed=speed, ridges=True, octaves=4)

    overlay *= 0.125

    overlay = rotate(overlay, value_shape, angle=rng.random_int(-15, 15), time=time, speed=speed)

    tensor *= 1.0 - overlay

    tensor += tint(
        tf.maximum(overlay * tf.ones(shape, dtype=tf.float32), 0),
        shape,
        alpha=1.0,
        time=time,
        speed=1.0,
    )

    return tensor


@effect()
def spatter(tensor: tf.Tensor, shape: list[int], color: bool = True, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        color: Color
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    value_shape = value.value_shape(shape)

    # Generate a smear
    smear = value.simple_multires(rng.random_int(3, 6), value_shape, time=time, speed=speed, distrib=ValueDistribution.exp, octaves=6, spline_order=3)

    smear = warp(
        smear,
        value_shape,
        [rng.random_int(2, 3), rng.random_int(1, 3)],
        octaves=rng.random_int(1, 2),
        displacement=1.0 + rng.random(),
        spline_order=3,
        time=time,
        speed=speed,
    )

    # Add spatter dots
    spatter = value.simple_multires(
        rng.random_int(32, 64), value_shape, time=time, speed=speed, distrib=ValueDistribution.exp, octaves=4, spline_order=InterpolationType.linear
    )

    spatter = adjust_brightness(spatter, shape, -1.0)
    spatter = adjust_contrast(spatter, shape, 4.0)

    smear = tf.maximum(smear, spatter)

    spatter = value.simple_multires(
        rng.random_int(150, 200), value_shape, time=time, speed=speed, distrib=ValueDistribution.exp, octaves=4, spline_order=InterpolationType.linear
    )

    spatter = adjust_brightness(spatter, shape, -1.25)
    spatter = adjust_contrast(spatter, shape, 4.0)

    smear = tf.maximum(smear, spatter)

    # Remove some of it
    smear = tf.maximum(
        0.0,
        smear
        - value.simple_multires(
            rng.random_int(2, 3), value_shape, time=time, speed=speed, distrib=ValueDistribution.exp, ridges=True, octaves=3, spline_order=2
        ),
    )

    #
    if color and shape[2] == 3:
        if color is True:
            splash = tf.image.random_hue(tf.ones(shape, dtype=tf.float32) * tf.stack([0.875, 0.125, 0.125]), 0.5, seed=rng.random_int(0, 0xFFFFFFFF))

        else:  # Pass in [r, g, b]
            splash = tf.ones(shape, dtype=tf.float32) * tf.stack(color)

    else:
        splash = tf.zeros(shape, dtype=tf.float32)

    return blend_layers(value.normalize(smear), shape, 0.005, tensor, splash * tensor)


@effect()
def clouds(tensor: tf.Tensor, shape: list[int], time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Top-down cloud cover effect

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    pre_shape = [int(shape[0] * 0.25) or 1, int(shape[1] * 0.25) or 1, 1]

    control = value.simple_multires(freq=rng.random_int(2, 4), shape=pre_shape, octaves=8, ridges=True, time=time, speed=speed)

    control = warp(control, pre_shape, freq=3, displacement=0.125, octaves=2)

    layer_0 = tf.ones(pre_shape, dtype=tf.float32)
    layer_1 = tf.zeros(pre_shape, dtype=tf.float32)

    combined = blend_layers(control, pre_shape, 1.0, layer_0, layer_1)

    shaded = value.offset(combined, pre_shape, rng.random_int(-15, 15), rng.random_int(-15, 15))
    shaded = tf.minimum(shaded * 2.5, 1.0)

    for _ in range(3):
        shaded = value.convolve(kernel=ValueMask.conv2d_blur, tensor=shaded, shape=pre_shape)

    post_shape = [shape[0], shape[1], 1]

    shaded = value.resample(shaded, post_shape)
    combined = value.resample(combined, post_shape)

    tensor = value.blend(tensor, tf.zeros(shape, dtype=tf.float32), shaded * 0.75)
    tensor = value.blend(tensor, tf.ones(shape, dtype=tf.float32), combined)

    tensor = shadow(tensor, shape, alpha=0.5)

    return tensor


@effect()
def tint(tensor: tf.Tensor, shape: list[int], time: float = 0.0, speed: float = 1.0, alpha: float = 0.5) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier
        alpha: Blending alpha value (0.0-1.0)

    Returns:
        Modified tensor
    """

    if shape[2] < 3:  # Not a color image
        return tensor

    color = value.values(freq=3, shape=shape, time=time, speed=speed, corners=True)

    # Confine hue to a range
    color = tf.stack([(tensor[:, :, 0] * 0.333 + rng.random() * 0.333 + rng.random()) % 1.0, tensor[:, :, 1], tensor[:, :, 2]], 2)

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
def adjust_hue(tensor: tf.Tensor, shape: list[int], amount: float = 0.25, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        amount: Amount
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """
    if amount not in (1.0, 0.0, None) and shape[2] == 3:
        tensor = tf.image.adjust_hue(tensor, amount)

    return tensor


@effect()
def adjust_saturation(tensor: tf.Tensor, shape: list[int], amount: float = 0.75, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        amount: Amount
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """
    if shape[2] == 3:
        tensor = tf.image.adjust_saturation(tensor, amount)

    return tensor


@effect()
def adjust_brightness(tensor: tf.Tensor, shape: list[int], amount: float = 0.125, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        amount: Amount
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """
    return tf.maximum(tf.minimum(tf.image.adjust_brightness(tensor, amount), 1.0), -1.0)


@effect()
def adjust_contrast(tensor: tf.Tensor, shape: list[int], amount: float = 1.25, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        amount: Amount
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """
    return value.clamp01(tf.image.adjust_contrast(tensor, amount))


@effect()
def normalize(tensor: tf.Tensor, shape: list[int], time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """
    return value.normalize(tensor)


@effect()
def ridge(tensor: tf.Tensor, shape: list[int], time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """
    return value.ridge(tensor)


@effect()
def sine(tensor: tf.Tensor, shape: list[int], amount: float = 1.0, time: float = 0.0, speed: float = 1.0, rgb: bool = False) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        amount: Amount
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier
        rgb: Treat as RGB (vs grayscale)

    Returns:
        Modified tensor
    """
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
def value_refract(
    tensor: tf.Tensor,
    shape: list[int],
    freq: int | list[int] = 4,
    distrib: int | PointDistribution | ValueDistribution = ValueDistribution.center_circle,
    displacement: float = 0.125,
    time: float = 0.0,
    speed: float = 1.0,
) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        freq: Noise frequency
        distrib: Distrib
        displacement: Displacement amount
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """

    blend_values = value.values(freq=freq, shape=value.value_shape(shape), distrib=distrib, time=time, speed=speed)  # type: ignore[arg-type]

    return value.refract(tensor, shape, time=time, speed=speed, reference_x=blend_values, displacement=displacement)


@effect()
def blur(
    tensor: tf.Tensor,
    shape: list[int],
    amount: float = 10.0,
    spline_order: int | InterpolationType = InterpolationType.bicubic,
    time: float = 0.0,
    speed: float = 1.0,
) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        amount: Amount
        spline_order: Interpolation type for resampling
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Modified tensor
    """
    ""
    ""

    tensor = value.proportional_downsample(tensor, shape, [max(int(shape[0] / amount), 1), max(int(shape[1] / amount), 1), shape[2]]) * 4.0

    tensor = value.resample(tensor, shape, spline_order=spline_order)

    return tensor
