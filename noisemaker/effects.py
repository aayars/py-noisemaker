from collections import defaultdict
from enum import Enum

import math
import random

import numpy as np
import tensorflow as tf


def post_process(tensor, shape, refract_range=0.0, reindex_range=0.0, clut=None, clut_horizontal=False, clut_range=0.5,
                 with_worms=False, worm_behavior=None, worm_density=4.0, worm_duration=4.0, worm_stride=1.0, worm_stride_deviation=.05,
                 worm_bg=.5, worm_kink=1.0, with_sobel=False, with_normal_map=False, deriv=False, with_wormhole=False, wormhole_kink=2.5, wormhole_stride=.1):
    """
    Apply post-processing effects.

    :param Tensor tensor:
    :param list[int] shape:
    :param float refract_range: Self-distortion gradient.
    :param float reindex_range: Self-reindexing gradient.
    :param str clut: PNG or JPG color lookup table filename.
    :param float clut_horizontal: Preserve clut Y axis.
    :param float clut_range: Gather range for clut.
    :param bool with_worms: Do worms.
    :param WormBehavior worm_behavior:
    :param float worm_density: Worm density multiplier (larger == slower)
    :param float worm_duration: Iteration multiplier (larger == slower)
    :param float worm_stride: Mean travel distance per iteration
    :param float worm_stride_deviation: Per-worm travel distance deviation
    :param float worm_bg: Background color brightness for worms
    :param float worm_kink: Worm twistiness
    :param bool with_sobel: Sobel operator
    :param bool with_normal_map: Create a tangent-space normal map
    :param bool with_wormhole: Wormhole effect. What is this?
    :param float wormhole_kink: Wormhole kinkiness, if you're into that.
    :param float wormhole_stride: Wormhole thickness range
    :param bool deriv: Derivative operator

    :return: Tensor
    """

    if refract_range != 0:
        tensor = refract(tensor, shape, displacement=refract_range)

    if reindex_range != 0:
        tensor = reindex(tensor, shape, displacement=reindex_range)

    if clut:
        tensor = color_map(tensor, clut, shape, horizontal=clut_horizontal, displacement=clut_range)

    else:
        tensor = normalize(tensor)

    if deriv:
        tensor = derivative(tensor)

    if with_worms:
        tensor = worms(tensor, shape, behavior=worm_behavior, density=worm_density, duration=worm_duration,
                       stride=worm_stride, stride_deviation=worm_stride_deviation, bg=worm_bg, kink=worm_kink)

    if with_wormhole:
        tensor = wormhole(tensor, shape, wormhole_kink, wormhole_stride)

    if with_sobel:
        tensor = sobel(tensor, shape)

    if with_normal_map:
        tensor = normal_map(tensor, shape)

    return tensor


class WormBehavior(Enum):
    """
    Specify the type of heading bias for worms to follow.

    .. code-block:: python

       image = worms(image, behavior=WormBehavior.unruly)
    """

    obedient = 0

    crosshatch = 1

    unruly = 2

    chaotic = 3


class ConvKernel(Enum):
    """
    A collection of convolution kernels for image post-processing, based on well-known recipes.

    Pass the desired kernel as an argument to :py:func:`convolve`.

    .. code-block:: python

       image = convolve(ConvKernel.shadow, image)
    """

    emboss = [
        [   0,   2,   4   ],
        [  -2,   1,   2   ],
        [  -4,  -2,   0   ]
    ]

    rand = np.random.normal(.5, .5, (5, 5)).tolist()

    shadow = [
        # yeah, one of the really big fuckers
        # [  0,   1,   1,   1,   1,   1,   1  ],
        # [ -1,   0,   2,   2,   1,   1,   1  ],
        # [ -1,  -2,   0,   4,   2,   1,   1  ],
        # [ -1,  -2,  -4,   12,   4,   2,   1  ],
        # [ -1,  -1,  -2,  -4,   0,   2,   1  ],
        # [ -1,  -1,  -1,  -2,  -2,   0,   1  ],
        # [ -1,  -1,  -1,  -1,  -1,  -1,   0  ]

        # [  0,  1,  1,  1, 0 ],
        # [ -1, -2,  4,  2, 1 ],
        # [ -1, -4,  2,  4, 1 ],
        # [ -1, -2, -4,  2, 1 ],
        # [  0, -1, -1, -1, 0 ]

        [  0,  1,  1,  1, 0 ],
        [ -1, -2,  4,  2, 1 ],
        [ -1, -4,  2,  4, 1 ],
        [ -1, -2, -4,  2, 1 ],
        [  0, -1, -1, -1, 0 ]
    ]

    edges = [
        [   1,   2,  1   ],
        [   2, -12,  2   ],
        [   1,   2,  1   ]
    ]

    sharpen = [
        [   0, -1,  0 ],
        [  -1,  5, -1 ],
        [   0, -1,  0 ]
    ]

    unsharp_mask = [
        [ 1,  4,     6,   4, 1 ],
        [ 4,  16,   24,  16, 4 ],
        [ 6,  24, -476,  24, 6 ],
        [ 4,  16,   24,  16, 4 ],
        [ 1,  4,     6,   4, 1 ]
    ]

    invert = [
        [ 0,  0,  0 ],
        [ 0, -1,  0 ],
        [ 0,  0,  0 ]
    ]

    sobel_x = [
        [ 1, 0, -1 ],
        [ 2, 0, -2 ],
        [ 1, 0, -1 ]
    ]

    sobel_y = [
        [  1,  2,  1 ],
        [  0,  0,  0 ],
        [ -1, -2, -1 ]
    ]


def _conform_kernel_to_tensor(kernel, tensor, shape):
    """ Re-shape a convolution kernel to match the given tensor's color dimensions. """

    l = len(kernel)

    channels = shape[-1]

    temp = np.repeat(kernel, channels)

    temp = tf.reshape(temp, (l, l, channels, 1))

    temp = tf.image.convert_image_dtype(temp, tf.float32)

    return temp


def convolve(kernel, tensor, shape):
    """
    Apply a convolution kernel to an image tensor.

    :param ConvKernel kernel: See ConvKernel enum
    :param Tensor tensor: An image tensor.
    :return: Tensor
    """

    height, width, channels = shape

    kernel_values = _conform_kernel_to_tensor(kernel.value, tensor, shape)

    # Give the conv kernel some room to play on the edges
    half_height = tf.cast(shape[0] / 2, tf.int32)
    half_width = tf.cast(shape[1] / 2, tf.int32)

    tensor = tf.tile(tensor, [3, 3, 1])  # Tile 3x3
    tensor = tensor[half_height:shape[0] * 2 + half_height, half_width:shape[1] * 2 + half_width]  # Center Crop 2x2
    tensor = tf.nn.depthwise_conv2d([tensor], kernel_values, [1,1,1,1], "VALID")[0]
    tensor = tensor[half_height:shape[0] + half_height, half_width:shape[1] + half_width]  # Center Crop 1x1
    tensor = normalize(tensor)

    if kernel == ConvKernel.edges:
        tensor = tf.abs(tensor - .5) * 2

    return tensor


def normalize(tensor):
    """
    Squeeze the given Tensor into a range between 0 and 1.

    :param Tensor tensor: An image tensor.
    :return: Tensor
    """

    return (tensor - tf.reduce_min(tensor)) / (tf.reduce_max(tensor) - tf.reduce_min(tensor))


def resample(tensor, shape, spline_order=3):
    """
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

        return blend_cubic(*points, resized_col_index_fract)


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

    return 1 - tf.abs(tensor * 2 - 1)


def derivative(tensor):
    """
    Extract a derivative from the given noise.

    .. image:: images/derived.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :return: Tensor
    """

    y, x = np.gradient(tensor.eval(), axis=(0, 1))  # Do this in TF with conv2D?

    return normalize(tf.sqrt(y*y + x*x))


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
    offset = tf.cast((reference * displacement * mod + reference) % mod, tf.int32)

    tensor = tf.gather_nd(tensor, tf.stack([offset, offset], 2))

    return tensor


def refract(tensor, shape, displacement=.5, reference=None):
    """
    Apply self-displacement along X and Y axes, based on each pixel value.

    .. image:: images/refract.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor: An image tensor.
    :param list[int] shape:
    :param float displacement:
    :param Tensor reference: An optional displacement map.
    :return: Tensor
    """

    height, width, channels = shape

    if reference is None:
        reference = tensor

    reference_x = value_map(reference, shape) * displacement

    x_index = row_index(shape)
    y_index = column_index(shape)

    # Create an offset Y channel, to get rid of diagonal banding.
    reference_y = tf.gather_nd(reference_x, offset_index(y_index, height, x_index, width))

    return tf.gather_nd(tensor, offset_index(y_index + tf.cast(reference_y * height, tf.int32), height, x_index + tf.cast(reference_x * width, tf.int32), width))


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
        with open(clut, "rb") as fh:
            if clut.endswith(".png"):
                clut = tf.image.decode_png(fh.read(), channels=3)

            elif clut.endswith(".jpg"):
                clut = tf.image.decode_jpeg(fh.read(), channels=3)

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


def worms(tensor, shape, behavior=0, density=4.0, duration=4.0, stride=1.0, stride_deviation=.05, bg=.5, kink=1.0):
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
    :param float bg: Background color intensity.
    :param float kink: Make your worms twist.
    :return: Tensor
    """

    height, width, channels = shape

    count = int(max(width, height) * density)

    worms_y = tf.random_uniform([count]) * (height - 1)
    worms_x = tf.random_uniform([count]) * (width - 1)
    worms_stride = tf.random_normal([count], mean=stride, stddev=stride_deviation)

    colors = tf.gather_nd(tensor, tf.cast(tf.stack([worms_y, worms_x], 1), tf.int32))

    if isinstance(behavior, int):
        behavior = WormBehavior(behavior)

    if behavior == WormBehavior.obedient:
        worms_rot = tf.zeros([count])

    elif behavior == WormBehavior.crosshatch:
        worms_rot = (tf.floor(tf.random_normal([count]) * 100) % 2) * 90

    elif behavior == WormBehavior.chaotic:
        worms_rot = tf.random_normal([count]) * 360.0

    else:
        worms_rot = tf.random_normal([count])

    index = value_map(tensor, shape) * 360.0 * math.radians(1) * kink

    iterations = int(math.sqrt(min(width, height)) * duration)

    out = tensor * bg

    scatter_shape = tf.shape(tensor)  # Might be different than `shape` due to clut

    # Make worms!
    for i in range(iterations):
        worm_positions = tf.cast(tf.stack([worms_y, worms_x], 1), tf.int32)

        exposure = 1 - abs(1 - i / (iterations - 1) * 2)  # Makes linear gradient [ 0 .. 1 .. 0 ]

        out += tf.scatter_nd(worm_positions, colors * exposure, scatter_shape)

        next_position = tf.gather_nd(index, worm_positions) + worms_rot

        worms_y = (worms_y + tf.cos(next_position) * worms_stride) % height
        worms_x = (worms_x + tf.sin(next_position) * worms_stride) % width

    out = tf.image.convert_image_dtype(out, tf.float32, saturate=True)

    return tf.sqrt(normalize(out))


def wormhole(tensor, shape, kink, input_stride):
    """
    """

    height, width, channels = shape

    values = value_map(tensor, shape)
    degrees = values * 360.0 * math.radians(1) * kink
    stride = values * height * input_stride

    x_index = tf.cast(row_index(shape), tf.float32)
    y_index = tf.cast(column_index(shape), tf.float32)

    x_offset = (tf.cos(degrees) + 1) * stride
    y_offset = (tf.sin(degrees) + 1) * stride

    x = tf.cast(x_index + x_offset, tf.int32) % width
    y = tf.cast(y_index + y_offset, tf.int32) % height

    out = tf.scatter_nd(offset_index(y, height, x, width), tensor, tf.shape(tensor))

    return tf.sqrt(tf.maximum(out, tensor))
    return tf.sqrt(out)


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

    return normalize(tensor - resample(resample(tensor, [int(height * .5), int(width * .5), channels]), shape))


def sobel(tensor, shape):
    """
    Apply a sobel operator.

    :param Tensor tensor:
    :param list[int] shape:
    :return: Tensor
    """

    x = convolve(ConvKernel.sobel_x, tensor, shape)
    y = convolve(ConvKernel.sobel_y, tensor, shape)

    return tf.abs(normalize(tf.sqrt(x * x + y * y)) * 2 - 1)


def normal_map(tensor, shape):
    """
    Generate a tangent-space normal map.

    :param Tensor tensor:
    :param list[int] shape:
    :return: Tensor
    """

    height, width, channels = shape

    reference = value_map(tensor, shape, keep_dims=True)

    x = normalize(1 - convolve(ConvKernel.sobel_x, reference, [height, width, 1]))
    y = normalize(convolve(ConvKernel.sobel_y, reference, [height, width, 1]))

    z = 1 - tf.abs(normalize(tf.sqrt(x * x + y * y)) * 2 - 1) * .5 + .5

    return tf.stack([x[:,:,0], y[:,:,0], z[:,:,0]], 2)


def value_map(tensor, shape, keep_dims=False):
    """
    """

    channels = shape[-1]

    return normalize(tf.reduce_sum(tensor, len(shape) - 1, keep_dims=keep_dims))


def jpeg_decimate(tensor):
    """
    Needs more JPEG? Never again.

    :param Tensor tensor:
    :return: Tensor
    """

    jpegged = tf.image.convert_image_dtype(tensor, tf.uint8, saturate=True)

    data = tf.image.encode_jpeg(jpegged, quality=random.random() * 5 + 10)
    jpegged = tf.image.decode_jpeg(data)

    data = tf.image.encode_jpeg(jpegged, quality=random.random() * 5)
    jpegged = tf.image.decode_jpeg(data)

    return tf.image.convert_image_dtype(jpegged, tf.float32, saturate=True)


def blend(a, b, g):
    """
    Blend a and b values, using g as a multiplier.

    :param Tensor a:
    :param Tensor b:
    :param Tensor g:
    :return Tensor:
    """

    return (a * (1 - g) + b * g)


def blend_cosine(a, b, g):
    """
    """

    # This guy is great http://paulbourke.net/miscellaneous/interpolation/
    g2 = (1 - tf.cos(g * math.pi)) / 2

    return a * (1 - g2) + b * g2


def blend_cubic(a, b, c, d, g):
    """
    """

    # This guy is great http://paulbourke.net/miscellaneous/interpolation/
    g2 = g * g

    a0 = d - c - a + b
    a1 = a - b - a0
    a2 = c - a
    a3 = b

    return a0 * g * g2 + a1 * g2 + a2 * g + a3


def center_mask(center, edges, shape):
    """
    Blend two image tensors from the center to the edges. Not perfect.

    :param Tensor center:
    :param Tensor edges:
    :param list[int] shape:
    :return: Tensor
    """

    m = tf.cast(tf.reshape([
        [ 1, 1, 1, 1, 1, 1, 1 ],
        [ 1, .75, .5, .5, .5, .75, 1 ],
        [ 1, .5, 0, 0, 0, .5, 1 ],
        [ 1, .5, 0, 0, 0, .5, 1 ],
        [ 1, .5, 0, 0, 0, .5, 1 ],
        [ 1, .75, .5, .5, .5, .75, 1 ],
        [ 1, 1, 1, 1, 1, 1, 1 ],
        ], [7, 7, 1]), tf.float32)

    m = resample(m, shape)

    return blend_cosine(center, edges, m)


def row_index(shape):
    """
    Generate an X index for the given tensor.

    .. code-block:: python

      [
        [ 0, 1, 2, ... width-1 ],
        [ 0, 1, 2, ... width-1 ],
        ... (x height)
      ]

    :param list[int] shape:
    :return: Tensor
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

    :param list[int] shape:
    :return: Tensor
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