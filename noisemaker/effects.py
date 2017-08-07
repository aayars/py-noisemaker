from collections import defaultdict
from enum import Enum

import math
import random

import numpy as np
import tensorflow as tf


def post_process(tensor, shape, freq, ridges=False, spline_order=3, reflect_range=0.0, refract_range=0.0, reindex_range=0.0,
                 clut=None, clut_horizontal=False, clut_range=0.5,
                 with_worms=False, worms_behavior=None, worms_density=4.0, worms_duration=4.0, worms_stride=1.0, worms_stride_deviation=.05,
                 worms_bg=.5, worms_kink=1.0, with_sobel=False, sobel_func=0, with_normal_map=False, deriv=False, deriv_func=0, with_outline=False,
                 with_wormhole=False, wormhole_kink=2.5, wormhole_stride=.1,
                 with_voronoi=0, voronoi_density=.1, voronoi_nth=0, voronoi_func=0, voronoi_alpha=1.0, voronoi_refract=0.0,
                 posterize_levels=0, with_erosion_worms=False, warp_range=0.0, warp_octaves=3, vortex_range=0.0,
                **convolve_kwargs):
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
    :param bool with_worms: Do worms.
    :param WormBehavior worms_behavior:
    :param float worms_density: Worm density multiplier (larger == slower)
    :param float worms_duration: Iteration multiplier (larger == slower)
    :param float worms_stride: Mean travel distance per iteration
    :param float worms_stride_deviation: Per-worm travel distance deviation
    :param float worms_bg: Background color brightness for worms
    :param float worms_kink: Worm twistiness
    :param bool with_sobel: Sobel operator
    :param DistanceFunction|int sobel_func: Sobel distance function
    :param bool with_normal_map: Create a tangent-space normal map
    :param bool with_wormhole: Wormhole effect. What is this?
    :param float wormhole_kink: Wormhole kinkiness, if you're into that.
    :param float wormhole_stride: Wormhole thickness range
    :param VoronoiDiagramType|int with_voronoi: Voronoi diagram type (0=Off, 1=Range, 2=Color Range, 3=Indexed, 4=Color Map, 5=Blended, 6=Flow)
    :param float voronoi_density: Voronoi cell count multiplier
    :param int voronoi_nth: Voronoi Nth nearest
    :param DistanceFunction|int voronoi_func: Voronoi distance function
    :param float voronoi_alpha: Blend with original tensor (0.0 = Original, 1.0 = Voronoi)
    :param float voronoi_refract: Domain warp input tensor against Voronoi
    :param bool ridges: Ridged multifractal hint for Voronoi
    :param bool deriv: Derivative operator
    :param DistanceFunction|int deriv_func: Derivative distance function
    :param bool with_outline: Multiply tensor vs. sobel operator. Can supply sobel_func.
    :param float posterize_levels: Posterize levels
    :param bool with_erosion_worms: Erosion worms
    :param float warp_range: Orthogonal distortion gradient.
    :param float vortex_range: Vortex tiling amount
    :param int warp_octaves: Multi-res iteration count for warp

    :return: Tensor
    """

    tensor = normalize(tensor)

    if with_voronoi:
        _voronoi = singularity if voronoi_density == 0 else voronoi

        tensor = _voronoi(tensor, shape, diagram_type=with_voronoi, density=voronoi_density, nth=voronoi_nth, dist_func=voronoi_func,
                          alpha=voronoi_alpha, with_refract=voronoi_refract, ridges=ridges)

    if refract_range != 0:
        tensor = refract(tensor, shape, displacement=refract_range)

    if reflect_range != 0:
        tensor = refract(tensor, shape, displacement=reflect_range, from_derivative=True)

    if reindex_range != 0:
        tensor = reindex(tensor, shape, displacement=reindex_range)

    if clut:
        tensor = color_map(tensor, clut, shape, horizontal=clut_horizontal, displacement=clut_range)

    if warp_range:
        tensor = warp(tensor, shape, freq, displacement=warp_range, octaves=warp_octaves, spline_order=spline_order)

    # else:
        # tensor = normalize(tensor)

    if vortex_range:
        tensor = vortex(tensor, shape, displacement=vortex_range)

    if deriv:
        tensor = derivative(tensor, shape, deriv_func)

    if posterize_levels:
        tensor = posterize(tensor, posterize_levels)

    if with_worms:
        tensor = worms(tensor, shape, behavior=worms_behavior, density=worms_density, duration=worms_duration,
                       stride=worms_stride, stride_deviation=worms_stride_deviation, bg=worms_bg, kink=worms_kink)

    if with_wormhole:
        tensor = wormhole(tensor, shape, wormhole_kink, wormhole_stride)

    if with_erosion_worms:
        tensor = erode(tensor, shape)

    if with_sobel:
        tensor = sobel(tensor, shape, sobel_func)

    if with_normal_map:
        tensor = normal_map(tensor, shape)

    for kernel in ConvKernel:
        if convolve_kwargs.get(kernel.name):
            tensor =  convolve(kernel, tensor, shape)

    if with_outline:
        tensor = outline(tensor, shape, sobel_func=sobel_func)

    tensor = normalize(tensor)

    return tensor


class VoronoiDiagramType(Enum):
    """
    Specify the artistic rendering function used for Voronoi diagrams.
    """

    #: No Voronoi
    none = 0

    #: Normalized neighbor distances
    range = 1

    #: Normalized neighbor distances blended with input Tensor
    color_range = 2

    #: Indexed regions
    regions = 3

    #: Color-mapped regions
    color_regions = 4

    #: Colorized neighbor distances blended with color-mapped regions
    range_regions = 5

    #: Edgeless voronoi. Natural logarithm of reduced distance sums.
    flow = 6


class DistanceFunction(Enum):
    """
    Specify the distance function used in various operations, such as Voronoi cells, derivatives, and sobel operators.
    """

    euclidean = 0

    manhattan = 1

    chebyshev = 2


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
        [  0,   1,   1,   1,   1,   1,   1  ],
        [ -1,   0,   2,   2,   1,   1,   1  ],
        [ -1,  -2,   0,   2,   2,   1,   1  ],
        [ -1,  -2,  -4,   12,   2,   2,   1  ],
        [ -1,  -1,  -2,  -4,   0,   2,   1  ],
        [ -1,  -1,  -1,  -2,  -2,   0,   1  ],
        [ -1,  -1,  -1,  -1,  -1,  -1,   0  ]

        # [  0,  1,  1,  1, 0 ],
        # [ -1, -2,  4,  2, 1 ],
        # [ -1, -4,  2,  4, 1 ],
        # [ -1, -2, -4,  2, 1 ],
        # [  0, -1, -1, -1, 0 ]

        # [  0,  1,  1,  1, 0 ],
        # [ -1, -2,  4,  2, 1 ],
        # [ -1, -4,  2,  4, 1 ],
        # [ -1, -2, -4,  2, 1 ],
        # [  0, -1, -1, -1, 0 ]
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

    deriv_x = [
        [ 0, 0, 0 ],
        [ 0, -1, 1 ],
        [ 0, 0, 0 ]
    ]

    deriv_y = [
        [ 0, 0, 0 ],
        [ 0, -1, 0 ],
        [ 0, 1, 0 ]
    ]

    blur = [
        [ 1, 4, 6, 4, 1 ],
        [ 4, 16, 24, 16, 4 ],
        [ 6, 24, 36, 24, 6 ],
        [ 4, 16, 24, 16, 4 ],
        [ 1, 4, 6, 4, 1 ]
    ]


def _conform_kernel_to_tensor(kernel, tensor, shape):
    """ Re-shape a convolution kernel to match the given tensor's color dimensions. """

    l = len(kernel)

    channels = shape[-1]

    temp = np.repeat(kernel, channels)

    temp = tf.reshape(temp, (l, l, channels, 1))

    temp = tf.image.convert_image_dtype(temp, tf.float32)

    return temp


def convolve(kernel, tensor, shape, with_normalize=True):
    """
    Apply a convolution kernel to an image tensor.

    :param ConvKernel kernel: See ConvKernel enum
    :param Tensor tensor: An image tensor.
    :param list[int] shape:
    :param bool with_normalize: Normalize output (True)
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

    if with_normalize:
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

    floor = tf.reduce_min(tensor)
    ceil = tf.reduce_max(tensor)

    return (tensor - floor) / (ceil - floor)


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

    return 1.0 - tf.abs(tensor * 2 - 1)


def erode(tensor, shape):
    """
    WIP hydraulic erosion effect.
    """

    # This will never be as good as
    # https://www.dropbox.com/s/kqv8b3w7o8ucbyi/Beyer%20-%20implementation%20of%20a%20methode%20for%20hydraulic%20erosion.pdf?dl=0

    height, width, channels = shape

    count = int(math.sqrt(height * width) * 25)
    iterations = 50

    x = tf.random_uniform([count]) * (width - 1)
    y = tf.random_uniform([count]) * (height - 1)

    x_dir = tf.random_normal([count])
    y_dir = tf.random_normal([count])

    length = tf.sqrt(x_dir * x_dir + y_dir * y_dir)
    x_dir /= length
    y_dir /= length

    inertia = tf.random_normal([count], mean=0.75, stddev=0.25)
    # inertia = tf.ones([count])

    out = tf.zeros(shape)

    colors = tf.gather_nd(tensor, tf.cast(tf.stack([y, x], 1), tf.int32))

    values = value_map(tensor, shape, keep_dims=True)

    x_index = tf.cast(x, tf.int32)
    y_index = tf.cast(y, tf.int32)
    index = tf.stack([y_index, x_index], 1)
    starting_colors = tf.gather_nd(tensor, index)

    for i in range(iterations):
        x_index = tf.cast(x, tf.int32)
        y_index = tf.cast(y, tf.int32)
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

        length = distance(g_x, g_y, 1)

        x_dir = blend(x_dir, g_x / length, inertia)
        y_dir = blend(y_dir, g_y / length, inertia)

        # step
        x = (x + x_dir) % width
        y = (y + y_dir) % height

    return tf.maximum(tf.minimum(out, 1.0), 0.0)


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


def refract(tensor, shape, displacement=.5, reference_x=None, reference_y=None, warp_freq=None, spline_order=3, from_derivative=False):
    """
    Apply self-displacement along X and Y axes, based on each pixel value.

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
            reference_x = convolve(ConvKernel.deriv_x, tensor, shape, with_normalize=False)

        elif warp_freq:
            reference_x = resample(tf.random_uniform(warp_shape), shape, spline_order=spline_order)

        else:
            reference_x = tensor

    if reference_y is None:
        if from_derivative:
            reference_y = convolve(ConvKernel.deriv_y, tensor, shape, with_normalize=False)

        elif warp_freq:
            reference_y = resample(tf.random_uniform(warp_shape), shape, spline_order=spline_order)

        else:
            y0_index += int(height * .5)
            reference_y = tf.gather_nd(reference_x, tf.stack([y0_index % height, x0_index], 2))

    reference_x = value_map(reference_x, shape)
    reference_y = value_map(reference_y, shape)

    reference_x = reference_x * displacement * width
    reference_y = reference_y * displacement * height

    # Bilinear interpolation of corners
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


def worms(tensor, shape, behavior=0, density=4.0, duration=4.0, stride=1.0, stride_deviation=.05, bg=.5, kink=1.0, colors=None):
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
    :param Tensor colors: Optional starting colors, if not from `tensor`.
    :return: Tensor
    """

    height, width, channels = shape

    count = int(max(width, height) * density)

    worms_y = tf.random_uniform([count]) * (height - 1)
    worms_x = tf.random_uniform([count]) * (width - 1)
    worms_stride = tf.random_normal([count], mean=stride, stddev=stride_deviation)

    color_source = colors if colors is not None else tensor

    colors = tf.gather_nd(color_source, tf.cast(tf.stack([worms_y, worms_x], 1), tf.int32))

    if isinstance(behavior, int):
        behavior = WormBehavior(behavior)

    if behavior == WormBehavior.obedient:
        worms_rot = tf.zeros([count])

    elif behavior == WormBehavior.crosshatch:
        worms_rot = (tf.floor(tf.random_normal([count]) * 100) % 2) * 90

    elif behavior == WormBehavior.chaotic:
        worms_rot = tf.random_normal([count]) * 360.0

    else:
        worms_rot = tf.random_normal([count]) * .25 - .125

    index = value_map(tensor, shape) * 360.0 * math.radians(1) * kink

    iterations = int(math.sqrt(min(width, height)) * duration)

    out = color_source * bg

    scatter_shape = tf.shape(tensor)  # Might be different than `shape` due to clut

    # Make worms!
    for i in range(iterations):
        worm_positions = tf.cast(tf.stack([worms_y, worms_x], 1), tf.int32)

        exposure = 1 - abs(1 - i / (iterations - 1) * 2)  # Makes linear gradient [ 0 .. 1 .. 0 ]

        out += tf.scatter_nd(worm_positions, colors * exposure, scatter_shape)
        # out = tf.maximum(tf.scatter_nd(worm_positions, colors * exposure, scatter_shape), out)

        next_position = tf.gather_nd(index, worm_positions) + ( worms_rot - 45.0 )

        worms_y = (worms_y + tf.cos(next_position) * worms_stride) % height
        worms_x = (worms_x + tf.sin(next_position) * worms_stride) % width

    out = tf.image.convert_image_dtype(out, tf.float32, saturate=True)

    return tf.sqrt(normalize(out))


def wormhole(tensor, shape, kink, input_stride):
    """
    Apply per-pixel field flow. Non-iterative.

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

    out = tf.scatter_nd(offset_index(y, height, x, width), tensor * luminosity, tf.shape(tensor))

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


def derivative(tensor, shape, dist_func=0):
    """
    Extract a derivative from the given noise.

    .. image:: images/derived.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :param list[int] shape:
    :param DistanceFunction|int dist_func: Derivative distance function
    :return: Tensor
    """

    x = convolve(ConvKernel.deriv_x, tensor, shape, with_normalize=False)
    y = convolve(ConvKernel.deriv_y, tensor, shape, with_normalize=False)

    return normalize(distance(x, y, dist_func))


def sobel(tensor, shape, dist_func=0):
    """
    Apply a sobel operator.

    :param Tensor tensor:
    :param list[int] shape:
    :param DistanceFunction|int dist_func: Sobel distance function
    :return: Tensor
    """

    x = convolve(ConvKernel.sobel_x, tensor, shape, with_normalize=False)
    y = convolve(ConvKernel.sobel_y, tensor, shape, with_normalize=False)

    return tf.abs(normalize(distance(x, y, dist_func)) * 2 - 1)


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
    Create a grayscale value map from the given image Tensor by reducing the sum across channels.

    :param Tensor tensor:
    :param list[int] shape:
    :param bool keep_dims: If True, don't collapse the channel dimension.
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
    Blend a and b values with linear interpolation.

    :param Tensor a:
    :param Tensor b:
    :param Tensor g: Blending gradient a to b (0..1)
    :return Tensor:
    """

    return (a * (1 - g) + b * g)


def blend_cosine(a, b, g):
    """
    Blend a and b values with cosine interpolation.

    :param Tensor a:
    :param Tensor b:
    :param Tensor g: Blending gradient a to b (0..1)
    :return Tensor:
    """

    # This guy is great http://paulbourke.net/miscellaneous/interpolation/
    g2 = (1 - tf.cos(g * math.pi)) / 2

    return a * (1 - g2) + b * g2


def blend_cubic(a, b, c, d, g):
    """
    Blend b and c values with bi-cubic interpolation.

    :param Tensor a:
    :param Tensor b:
    :param Tensor c:
    :param Tensor d:
    :param Tensor g: Blending gradient b to c (0..1)
    :return Tensor:
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
    Blend two image tensors from the center to the edges.

    :param Tensor center:
    :param Tensor edges:
    :param list[int] shape:
    :return: Tensor
    """

    mask = singularity(None, shape, dist_func=DistanceFunction.chebyshev)

    return blend_cosine(center, edges, mask)


def voronoi(tensor, shape, diagram_type=1, density=.1, nth=0, dist_func=0, alpha=1.0, with_refract=0.0, xy=None, ridges=False):
    """
    Create a voronoi diagram, blending with input image Tensor color values.

    :param Tensor tensor:
    :param list[int] shape:
    :param VoronoiDiagramType|int diagram_type: Diagram type (0=Off, 1=Range, 2=Color Range, 3=Indexed, 4=Color Map, 5=Blended, 6=Flow)
    :param float density: Cell count multiplier (1.0 = min(height, width); larger is more costly)    `
    :param float nth: Plot Nth nearest neighbor, or -Nth farthest
    :param DistanceFunction|int dist_func: Voronoi distance function (0=Euclidean, 1=Manhattan, 2=Chebyshev)
    :param bool regions: Assign colors to control points (memory intensive)
    :param float alpha: Blend with original tensor (0.0 = Original, 1.0 = Voronoi)
    :param float with_refract: Domain warp input tensor against resulting voronoi
    :param (Tensor, Tensor, int) xy: Bring your own x, y, and point count (You shouldn't normally need this)
    :param float ridges: Adjust output colors to match ridged multifractal output (You shouldn't normally need this)
    :return: Tensor
    """

    original_shape = shape

    shape = [int(shape[0] * .5), int(shape[1] * .5), shape[2]]  # Gotta upsample later, this one devours memory.

    height, width, channels = shape

    if xy is None:
        point_count = int(min(width, height) * density)

        x = tf.random_uniform([point_count]) * width
        y = tf.random_uniform([point_count]) * height

    else:
        x, y, point_count = xy

    value_shape = [height, width, 1, 1]
    x_index = tf.cast(tf.reshape(row_index(shape), value_shape), tf.float32)
    y_index = tf.cast(tf.reshape(column_index(shape), value_shape), tf.float32)

    half_width = width * .5
    half_height = height * .5

    # Wrapping edges!
    x0_diff = (x_index - x - half_width) / width
    x1_diff = (x_index - x + half_width) / width
    y0_diff = (y_index - y - half_height) / height
    y1_diff = (y_index - y + half_height) / height

    x_diff = tf.minimum(tf.abs(x0_diff), tf.abs(x1_diff))
    y_diff = tf.minimum(tf.abs(y0_diff), tf.abs(y1_diff))

    # Not-wrapping edges!
    # x_diff = (x_index - x) / width
    # y_diff = (y_index - y) / height

    dist = distance(x_diff, y_diff, dist_func)

    if isinstance(diagram_type, VoronoiDiagramType):
       diagram_type = diagram_type.value

    ###
    if diagram_type not in (VoronoiDiagramType.flow.value, ):
        dist, indices = tf.nn.top_k(dist, k=point_count)
        index = int((nth + 1) * -1)

    ###
    if diagram_type in (VoronoiDiagramType.range.value, VoronoiDiagramType.color_range.value, VoronoiDiagramType.range_regions.value):
        range_slice = resample(tf.sqrt(normalize(dist[:,:,:,index])), original_shape)

    if diagram_type in (VoronoiDiagramType.regions.value, VoronoiDiagramType.color_regions.value, VoronoiDiagramType.range_regions.value):
        regions_slice = indices[:,:,:,index]

    ###
    if diagram_type == VoronoiDiagramType.range.value:
        range_out = range_slice

    if diagram_type in (VoronoiDiagramType.flow.value, ):
        range_out = resample(normalize(tf.reduce_sum(tf.log(dist), 3)), original_shape)

    if diagram_type in (VoronoiDiagramType.color_range.value, VoronoiDiagramType.range_regions.value):
        range_out = blend(tensor * range_slice, range_slice, range_slice)

    if diagram_type == VoronoiDiagramType.regions.value:
        regions_out = resample(tf.cast(regions_slice, tf.float32), original_shape)

    if diagram_type in (VoronoiDiagramType.color_regions.value, VoronoiDiagramType.range_regions.value):
        colors = tf.gather_nd(tensor, tf.cast(tf.stack([y * 2, x * 2], 1), tf.int32))

        if ridges:
            colors = tf.abs(colors * 2 - 1)

        regions_out = resample(tf.reshape(tf.gather(colors, regions_slice), shape), original_shape)

    ###
    if diagram_type == VoronoiDiagramType.range_regions.value:
        out = blend(regions_out, range_out, tf.square(range_out))

    elif diagram_type in (VoronoiDiagramType.range.value, VoronoiDiagramType.color_range.value, VoronoiDiagramType.flow.value):
        out = range_out

    elif diagram_type in (VoronoiDiagramType.regions.value, VoronoiDiagramType.color_regions.value):
        out = regions_out

    if with_refract != 0.0:
        # ref_shape = [original_shape[0], original_shape[1], 1]
        # out = tf.square(out)
        # ref_x = convolve(ConvKernel.sobel_x, out, ref_shape)
        # ref_y = convolve(ConvKernel.sobel_y, out, ref_shape)

        # out = distance(ref_x, ref_y, 0)
        # out = refract(tensor, original_shape, displacement=with_refract, reference_x=out)
        # out = refract(tensor, original_shape, displacement=with_refract, reference_x=ref_x, reference_y=ref_y)

        out = refract(tensor, original_shape, displacement=with_refract, reference_x=out)

    if tensor is not None:
        out = blend(tensor, out, alpha)

    return out


def distance(a, b, func):
    """
    Compute the distance from a to b using the specified function.

    :param Tensor a:
    :param Tensor b:
    :param DistanceFunction|int dist_func: Distance function (0=Euclidean, 1=Manhattan, 2=Chebyshev)
    :return: Tensor
    """

    if isinstance(func, DistanceFunction):
       func = func.value

    if func == DistanceFunction.euclidean.value:
        dist = tf.sqrt(a * a + b * b)

    elif func == DistanceFunction.manhattan.value:
        dist = tf.abs(a) + tf.abs(b)

    elif func == DistanceFunction.chebyshev.value:
        dist = tf.maximum(tf.abs(a), tf.abs(b))

    else:
        raise ValueError("{0} isn't a distance function.".format(func))

    return dist


def posterize(tensor, levels):
    """
    Reduce the number of color levels per channel.

    :param Tensor tensor:
    :param int levels:
    :return: Tensor
    """

    tensor *= levels

    tensor = tf.floor(tensor)

    tensor /= levels

    return tensor


def _inner_tile(tensor, shape, freq):
    """
    WIP for pop art filter.
    """

    if isinstance(freq, int):
        freq = freq_for_shape(freq, shape)

    small_shape = [int(shape[0] / freq[0]), int(shape[1] / freq[1]), shape[2]]

    y_index = tf.tile(column_index(small_shape) * freq[0], [freq[0], freq[0]])
    x_index = tf.tile(row_index(small_shape) * freq[1], [freq[0], freq[0]])

    tiled = tf.gather_nd(tensor, tf.stack([y_index, x_index], 2))

    tiled = resample(tiled, shape, spline_order=1)

    return tiled


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


def warp(tensor, shape, freq, octaves=5, displacement=1, spline_order=3):
    for octave in range(1, octaves + 1):
        multiplier = 2 ** octave

        base_freq = [int(f * .5 * multiplier) for f in freq]

        if base_freq[0] >= shape[0] or base_freq[1] >= shape[1]:
            break

        tensor = refract(tensor, shape, displacement=displacement / multiplier, warp_freq=base_freq, spline_order=spline_order)

    return tensor


def outline(tensor, shape, sobel_func=0):
    """
    Superimpose sobel operator results (cartoon edges)

    :param Tensor tensor:
    :param list[int] shape:
    :param DistanceFunction|int sobel_func: Sobel distance function
    """

    height, width, channels = shape

    value_shape = [height, width, 1]

    values = value_map(tensor, shape, keep_dims=True)

    edges = sobel(values, value_shape, dist_func=sobel_func)

    return edges * tensor


def singularity(tensor, shape, diagram_type=1, **kwargs):
    """
    Return the range diagram for a single voronoi point, approximately centered.

    :param list[int] shape:
    :param DistanceFunction|int dist_func:
    :param VoronoiDiagramType|int diagram_type:

    Additional kwargs will be sent to the `voronoi` function.
    """

    x = tf.stack([0.0])
    y = tf.stack([0.0])

    point_count = 1

    return convolve(ConvKernel.blur, voronoi(tensor, shape, diagram_type=diagram_type, xy=(x, y, point_count), **kwargs) * tf.ones(shape), shape)


def vortex(tensor, shape, displacement=64.0):
    """
    Vortex tiling effect

    :param Tensor tensor:
    :param list[int] shape:
    :param float displacement:
    """

    displacement_map = singularity(None, shape)

    x = convolve(ConvKernel.deriv_x, displacement_map, shape, with_normalize=False)
    y = convolve(ConvKernel.deriv_y, displacement_map, shape, with_normalize=False)

    warped = refract(tensor, shape, displacement=displacement, reference_x=x, reference_y=y)

    return center_mask(warped, convolve(ConvKernel.blur, tensor, shape) * .25, shape)