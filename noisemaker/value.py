"""Low-level value noise functions"""

from collections import defaultdict

import math
import random

import numpy as np
import tensorflow as tf

from noisemaker.constants import InterpolationType, ValueDistribution, ValueMask

import noisemaker.fastnoise as fastnoise
import noisemaker.masks as masks
import noisemaker.simplex as simplex


def set_seed(seed):
    """
    """

    if seed is not None:
        random.seed(seed)

        np.random.seed(seed)

        tf.random.set_seed(seed)

        simplex._seed = seed


def values(freq, shape, distrib=ValueDistribution.normal, corners=False, mask=None, mask_inverse=False, mask_static=False,
           spline_order=InterpolationType.bicubic, time=0.0, speed=1.0):
    """
    """

    if isinstance(freq, int):
        freq = freq_for_shape(freq, shape)

    initial_shape = freq + [shape[-1]]

    if distrib is None:
        distrib = ValueDistribution.normal

    if isinstance(distrib, int):
        distrib = ValueDistribution(distrib)

    elif isinstance(distrib, str):
        distrib = ValueDistribution[distrib]

    if isinstance(mask, int):
        mask = ValueMask(mask)

    elif isinstance(mask, str):
        mask = ValueMask[mask]

    if distrib == ValueDistribution.ones:
        tensor = tf.ones(initial_shape)

    elif distrib == ValueDistribution.mids:
        tensor = tf.ones(initial_shape) * .5

    elif distrib == ValueDistribution.zeros:
        tensor = tf.zeros(initial_shape)

    elif distrib == ValueDistribution.normal:
        tensor = tf.random.normal(initial_shape, mean=0.5, stddev=0.25)
        tensor = tf.math.minimum(tf.math.maximum(tensor, 0.0), 1.0)

    elif distrib == ValueDistribution.uniform:
        tensor = tf.random.uniform(initial_shape)

    elif distrib == ValueDistribution.exp:
        tensor = tf.cast(tf.stack(np.random.exponential(size=initial_shape)), tf.float32)

    elif distrib == ValueDistribution.laplace:
        tensor = tf.cast(tf.stack(np.random.laplace(size=initial_shape)), tf.float32)

    elif distrib == ValueDistribution.lognormal:
        tensor = tf.cast(tf.stack(np.random.lognormal(size=initial_shape)), tf.float32)

    elif distrib == ValueDistribution.column_index:
        tensor = tf.expand_dims(normalize(tf.cast(column_index(initial_shape), tf.float32)), -1) * tf.ones(initial_shape, tf.float32)

    elif distrib == ValueDistribution.row_index:
        tensor = tf.expand_dims(normalize(tf.cast(row_index(initial_shape), tf.float32)), -1) * tf.ones(initial_shape, tf.float32)

    elif ValueDistribution.is_simplex(distrib):
        tensor = simplex.simplex(initial_shape, time=time, speed=speed)

        if distrib == ValueDistribution.simplex_exp:
            tensor = tf.math.pow(tensor, 4)

        elif distrib == ValueDistribution.simplex_pow_inv_1:
            tensor = tf.math.pow(tensor, -1)

    elif ValueDistribution.is_fastnoise(distrib):

        tensor = fastnoise.fastnoise(shape, freq, seed=simplex._seed, time=time, speed=speed)

        if distrib == ValueDistribution.fastnoise_exp:
            tensor = tf.math.pow(tensor, 4)

    elif ValueDistribution.is_periodic(distrib):
        # we need to control the periodic function's visual speed (i.e. scale the time factor), but without breaking loops.
        # to accomplish this, we will use a scaled periodic uniform noise as the time value for periodic noise types.
        # since time values are per-pixel, this has the added bonus of animating different parts of the image at different
        # rates, rather than ping-ponging the entire image back and forth in lockstep. this creates a visual effect which
        # closely resembles higher-dimensional noise.

        # get a periodic uniform noise, and scale it to speed:
        scaled_time = periodic_value(time, tf.random.uniform(initial_shape)) * speed

        tensor = periodic_value(scaled_time, tf.random.uniform(initial_shape))

        if distrib == ValueDistribution.periodic_exp:
            tensor = tf.math.pow(tensor, 4)

        elif distrib == ValueDistribution.periodic_pow_inv_1:
            tensor = tf.math.pow(tensor, -1)

    else:
        raise ValueError("%s (%s) is not a ValueDistribution" % (distrib, type(distrib)))

    # Skip the below post-processing for fastnoise, since it's generated at a different size.
    if distrib not in (ValueDistribution.fastnoise, ValueDistribution.fastnoise_exp):
        if mask:
            atlas = masks.get_atlas(mask)

            glyph_shape = freq + [1]

            mask_values, _ = masks.mask_values(mask, glyph_shape, atlas=atlas, inverse=mask_inverse,
                                               time=0 if mask_static else time, speed=speed)

            if shape[2] == 2:
                tensor = tf.stack([tensor[:, :, 0], tf.stack(mask_values)[:, :, 0]], 2)

            elif shape[2] == 4:
                tensor = tf.stack([tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2], tf.stack(mask_values)[:, :, 0]], 2)

            else:
                tensor *= mask_values

        tensor = resample(tensor, shape, spline_order=spline_order)

        if (not corners and (freq[0] % 2) == 0) or (corners and (freq[0] % 2) == 1):
            tensor = offset(tensor, shape, x=int((shape[1] / freq[1]) * .5), y=int((shape[0] / freq[0]) * .5))

    if distrib not in (ValueDistribution.ones, ValueDistribution.mids, ValueDistribution.zeros):
        # I wish we didn't have to do this, but values out of the 0..1 range screw all kinds of things up
        tensor = normalize(tensor)

    return tensor


def periodic_value(time, value):
    """
    Coerce the received value to animate smoothly between time values 0 and 1, by applying a sine function and scaling the result.

    :param float time:
    :param float|Tensor value:
    """

    # h/t Etienne Jacob again
    # https://bleuje.github.io/tutorial2/
    return (tf.sin((time - value) * math.tau) + 1.0) * 0.5


def normalize(tensor, signed_range=False):
    """
    Squeeze the given Tensor into a range between 0 and 1.

    :param Tensor tensor: An image tensor.
    :param bool signed_range: Use a range between -1 and 1.
    :return: Tensor
    """

    floor = float(tf.reduce_min(tensor))
    if floor == math.inf or floor == -math.inf or floor == math.nan:  # Avoid GIGO
        raise ValueError(f"Input tensor contains {floor}, check caller for shenanigans")

    ceil = float(tf.reduce_max(tensor))
    if ceil == math.inf or ceil == -math.inf or ceil == math.nan:  # Avoid GIGO
        raise ValueError(f"Input tensor contains {ceil}, check caller for shenanigans")

    if floor == ceil:  # Avoid divide by zero
        raise ValueError(f"Input tensor min and max are each {floor}, check caller for shenanigans")

    delta = ceil - floor

    values = (tensor - floor) / delta

    if signed_range:
        values = values * 2.0 - 1.0

    return values


def _gather_scaled_offset(tensor, input_column_index, input_row_index, output_index):
    """ Helper function for resample(). Apply index offset to input tensor, return output_index values gathered post-offset. """

    return tf.gather_nd(tf.gather_nd(tensor, tf.stack([input_column_index, input_row_index], 2)), output_index)


def resample(tensor, shape, spline_order=3):
    """
    Resize an image tensor to the specified shape.

    :param Tensor tensor:
    :param list[int] shape:
    :param int spline_order: Spline point count. 0=Constant, 1=Linear, 2=Cosine, 3=Bicubic
    :return: Tensor
    """

    if isinstance(spline_order, int):
        spline_order = InterpolationType(spline_order)

    elif isinstance(spline_order, str):
        spline_order = InterpolationType[spline_order]

    input_shape = tf.shape(tensor)

    # Blown up row and column indices. These map into input tensor, producing a big blocky version.
    resized_row_index = tf.cast(row_index(shape), tf.float32) * (tf.cast(input_shape[1], tf.float32) / tf.cast(shape[1], tf.float32))   # 0, 1, 2, 3, -> 0, 0.5, 1, 1.5A
    resized_col_index = tf.cast(column_index(shape), tf.float32) * (tf.cast(input_shape[0], tf.float32) / tf.cast(shape[0], tf.float32))

    # Map to input indices as int
    resized_row_index_trunc = tf.floor(resized_row_index)
    resized_col_index_trunc = tf.floor(resized_col_index)
    resized_index_trunc = tf.cast(tf.stack([resized_col_index_trunc, resized_row_index_trunc], 2), tf.int32)

    # Resized original
    resized = defaultdict(dict)
    resized[1][1] = tf.gather_nd(tensor, resized_index_trunc)

    if spline_order == InterpolationType.constant:
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

    if spline_order == InterpolationType.linear:
        y1 = blend(resized[1][1], resized[1][2], resized_row_index_fract)
        y2 = blend(resized[2][1], resized[2][2], resized_row_index_fract)

        return blend(y1, y2, resized_col_index_fract)

    if spline_order == InterpolationType.cosine:
        y1 = blend_cosine(resized[1][1], resized[1][2], resized_row_index_fract)
        y2 = blend_cosine(resized[2][1], resized[2][2], resized_row_index_fract)

        return blend_cosine(y1, y2, resized_col_index_fract)

    if spline_order == InterpolationType.bicubic:
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


def proportional_downsample(tensor, shape, new_shape):
    """
    Given a new shape which is evenly divisible by the old shape, shrink the image by averaging pixel values.

    :param Tensor tensor:
    :param list[int] shape:
    :param list[int] new_shape:
    """

    kernel_shape = [int(max(shape[0] / new_shape[0], 1)), int(max(shape[1] / new_shape[1], 1)), shape[2], 1]

    kernel = tf.ones(kernel_shape)

    out = tf.nn.depthwise_conv2d([tensor], kernel, [1, kernel_shape[0], kernel_shape[1], 1], "VALID")[0] / (kernel_shape[0] * kernel_shape[1])

    return resample(out, new_shape)


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


def offset(tensor, shape, x=0, y=0):
    """
    """

    if x == 0 and y == 0:
        return tensor

    return tf.gather_nd(tensor, tf.stack([(column_index(shape) + y) % shape[0], (row_index(shape) + x) % shape[1]], 2))


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


def ridge(tensor):
    """
    Create a "ridge" at midpoint values. 1 - abs(n * 2 - 1)

    .. image:: images/crease.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor: An image tensor.
    :return: Tensor
    """

    return 1.0 - tf.abs(tensor * 2 - 1)


def simple_multires(freq, shape, octaves=1, spline_order=InterpolationType.bicubic, distrib=ValueDistribution.uniform, corners=False,
                    ridges=False, mask=None, mask_inverse=False, mask_static=False, time=0.0, speed=1.0):
    """Generate multi-octave value noise. Unlike generators.multires, this function is single-channel and does not apply effects."""

    if isinstance(freq, int):
        freq = freq_for_shape(freq, shape)

    tensor = tf.zeros(shape)

    for octave in range(1, octaves + 1):
        multiplier = 2 ** octave

        base_freq = [int(f * .5 * multiplier) for f in freq]

        if all(base_freq[i] > shape[i] for i in range(len(base_freq))):
            break

        layer = values(freq=base_freq, shape=shape, spline_order=spline_order, distrib=distrib, corners=corners,
                       mask=mask, mask_inverse=mask_inverse, mask_static=mask_static, time=time, speed=speed)

        if ridges:
            layer = ridge(layer)

        tensor += layer / multiplier

    return normalize(tensor)
