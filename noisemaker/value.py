"""Low-level value noise functions"""

from collections import defaultdict

import math
import random

import numpy as np
import tensorflow as tf

from noisemaker.constants import (
    DistanceMetric,
    InterpolationType,
    PointDistribution,
    ValueDistribution,
    ValueMask,
    VoronoiDiagramType,
)
from noisemaker.effects_registry import effect
from noisemaker.points import point_cloud

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

    elif ValueDistribution.is_center_distance(distrib):
        metric = DistanceMetric[distrib.name.replace("center_", "")]

        # make sure speed doesn't break looping
        if speed > 0:
            rounded_speed = math.floor(1 + speed)
        else:
            rounded_speed = math.ceil(-1 + speed)

        tensor = normalized_sine(singularity(None, shape, dist_metric=metric) * math.tau * max(freq[0], freq[1])
                                 - math.tau * time * rounded_speed) * tf.ones(shape)

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

    if mask:
        atlas = masks.get_atlas(mask)

        glyph_shape = freq + [1]

        mask_values, _ = masks.mask_values(mask, glyph_shape, atlas=atlas, inverse=mask_inverse,
                                           time=0 if mask_static else time, speed=speed)

        # These noise types are generated at full size
        if ValueDistribution.is_fastnoise(distrib) or ValueDistribution.is_center_distance(distrib):
            mask_values = resample(mask_values, shape, spline_order=spline_order)
            mask_values = pin_corners(mask_values, shape, freq, corners)

        if shape[2] == 2:
            tensor = tf.stack([tensor[:, :, 0], tf.stack(mask_values)[:, :, 0]], 2)

        elif shape[2] == 4:
            tensor = tf.stack([tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2], tf.stack(mask_values)[:, :, 0]], 2)

        else:
            tensor *= mask_values

    if not ValueDistribution.is_fastnoise(distrib) and not ValueDistribution.is_center_distance(distrib):
        tensor = resample(tensor, shape, spline_order=spline_order)

        tensor = pin_corners(tensor, shape, freq, corners)

    if distrib not in (ValueDistribution.ones, ValueDistribution.mids, ValueDistribution.zeros):
        # I wish we didn't have to do this, but values out of the 0..1 range screw all kinds of things up
        tensor = normalize(tensor)

    return tensor


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


@effect()
def voronoi(tensor, shape, diagram_type=VoronoiDiagramType.range, nth=0,
            dist_metric=DistanceMetric.euclidean, alpha=1.0, with_refract=0.0, inverse=False,
            xy=None, ridges_hint=False, refract_y_from_offset=True, time=0.0, speed=1.0,
            point_freq=3, point_generations=1, point_distrib=PointDistribution.random, point_drift=0.0, point_corners=False):
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
        if point_freq == 1:
            x, y = point_cloud(point_freq, PointDistribution.square, shape)

        else:
            x, y = point_cloud(point_freq, distrib=point_distrib, shape=shape, corners=point_corners, generations=point_generations,
                               drift=point_drift, time=time, speed=speed)

        point_count = len(x)

    else:
        if len(xy) == 2:
            x, y = xy
            point_count = len(x)

        else:
            x, y, point_count = xy

        x = tf.cast(tf.stack(x), tf.float32) / 2.0
        y = tf.cast(tf.stack(y), tf.float32) / 2.0

    vshape = value_shape(shape)

    x_index = tf.cast(tf.reshape(row_index(shape), vshape), tf.float32)
    y_index = tf.cast(tf.reshape(column_index(shape), vshape), tf.float32)

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
        index = min(nth + 1, point_count - 1) * -1

    ###

    # Seamless alg offset pixels by half image size. Move results slice back to starting points with `offset`:
    offset_kwargs = {
        'x': 0.0 if is_triangular else half_width,
        'y': 0.0 if is_triangular else half_height,
    }

    if diagram_type in (VoronoiDiagramType.range, VoronoiDiagramType.color_range, VoronoiDiagramType.range_regions):
        range_slice = normalize(dist[:, :, index])
        range_slice = tf.expand_dims(tf.sqrt(range_slice), -1)
        range_slice = resample(offset(range_slice, shape, **offset_kwargs), original_shape)

        if inverse:
            range_slice = 1.0 - range_slice

    if diagram_type in (VoronoiDiagramType.regions, VoronoiDiagramType.color_regions, VoronoiDiagramType.range_regions):
        regions_slice = offset(indices[:, :, index], shape, **offset_kwargs)

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
            range_out = normalize(tf.math.reduce_mean(1.0 - (1.0 - normalize(dist * colors)), 2))

        else:  # flow
            # This is dicey as hell. Try to get range_out into a reasonable range.
            # Difficulty level: Without using normalize()
            range_out = (tf.math.reduce_mean(dist, 2) + 1.75) / 1.45
            # print(tf.reduce_min(range_out))
            # print(tf.reduce_max(range_out))

        range_out = resample(offset(range_out, shape, **offset_kwargs), original_shape)

        if inverse:
            range_out = 1.0 - range_out

    if diagram_type in (VoronoiDiagramType.color_range, VoronoiDiagramType.range_regions):
        # range_out = regions_out * range_slice
        range_out = blend(tensor * range_slice, range_slice, range_slice)

    if diagram_type == VoronoiDiagramType.regions:
        regions_out = resample(tf.cast(regions_slice, tf.float32), original_shape, spline_order=InterpolationType.constant)

    if diagram_type in (VoronoiDiagramType.color_regions, VoronoiDiagramType.range_regions):
        colors = tf.gather_nd(tensor, tf.cast(tf.stack([y * 2, x * 2], 1), tf.int32))

        if ridges_hint:
            colors = tf.abs(colors * 2 - 1)

        spline_order = 0 if diagram_type == VoronoiDiagramType.color_regions else 3

        regions_out = resample(tf.reshape(tf.gather(colors, regions_slice), shape), original_shape, spline_order=spline_order)

    ###
    if diagram_type == VoronoiDiagramType.range_regions:
        out = blend(regions_out, range_out, tf.square(range_out))

    elif diagram_type in [VoronoiDiagramType.range, VoronoiDiagramType.color_range] + VoronoiDiagramType.flow_members():
        out = range_out

    elif diagram_type in (VoronoiDiagramType.regions, VoronoiDiagramType.color_regions):
        out = regions_out

    else:
        raise Exception(f"Not sure what to do with diagram type {diagram_type}")

    if diagram_type == VoronoiDiagramType.regions:
        out = tf.expand_dims(out, -1) / point_count

    if with_refract != 0.0:
        out = refract(tensor, original_shape, displacement=with_refract, reference_x=out,
                      y_from_offset=refract_y_from_offset)

    if tensor is not None:
        out = blend(tensor, out, alpha)

    return out


def periodic_value(time, value):
    """
    Coerce the received value to animate smoothly between time values 0 and 1, by applying a sine function and scaling the result.

    :param float time:
    :param float|Tensor value:
    """

    # h/t Etienne Jacob again
    # https://bleuje.github.io/tutorial2/
    return normalized_sine((time - value) * math.tau)


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
    resized_row_index = tf.cast(row_index(shape), tf.float32) \
        * (tf.cast(input_shape[1], tf.float32) / tf.cast(shape[1], tf.float32))   # 0, 1, 2, 3, -> 0, 0.5, 1, 1.5A

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
    vshape = value_shape(shape)
    resized_row_index_fract = tf.reshape(resized_row_index - resized_row_index_trunc, vshape)  # 0, 0.5, 1, 1.5 -> 0, .5, 0, .5
    resized_col_index_fract = tf.reshape(resized_col_index - resized_col_index_trunc, vshape)

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


def value_shape(shape):
    """
    """

    return [shape[0], shape[1], 1]


def normalized_sine(value):
    """
    """

    return (tf.sin(value) + 1.0) * 0.5


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
def convolve(tensor, shape, kernel=None, with_normalize=True, alpha=1.0, time=0.0, speed=1.0):
    """
    Apply a convolution kernel to an image tensor.

    .. code-block:: python

       image = convolve(image, shape, ValueMask.conv2d_shadow)

    :param Tensor tensor: An image tensor.
    :param list[int] shape:
    :param ValueMask kernel: See conv2d_* members in ValueMask enum
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

    out = tf.image.resize_with_crop_or_pad(out, height, width)

    if with_normalize:
        out = normalize(out)

    if kernel == ValueMask.conv2d_edges:
        out = tf.abs(out - .5) * 2

    if alpha == 1.0:
        return out

    return blend(tensor, out, alpha)


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
    :param int spline_order: Interpolation for warp effect only. 0=Constant, 1=Linear, 2=Cosine, 3=Bicubic
    :param bool from_derivative: If True, generate X and Y offsets from noise derivatives.
    :param bool signed_range: Scale displacement values from -1..1 instead of 0..1
    :param bool y_from_offset: If True, derive Y offsets from offsetting the image
    :return: Tensor
    """

    height, width, channels = shape

    x0_index = row_index(shape)
    y0_index = column_index(shape)

    warp_shape = None

    if warp_freq:
        warp_shape = [height, width, 1]

    if reference_x is None:
        if from_derivative:
            reference_x = convolve(kernel=ValueMask.conv2d_deriv_x, tensor=tensor, shape=shape, with_normalize=False)

        elif warp_freq:
            reference_x = values(freq=warp_freq, shape=warp_shape, distrib=ValueDistribution.periodic_uniform,
                                 time=time, speed=speed, spline_order=spline_order)

        else:
            reference_x = tensor

    if reference_y is None:
        if from_derivative:
            reference_y = convolve(kernel=ValueMask.conv2d_deriv_y, tensor=tensor, shape=shape, with_normalize=False)

        elif warp_freq:
            reference_y = values(freq=warp_freq, shape=warp_shape, distrib=ValueDistribution.periodic_uniform,
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
    x_offsets = value_map(reference_x, shape, signed_range=quad_directional, with_normalize=False) * displacement * tf.cast(width, tf.float32)
    y_offsets = value_map(reference_y, shape, signed_range=quad_directional, with_normalize=False) * displacement * tf.cast(height, tf.float32)
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
        tensor = normalize(tensor, signed_range=signed_range)

    elif signed_range:
        tensor = tensor * 2.0 - 1.0

    return tensor


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


def pin_corners(tensor, shape, freq, corners):
    """Pin values to image corners, or align with image center, as per the given "corners" arg."""

    if (not corners and (freq[0] % 2) == 0) or (corners and (freq[0] % 2) == 1):
        tensor = offset(tensor, shape, x=int((shape[1] / freq[1]) * .5), y=int((shape[0] / freq[0]) * .5))

    return tensor
