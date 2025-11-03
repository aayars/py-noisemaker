"""Low-level value noise functions"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import tensorflow as tf

import noisemaker.masks as masks
import noisemaker.oklab as oklab
import noisemaker.rng as rng
import noisemaker.simplex as simplex
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


def set_seed(seed: int | None) -> None:
    """
    Set the random seed for noise generation.

    Args:
        seed: Random seed value, or None to skip seeding

    Returns:
        Processed tensor
    """

    if seed is not None:
        rng.set_seed(seed)
        simplex._seed = seed


def value_noise(count: int, freq: int = 8) -> tf.Tensor:
    """
    Generate 1D value noise samples.

    Args:
        count: Number of samples to generate
        freq: Frequency for noise generation

    Returns:
        Processed tensor
    """

    lattice = [rng.random() for _ in range(freq + 1)]
    out = []
    for i in range(count):
        x = i / count * freq
        xi = int(x)
        xf = x - xi
        t = xf * xf * (3 - 2 * xf)
        out.append(lattice[xi] * (1 - t) + lattice[xi + 1] * t)

    return tf.constant(out, dtype=tf.float32)


def values(
    freq: int | list[int],
    shape: list[int],
    distrib: ValueDistribution | None = ValueDistribution.simplex,
    corners: bool = False,
    mask: ValueMask | None = None,
    mask_inverse: bool = False,
    mask_static: bool = False,
    spline_order: int | InterpolationType = InterpolationType.bicubic,
    time: float = 0.0,
    speed: float = 1.0,
) -> tf.Tensor:
    """
    Generate a tensor of noise values with specified distribution.

    Args:
        freq: Frequency for noise generation
        shape: Shape of the tensor [height, width, channels]
        distrib: Value distribution method
        corners: Include corner values
        mask: Value mask to apply
        mask_inverse: Invert the mask
        mask_static: Use static mask values
        spline_order: Interpolation type for resampling
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Processed tensor
    """

    if isinstance(spline_order, int):
        spline_order = InterpolationType(spline_order)

    freq_list: list[int]
    if isinstance(freq, int):
        freq_list = freq_for_shape(freq, shape)
    else:
        freq_list = freq

    initial_shape: list[int] = freq_list + [shape[-1]]

    if distrib is None:
        distrib = ValueDistribution.simplex

    distrib = coerce_enum(distrib, ValueDistribution)

    mask = coerce_enum(mask, ValueMask)

    if distrib == ValueDistribution.ones:
        tensor = tf.ones(initial_shape, dtype=tf.float32)

    elif distrib == ValueDistribution.mids:
        tensor = tf.ones(initial_shape, dtype=tf.float32) * 0.5

    elif distrib == ValueDistribution.zeros:
        tensor = tf.zeros(initial_shape, dtype=tf.float32)

    elif distrib == ValueDistribution.column_index:
        tensor = tf.expand_dims(normalize(tf.cast(column_index(initial_shape), tf.float32)), -1) * tf.ones(initial_shape, dtype=tf.float32)

    elif distrib == ValueDistribution.row_index:
        tensor = tf.expand_dims(normalize(tf.cast(row_index(initial_shape), tf.float32)), -1) * tf.ones(initial_shape, dtype=tf.float32)

    elif ValueDistribution.is_center_distance(distrib):
        sdf_sides = None

        if distrib == ValueDistribution.center_circle:
            metric = DistanceMetric.euclidean
        elif distrib == ValueDistribution.center_triangle:
            metric = DistanceMetric.triangular
        elif distrib == ValueDistribution.center_diamond:
            metric = DistanceMetric.manhattan
        elif distrib == ValueDistribution.center_square:
            metric = DistanceMetric.chebyshev
        elif distrib == ValueDistribution.center_pentagon:
            metric = DistanceMetric.sdf
            sdf_sides = 5
        elif distrib == ValueDistribution.center_hexagon:
            metric = DistanceMetric.hexagram
        elif distrib == ValueDistribution.center_heptagon:
            metric = DistanceMetric.sdf
            sdf_sides = 7
        elif distrib == ValueDistribution.center_octagon:
            metric = DistanceMetric.octagram
        elif distrib == ValueDistribution.center_nonagon:
            metric = DistanceMetric.sdf
            sdf_sides = 9
        elif distrib == ValueDistribution.center_decagon:
            metric = DistanceMetric.sdf
            sdf_sides = 10
        elif distrib == ValueDistribution.center_hendecagon:
            metric = DistanceMetric.sdf
            sdf_sides = 11
        elif distrib == ValueDistribution.center_dodecagon:
            metric = DistanceMetric.sdf
            sdf_sides = 12

        # make sure speed doesn't break looping
        if speed > 0:
            rounded_speed = math.floor(1 + speed)
        else:
            rounded_speed = math.ceil(-1 + speed)

        tensor = normalized_sine(
            singularity(None, shape, dist_metric=metric, sdf_sides=sdf_sides) * math.tau * max(freq_list[0], freq_list[1]) - math.tau * time * rounded_speed
        ) * tf.ones(shape, dtype=tf.float32)

    elif ValueDistribution.is_noise(distrib):
        base_seed = simplex.get_seed()
        value_noise = tf.cast(
            simplex.simplex(initial_shape, time=time, seed=base_seed, speed=speed),
            tf.float32,
        )

        if speed == 0:
            tensor = value_noise
        else:
            time_seed = (base_seed + 0x9E3779B1) & 0xFFFFFFFF
            time_noise = tf.cast(
                simplex.simplex(initial_shape, time=0.0, seed=time_seed, speed=1),
                tf.float32,
            )
            scaled_time = periodic_value(time, time_noise) * speed
            tensor = periodic_value(scaled_time, value_noise)

        if distrib == ValueDistribution.exp:
            tensor = tf.math.pow(tensor, 4)

    else:
        raise ValueError("%s (%s) is not a ValueDistribution" % (distrib, type(distrib)))

    if mask:
        atlas = masks.get_atlas(mask)

        glyph_shape = freq_list + [1]

        mask_values, _ = masks.mask_values(mask, glyph_shape, atlas=atlas, inverse=mask_inverse, time=0 if mask_static else time, speed=speed)

        # These noise types are generated at full size, resize and pin just the mask.
        if ValueDistribution.is_native_size(distrib):
            mask_values = resample(mask_values, shape, spline_order=spline_order)
            mask_values = pin_corners(mask_values, shape, freq_list, corners)

        if shape[2] == 2:
            tensor = tf.stack([tensor[:, :, 0], tf.stack(mask_values)[:, :, 0]], 2)

        elif shape[2] == 4:
            tensor = tf.stack([tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2], tf.stack(mask_values)[:, :, 0]], 2)

        else:
            tensor *= mask_values

    if not ValueDistribution.is_native_size(distrib):
        tensor = resample(tensor, shape, spline_order=spline_order)
        tensor = pin_corners(tensor, shape, freq_list, corners)

    if distrib not in (ValueDistribution.ones, ValueDistribution.mids, ValueDistribution.zeros):
        # I wish we didn't have to do this, but values out of the 0..1 range screw all kinds of things up
        tensor = normalize(tensor)

    return tensor


def distance(a: tf.Tensor, b: tf.Tensor, metric: int | DistanceMetric = DistanceMetric.euclidean, sdf_sides: int = 5) -> tf.Tensor:
    """
    Compute the distance from a to b, using the specified metric.

    Args:
        a: First value for blending/distance
        b: Second value for blending/distance
        metric: Distance metric to use
        sdf_sides: SDF polygon sides for distance metric

    Returns:
        Processed tensor
    """

    metric = coerce_enum(metric, DistanceMetric)

    if metric == DistanceMetric.euclidean:
        dist = tf.sqrt(a * a + b * b)

    elif metric == DistanceMetric.manhattan:
        dist = tf.abs(a) + tf.abs(b)

    elif metric == DistanceMetric.chebyshev:
        dist = tf.maximum(tf.abs(a), tf.abs(b))

    elif metric == DistanceMetric.octagram:
        dist = tf.maximum((tf.abs(a) + tf.abs(b)) / math.sqrt(2), tf.maximum(tf.abs(a), tf.abs(b)))

    elif metric == DistanceMetric.triangular:
        dist = tf.maximum(tf.abs(a) - b * 0.5, b)

    elif metric == DistanceMetric.hexagram:
        dist = tf.maximum(tf.maximum(tf.abs(a) - b * 0.5, b), tf.maximum(tf.abs(a) - b * -0.5, b * -1))

    elif metric == DistanceMetric.sdf:
        # https://thebookofshaders.com/07/
        arctan = tf.math.atan2(a, -b) + math.pi
        r = math.tau / sdf_sides

        dist = tf.math.cos(tf.math.floor(0.5 + arctan / r) * r - arctan) * tf.sqrt(a * a + b * b)

    else:
        raise ValueError(f"{metric} isn't a distance metric.")

    return dist


@effect()
def voronoi(
    tensor: tf.Tensor,
    shape: list[int],
    diagram_type: VoronoiDiagramType = VoronoiDiagramType.range,
    nth: int = 0,
    dist_metric: int | DistanceMetric = DistanceMetric.euclidean,
    sdf_sides: int = 3,
    alpha: float = 1.0,
    with_refract: float = 0.0,
    inverse: bool = False,
    xy: tf.Tensor | None = None,
    ridges_hint: bool = False,
    refract_y_from_offset: bool = True,
    time: float = 0.0,
    speed: float = 1.0,
    point_freq: int = 3,
    point_generations: int = 1,
    point_distrib: PointDistribution = PointDistribution.random,
    point_drift: float = 0.0,
    point_corners: bool = False,
    downsample: bool = True,
) -> tf.Tensor:
    """
    Create a voronoi diagram, blending with input image Tensor color values.


    .. noisemaker-live::
       :effect: voronoi
       :input: basic
       :seed: 42
       :width: 512
       :height: 256
       :lazy:

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        diagram_type: Type of Voronoi diagram
        nth: Use nth closest point
        dist_metric: Distance metric to use
        sdf_sides: SDF polygon sides for distance metric
        alpha: Blending alpha value (0.0-1.0)
        with_refract: Apply refraction amount
        inverse: Invert the effect
        xy: Optional XY coordinates for point cloud
        ridges_hint: Apply ridge transformation hint
        refract_y_from_offset: Use Y offset for refraction
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier
        point_freq: Point frequency for Voronoi
        point_generations: Point generation count
        point_distrib: Point distribution method
        point_drift: Point drift amount
        point_corners: Include corner points
        downsample: Downsample the result

    Returns:
        Processed tensor
    """

    diagram_type = coerce_enum(diagram_type, VoronoiDiagramType)

    dist_metric = coerce_enum(dist_metric, DistanceMetric)

    original_shape = shape

    if downsample:  # To save memory
        shape = [int(shape[0] * 0.5), int(shape[1] * 0.5), shape[2]]

    height, width, channels = shape

    if xy is None:
        if point_freq == 1:
            result = point_cloud(point_freq, PointDistribution.square, shape)
            if result is None:
                raise ValueError("point_cloud returned None")
            x, y = result
            point_count = len(x)

        else:
            result = point_cloud(
                point_freq, distrib=point_distrib, shape=shape, corners=point_corners, generations=point_generations, drift=point_drift, time=time, speed=speed
            )
            if result is None:
                raise ValueError("point_cloud returned None")
            x0, y0 = result
            point_count = len(x0)

            x = []
            y = []
            for i in range(point_count):
                x.append(blend_cosine(x0[i], x0[(i + 1) % point_count], time))
                y.append(blend_cosine(y0[i], y0[(i + 1) % point_count], time))

    else:
        if len(xy) == 2:
            x, y = xy
            point_count = len(x)

        else:
            x, y, point_count = xy

        x = tf.cast(tf.stack(x), tf.float32)
        y = tf.cast(tf.stack(y), tf.float32)

        if downsample:
            x /= 2.0
            y /= 2.0

    vshape = value_shape(shape)

    x_index = tf.cast(tf.reshape(row_index(shape), vshape), tf.float32)
    y_index = tf.cast(tf.reshape(column_index(shape), vshape), tf.float32)

    is_triangular = dist_metric in (
        DistanceMetric.triangular,
        DistanceMetric.hexagram,
        DistanceMetric.sdf,
    )

    if diagram_type in VoronoiDiagramType.flow_members():
        # If we're using flow with a perfectly tiled grid, it just disappears. Perturbing the points seems to prevent this from happening.
        x += rng.normal(tf.shape(x), stddev=0.0001, dtype=tf.float32)
        y += rng.normal(tf.shape(y), stddev=0.0001, dtype=tf.float32)

    if is_triangular:
        # Keep it visually flipped "horizontal"-side-up
        y_sign = -1.0 if inverse else 1.0

        dist = distance((x_index - x) / width, (y_index - y) * y_sign / height, dist_metric, sdf_sides=sdf_sides)

    else:
        half_width = int(width * 0.5)
        half_height = int(height * 0.5)

        # Wrapping edges! Nearest neighbors might be actually be "wrapped around", on the opposite side of the image.
        # Determine which direction is closer, and use the minimum.

        # Subtracting the list of points from the index results in a new shape
        # [y, x, value] - [point_count] -> [y, x, value, point_count]
        x0_diff = x_index - x - half_width
        x1_diff = x_index - x + half_width
        y0_diff = y_index - y - half_height
        y1_diff = y_index - y + half_height

        #
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
        "x": 0.0 if is_triangular else half_width,
        "y": 0.0 if is_triangular else half_height,
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

            # Trying to avoid normalize() here, since it tends to make animations twitchy.
            range_out = tf.math.reduce_mean(1.0 - (1.0 - (dist * colors)), 2)

        else:  # flow
            # Trying to avoid normalize() here, since it tends to make animations twitchy.
            range_out = (tf.math.reduce_mean(dist, 2) + 1.75) / 1.45

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
        out = refract(tensor, original_shape, displacement=with_refract, reference_x=out, y_from_offset=refract_y_from_offset)

    if tensor is not None:
        out = blend(tensor, out, alpha)

    return out


def periodic_value(time: float, value: float) -> tf.Tensor:
    """
    Coerce the received value to animate smoothly between time values 0 and 1, by applying a sine function and scaling the result.

    Args:
        time: Time value for animation (0.0-1.0)
        value: Input value

    Returns:
        Processed tensor
    """

    # h/t Etienne Jacob again
    # https://bleuje.github.io/tutorial2/
    return normalized_sine((time - value) * math.tau)


def normalize(tensor: tf.Tensor, signed_range: bool = False) -> tf.Tensor:
    """
    Squeeze the given Tensor into a range between 0 and 1.

    Args:
        tensor: Input tensor to process
        signed_range: Use signed range (-1 to 1)

    Returns:
        Normalized tensor
    """

    floor = float(tf.reduce_min(tensor))
    if floor == math.inf or floor == -math.inf or floor == math.nan:  # Avoid GIGO
        raise ValueError(f"Input tensor contains {floor}, check caller for shenanigans")

    ceil = float(tf.reduce_max(tensor))
    if ceil == math.inf or ceil == -math.inf or ceil == math.nan:  # Avoid GIGO
        raise ValueError(f"Input tensor contains {ceil}, check caller for shenanigans")

    if floor == ceil:  # Avoid divide by zero
        return tensor

    delta = ceil - floor

    values = (tensor - floor) / delta

    if signed_range:
        values = values * 2.0 - 1.0

    return values


def _gather_scaled_offset(tensor: tf.Tensor, input_column_index: tf.Tensor, input_row_index: tf.Tensor, output_index: tf.Tensor) -> tf.Tensor:
    """
    Helper function for resample(). Apply index offset to input tensor, return output_index values gathered post-offset.

    Args:
        tensor: Input tensor to process
        input_column_index: Column indices for input
        input_row_index: Row indices for input
        output_index: Output index values

    Returns:
        Processed value
    """

    return tf.gather_nd(tf.gather_nd(tensor, tf.stack([input_column_index, input_row_index], 2)), output_index)


def resample(tensor: tf.Tensor, shape: list[int], spline_order: int | InterpolationType = 3) -> tf.Tensor:
    """
    Resize an image tensor to the specified shape.

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        spline_order: Interpolation type for resampling

    Returns:
        Processed tensor
    """

    spline_order = coerce_enum(spline_order, InterpolationType)

    input_shape = tf.shape(tensor)

    if input_shape[2] != shape[2]:  # Channels differ; perform conversion
        if input_shape[2] == 1:
            if shape[2] == 2:
                # Grayscale → Grayscale+Alpha: append alpha=1 channel
                alpha = tf.ones_like(tensor)
                tensor = tf.concat([tensor, alpha], axis=2)

            elif shape[2] == 3:
                # Grayscale → RGB: replicate grayscale value across R, G, B
                tensor = tf.concat([tensor, tensor, tensor], axis=2)

            elif shape[2] == 4:
                # Grayscale → RGBA: replicate grayscale for RGB, then append alpha=1
                rgb = tf.concat([tensor, tensor, tensor], axis=2)
                alpha = tf.ones_like(tensor)
                tensor = tf.concat([rgb, alpha], axis=2)

        elif input_shape[2] == 2:
            lum = tensor[..., 0:1]
            alpha = tensor[..., 1:2]

            if shape[2] == 1:
                # Grayscale+Alpha → Grayscale: multiply lum by alpha
                tensor = lum * alpha

            elif shape[2] == 3:
                # Grayscale+Alpha → RGB: multiply lum by alpha, then replicate for R, G, B
                lum_alpha = lum * alpha
                tensor = tf.concat([lum_alpha, lum_alpha, lum_alpha], axis=2)

            elif shape[2] == 4:
                # Grayscale+Alpha → RGBA: replicate lum for RGB, keep original alpha
                rgb = tf.concat([lum, lum, lum], axis=2)
                tensor = tf.concat([rgb, alpha], axis=2)

        elif input_shape[2] == 3:
            if shape[2] == 1:
                # RGB → Grayscale: use value_map to compute luminance, keep dimensions
                tensor = value_map(tensor, shape, keepdims=True)

            elif shape[2] == 2:
                # RGB → Grayscale+Alpha: compute grayscale, then append alpha=1
                gray = value_map(tensor, shape, keepdims=True)
                alpha = tf.ones_like(gray)
                tensor = tf.concat([gray, alpha], axis=2)

            elif shape[2] == 4:
                # RGB → RGBA: append alpha=1 channel to RGB
                alpha = tf.ones_like(tensor[..., 0:1])
                tensor = tf.concat([tensor, alpha], axis=2)

        elif input_shape[2] == 4:
            rgb = tensor[..., 0:3]
            alpha = tensor[..., 3:4]

            if shape[2] == 1:
                # RGBA → Grayscale: drop alpha, compute grayscale on RGB
                tensor = value_map(rgb, shape, keepdims=True)

            elif shape[2] == 2:
                # RGBA → Grayscale+Alpha: compute grayscale from RGB, keep original alpha
                gray = value_map(rgb, shape, keepdims=True)
                tensor = tf.concat([gray, alpha], axis=2)

            elif shape[2] == 3:
                # RGBA → RGB: drop alpha channel
                tensor = rgb

    # Blown up row and column indices. These map into input tensor, producing a big blocky version.
    resized_row_index = tf.cast(row_index(shape), tf.float32) * (
        tf.cast(input_shape[1], tf.float32) / tf.cast(shape[1], tf.float32)
    )  # 0, 1, 2, 3, -> 0, 0.5, 1, 1.5A

    resized_col_index = tf.cast(column_index(shape), tf.float32) * (tf.cast(input_shape[0], tf.float32) / tf.cast(shape[0], tf.float32))

    # Map to input indices as int
    resized_row_index_trunc = tf.floor(resized_row_index)
    resized_col_index_trunc = tf.floor(resized_col_index)
    resized_index_trunc = tf.cast(tf.stack([resized_col_index_trunc, resized_row_index_trunc], 2), tf.int32)

    # Resized original
    resized: defaultdict[int, dict[int, tf.Tensor]] = defaultdict(dict)
    resized[1][1] = tf.gather_nd(tensor, resized_index_trunc)

    if spline_order == InterpolationType.constant:
        return resized[1][1]

    # Resized neighbors
    input_rows: defaultdict[int, tf.Tensor] = defaultdict(dict)
    input_columns: defaultdict[int, tf.Tensor] = defaultdict(dict)

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


def proportional_downsample(tensor: tf.Tensor, shape: list[int], new_shape: list[int]) -> tf.Tensor:
    """
    Given a new shape which is evenly divisible by the old shape, shrink the image by averaging pixel values.

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        new_shape: New shape for resampling

    Returns:
        Processed tensor
    """

    kernel_shape = [max(int(shape[0] / new_shape[0]), 1), max(int(shape[1] / new_shape[1]), 1), shape[2], 1]

    kernel = tf.ones(kernel_shape, dtype=tf.float32)

    try:
        out = tf.nn.depthwise_conv2d([tensor], kernel, [1, kernel_shape[0], kernel_shape[1], 1], "VALID")[0] / (kernel_shape[0] * kernel_shape[1])
    except Exception:
        out = tensor
        # ValueError(f"Could not convolve with kernel shape: {kernel_shape}: {e}")

    return resample(out, new_shape)


def row_index(shape: list[int]) -> tf.Tensor:
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

    Args:
        shape: Shape of the tensor [height, width, channels]

    Returns:
        Index tensor
    """

    height = shape[0]
    width = shape[1]

    row_identity = tf.cumsum(tf.ones([width], dtype=tf.int32), exclusive=True)
    row_identity = tf.reshape(tf.tile(row_identity, [height]), [height, width])

    return row_identity


def column_index(shape: list[int]) -> tf.Tensor:
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

    Args:
        shape: Shape of the tensor [height, width, channels]

    Returns:
        Index tensor
    """

    height = shape[0]
    width = shape[1]

    column_identity = tf.ones([width], dtype=tf.int32)
    column_identity = tf.tile(column_identity, [height])
    column_identity = tf.reshape(column_identity, [height, width])
    column_identity = tf.cumsum(column_identity, exclusive=True)

    return column_identity


def offset(tensor: tf.Tensor, shape: list[int], x: int | float = 0, y: int | float = 0) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        x: X offset amount
        y: Y offset amount

    Returns:
        Processed tensor
    """

    if x == 0 and y == 0:
        return tensor

    return tf.gather_nd(tensor, tf.stack([(column_index(shape) + y) % shape[0], (row_index(shape) + x) % shape[1]], 2))


def _linear_components(a: tf.Tensor, b: tf.Tensor, g: tf.Tensor) -> tf.Tensor:
    """
    Args:
        a: First value for blending/distance
        b: Second value for blending/distance
        g: Interpolation factor (0.0-1.0)

    Returns:
        Processed value
    """
    return a * (1 - g), b * g


def blend(a: tf.Tensor, b: tf.Tensor, g: tf.Tensor) -> tf.Tensor:
    """
    Blend a and b values with linear interpolation.

    Args:
        a: First value for blending/distance
        b: Second value for blending/distance
        g: Interpolation factor (0.0-1.0)

    Returns:
        Blended tensor
    """

    return sum(_linear_components(a, b, g))


def _cosine_components(a: tf.Tensor, b: tf.Tensor, g: tf.Tensor) -> tf.Tensor:
    """
    Args:
        a: First value for blending/distance
        b: Second value for blending/distance
        g: Interpolation factor (0.0-1.0)

    Returns:
        Processed value
    """
    # This guy is great http://paulbourke.net/miscellaneous/interpolation/

    g2 = (1 - tf.cos(g * math.pi)) / 2

    return a * (1 - g2), b * g2


def blend_cosine(a: tf.Tensor, b: tf.Tensor, g: tf.Tensor) -> tf.Tensor:
    """
    Blend a and b values with cosine interpolation.

    Args:
        a: First value for blending/distance
        b: Second value for blending/distance
        g: Interpolation factor (0.0-1.0)

    Returns:
        Blended tensor
    """

    return sum(_cosine_components(a, b, g))


def _cubic_components(a: tf.Tensor, b: tf.Tensor, c: tf.Tensor, d: tf.Tensor, g: tf.Tensor) -> tf.Tensor:
    """
    Args:
        a: First value for blending/distance
        b: Second value for blending/distance
        c: Third value for blending
        d: Fourth value for blending
        g: Interpolation factor (0.0-1.0)

    Returns:
        Processed value
    """
    # This guy is great http://paulbourke.net/miscellaneous/interpolation/

    g2 = g * g

    a0 = d - c - a + b
    a1 = a - b - a0
    a2 = c - a
    a3 = b

    return a0 * g * g2, a1 * g2, a2 * g + a3


def blend_cubic(a: tf.Tensor, b: tf.Tensor, c: tf.Tensor, d: tf.Tensor, g: tf.Tensor) -> tf.Tensor:
    """
    Blend b and c values with bi-cubic interpolation.

    Args:
        a: First value for blending/distance
        b: Second value for blending/distance
        c: Third value for blending
        d: Fourth value for blending
        g: Interpolation factor (0.0-1.0)

    Returns:
        Blended tensor
    """

    return sum(_cubic_components(a, b, c, d, g))


@effect()
def smoothstep(tensor: tf.Tensor, shape: list[int], a: tf.Tensor = 0.0, b: tf.Tensor = 1.0, time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """

    .. noisemaker-live::
       :effect: smoothstep
       :input: basic
       :seed: 42
       :width: 512
       :height: 256
       :lazy:

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        a: First value for blending/distance
        b: Second value for blending/distance
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Processed tensor
    """
    t = tf.clip_by_value((tensor - a) / (b - a), 0.0, 1.0)

    return t * t * (3.0 - 2.0 * t)


def freq_for_shape(freq: int | list[int], shape: list[int]) -> tf.Tensor:
    """
    Given a base frequency as int, generate noise frequencies for each spatial dimension.

    Args:
        freq: Frequency for noise generation
        shape: Shape of the tensor [height, width, channels]

    Returns:
        Processed tensor
    """

    if isinstance(freq, list):
        freq = freq[0]

    height = shape[0]
    width = shape[1]

    if height == width:
        return [freq, freq]

    elif height < width:
        return [freq, int(freq * width / height)]

    else:
        return [int(freq * height / width), freq]


def ridge(tensor: tf.Tensor) -> tf.Tensor:
    """
    Create a "ridge" at midpoint values. 1 - abs(n * 2 - 1)
    .. image:: images/crease.jpg
    :width: 1024
    :height: 256
    :alt: Noisemaker example output (CC0)

    Args:
        tensor: Input tensor to process

    Returns:
        Processed tensor
    """

    return 1.0 - tf.abs(tensor * 2 - 1)


def simple_multires(
    freq: int | list[int],
    shape: list[int],
    octaves: int = 1,
    spline_order: int | InterpolationType = InterpolationType.bicubic,
    distrib: ValueDistribution = ValueDistribution.simplex,
    corners: bool = False,
    ridges: bool = False,
    mask: ValueMask | None = None,
    mask_inverse: bool = False,
    mask_static: bool = False,
    time: float = 0.0,
    speed: float = 1.0,
) -> tf.Tensor:
    """
    Generate multi-octave value noise. Unlike generators.multires, this function is single-channel and does not apply effects.

    Args:
        freq: Frequency for noise generation
        shape: Shape of the tensor [height, width, channels]
        octaves: Number of octave layers
        spline_order: Interpolation type for resampling
        distrib: Value distribution method
        corners: Include corner values
        ridges: Apply ridge transformation
        mask: Value mask to apply
        mask_inverse: Invert the mask
        mask_static: Use static mask values
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Processed tensor
    """

    if isinstance(freq, int):
        freq = freq_for_shape(freq, shape)

    tensor = tf.zeros(shape, dtype=tf.float32)

    for octave in range(1, octaves + 1):
        multiplier = 2**octave

        freq_list = freq if isinstance(freq, list) else [freq, freq]
        base_freq = [int(f * 0.5 * multiplier) for f in freq_list]

        if all(base_freq[i] > shape[i] for i in range(len(base_freq))):
            break

        layer = values(
            freq=base_freq,
            shape=shape,
            spline_order=spline_order,
            distrib=distrib,
            corners=corners,
            mask=mask,
            mask_inverse=mask_inverse,
            mask_static=mask_static,
            time=time,
            speed=speed,
        )

        if ridges:
            layer = ridge(layer)

        tensor += layer / multiplier

    return normalize(tensor)


def value_shape(shape: list[int]) -> tf.Tensor:
    """
    Args:
        shape: Shape of the tensor [height, width, channels]

    Returns:
        Processed tensor
    """

    return [shape[0], shape[1], 1]


def normalized_sine(value: float) -> tf.Tensor:
    """
    Args:
        value: Input value

    Returns:
        Normalized tensor
    """

    return (tf.sin(value) + 1.0) * 0.5


def _conform_kernel_to_tensor(kernel: ValueMask, tensor: tf.Tensor, shape: list[int]) -> tf.Tensor:
    """
    Re-shape a convolution kernel to match the given tensor's color dimensions.

    Args:
        kernel: Convolution kernel mask
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]

    Returns:
        Processed value
    """

    if isinstance(kernel, ValueMask):
        values, _ = masks.mask_values(kernel)
        arr = np.asarray(values, dtype=np.float32)
    else:
        arr = np.asarray(kernel, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]

    if arr.ndim != 2:
        raise ValueError("Convolution kernel must be 2-D")

    height, width = arr.shape
    channels = shape[-1]

    tiled = np.repeat(arr[:, :, None], channels, axis=2)

    temp = tf.reshape(tiled, (height, width, channels, 1))

    temp = tf.cast(temp, tf.float32)

    # Normalize the kernel to match the JavaScript implementation, which scales
    # the filter by the largest absolute value to preserve relative weights.
    denom = tf.maximum(tf.reduce_max(temp), tf.reduce_min(temp) * -1)
    temp = tf.math.divide_no_nan(temp, denom)

    return temp


@effect()
def convolve(
    tensor: tf.Tensor,
    shape: list[int],
    kernel: ValueMask = ValueMask.conv2d_blur,
    with_normalize: bool = True,
    alpha: float = 1.0,
    time: float = 0.0,
    speed: float = 1.0,
) -> tf.Tensor:
    """
    Apply a convolution kernel to an image tensor.
    .. code-block:: python
    image = convolve(image, shape, ValueMask.conv2d_shadow)


    .. noisemaker-live::
       :effect: convolve
       :input: basic
       :seed: 42
       :width: 512
       :height: 256
       :lazy:

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        kernel: Convolution kernel mask
        with_normalize: Normalize the output
        alpha: Blending alpha value (0.0-1.0)
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Processed tensor
    """

    height, width, channels = shape

    if kernel is None:
        kernel = ValueMask.conv2d_blur

    kernel = coerce_enum(kernel, ValueMask)

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
        out = tf.abs(out - 0.5) * 2

    if alpha == 1.0:
        return out

    return blend(tensor, out, alpha)


@effect()
def refract(
    tensor: tf.Tensor,
    shape: list[int],
    displacement: float = 0.5,
    reference_x: tf.Tensor | None = None,
    reference_y: tf.Tensor | None = None,
    warp_freq: int | list[int] | None = None,
    spline_order: int | InterpolationType = InterpolationType.bicubic,
    from_derivative: bool = False,
    signed_range: bool = True,
    time: float = 0.0,
    speed: float = 1.0,
    y_from_offset: bool = False,
) -> tf.Tensor:
    """
    Apply displacement from pixel values.


    .. noisemaker-live::
       :effect: refract
       :input: basic
       :seed: 42
       :width: 512
       :height: 256
       :lazy:

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        displacement: Displacement
        reference_x: Reference x
        reference_y: Reference y
        warp_freq: Warp freq
        spline_order: Interpolation type for resampling
        from_derivative: From derivative
        signed_range: Use signed range (-1 to 1)
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier
        y_from_offset: Y from offset

    Returns:
        Processed tensor
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
            assert warp_shape is not None
            reference_x = values(freq=warp_freq, shape=warp_shape, distrib=ValueDistribution.simplex, time=time, speed=speed, spline_order=spline_order)

        else:
            reference_x = tensor

    if reference_y is None:
        if from_derivative:
            reference_y = convolve(kernel=ValueMask.conv2d_deriv_y, tensor=tensor, shape=shape, with_normalize=False)

        elif warp_freq:
            assert warp_shape is not None
            reference_y = values(freq=warp_freq, shape=warp_shape, distrib=ValueDistribution.simplex, time=time, speed=speed, spline_order=spline_order)

        else:
            if y_from_offset:
                # "the old way"
                y0_index += int(height * 0.5)
                x0_index += int(width * 0.5)
                reference_y = tf.gather_nd(reference_x, tf.stack([y0_index % height, x0_index % width], 2))
            else:
                reference_y = reference_x
                reference_x = tf.cos(reference_x * math.tau)
                reference_y = tf.sin(reference_y * math.tau)
                reference_x = tf.clip_by_value(reference_x * 0.5 + 0.5, 0.0, 1.0)
                reference_y = tf.clip_by_value(reference_y * 0.5 + 0.5, 0.0, 1.0)

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
    x_coords = tf.cast(x0_index, tf.float32)
    y_coords = tf.cast(y0_index, tf.float32)

    sample_x = x_coords + x_offsets
    sample_y = y_coords + y_offsets

    width_f = tf.cast(width, tf.float32)
    height_f = tf.cast(height, tf.float32)

    sample_x_wrapped = tf.math.floormod(sample_x, width_f)
    sample_y_wrapped = tf.math.floormod(sample_y, height_f)

    x0_base = tf.floor(sample_x_wrapped)
    y0_base = tf.floor(sample_y_wrapped)

    x0_base = tf.minimum(x0_base, width_f - 1)
    y0_base = tf.minimum(y0_base, height_f - 1)

    x0_int = tf.cast(x0_base, tf.int32)
    y0_int = tf.cast(y0_base, tf.int32)

    x0_offsets = tf.math.floormod(x0_int, width)
    x1_offsets = tf.math.floormod(x0_int + 1, width)
    y0_offsets = tf.math.floormod(y0_int, height)
    y1_offsets = tf.math.floormod(y0_int + 1, height)

    x0_y0 = tf.gather_nd(tensor, tf.stack([y0_offsets, x0_offsets], 2))
    x1_y0 = tf.gather_nd(tensor, tf.stack([y0_offsets, x1_offsets], 2))
    x0_y1 = tf.gather_nd(tensor, tf.stack([y1_offsets, x0_offsets], 2))
    x1_y1 = tf.gather_nd(tensor, tf.stack([y1_offsets, x1_offsets], 2))

    x_fract = tf.reshape(sample_x_wrapped - x0_base, [height, width, 1])
    y_fract = tf.reshape(sample_y_wrapped - y0_base, [height, width, 1])
    x_fract = tf.clip_by_value(x_fract, 0.0, 1.0)
    y_fract = tf.clip_by_value(y_fract, 0.0, 1.0)

    x_y0 = blend(x0_y0, x1_y0, x_fract)
    x_y1 = blend(x0_y1, x1_y1, x_fract)

    return blend(x_y0, x_y1, y_fract)


def value_map(tensor: tf.Tensor, shape: list[int], keepdims: bool = False, signed_range: bool = False, with_normalize: bool = True) -> tf.Tensor:
    """
    Create a grayscale value map from the given image Tensor, based on apparent luminance.
    Return value ranges between 0 and 1.

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        keepdims: Keepdims
        signed_range: Use signed range (-1 to 1)
        with_normalize: Normalize the output

    Returns:
        Processed tensor
    """

    # XXX Why is shape sometimes wrong when passed in from refract?
    shape = tf.shape(tensor)

    if shape[2] in (1, 2):
        tensor = tensor[:, :, 0]

    elif shape[2] == 3:
        tensor = oklab.rgb_to_oklab(clamp01(tensor))[:, :, 0]

    elif shape[2] == 4:
        tensor = clamp01(tensor)
        tensor = oklab.rgb_to_oklab(tf.stack([tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]], 2))[:, :, 0]

    if keepdims:
        tensor = tf.expand_dims(tensor, -1)

    if with_normalize:
        tensor = normalize(tensor, signed_range=signed_range)

    elif signed_range:
        tensor = tensor * 2.0 - 1.0

    return tensor


def singularity(tensor: tf.Tensor, shape: list[int], diagram_type: VoronoiDiagramType = VoronoiDiagramType.range, **kwargs: Any) -> tf.Tensor:
    """
    Return the range diagram for a single voronoi point, approximately centered.

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        diagram_type: Type of Voronoi diagram
        **kwargs: Additional keyword arguments for voronoi

    Returns:
        Processed tensor
    """

    result = point_cloud(1, PointDistribution.square, shape)
    if result is None:
        raise ValueError("point_cloud returned None")
    x, y = result

    return voronoi(tensor, shape, diagram_type=diagram_type, xy=(x, y, 1), **kwargs)


def pin_corners(tensor: tf.Tensor, shape: list[int], freq: int | list[int], corners: bool) -> tf.Tensor:
    """
    Pin values to image corners, or align with image center, as per the given "corners" arg.

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        freq: Frequency for noise generation
        corners: Include corner values

    Returns:
        Processed tensor
    """

    if isinstance(freq, int):
        freq = [freq, freq]

    if (not corners and (freq[0] % 2) == 0) or (corners and (freq[0] % 2) == 1):
        tensor = offset(tensor, shape, x=int((shape[1] / freq[1]) * 0.5), y=int((shape[0] / freq[0]) * 0.5))

    return tensor


def coerce_enum(value: float | int | str | Any, cls: type) -> tf.Tensor:
    """
    Attempt to coerce a given string or int value into an Enum instance.

    Args:
        value: Input value
        cls: Enum class to coerce to

    Returns:
        Processed tensor
    """

    if isinstance(value, int):
        value = cls(value)

    elif isinstance(value, str):
        value = cls[value]  # type: ignore[index]

    return value


def clamp01(tensor: tf.Tensor) -> tf.Tensor:
    """
    Args:
        tensor: Input tensor to process

    Returns:
        Processed tensor
    """
    return tf.maximum(tf.minimum(tensor, 1.0), 0.0)


@effect()
def fxaa(tensor: tf.Tensor, shape: list[int], time: float = 0.0, speed: float = 1.0) -> tf.Tensor:
    """

    .. noisemaker-live::
       :effect: fxaa
       :input: basic
       :seed: 42
       :width: 512
       :height: 256
       :lazy:

    Args:
        tensor: Input tensor to process
        shape: Shape of the tensor [height, width, channels]
        time: Time value for animation (0.0-1.0)
        speed: Animation speed multiplier

    Returns:
        Processed tensor
    """
    # Determine the number of channels
    channels = shape[2]

    # Pad tensor to handle boundary conditions
    padded = tf.pad(tensor, [[1, 1], [1, 1], [0, 0]], mode="REFLECT")

    # Extract neighbors for all channels
    center = padded[1:-1, 1:-1, :]
    north = padded[:-2, 1:-1, :]
    south = padded[2:, 1:-1, :]
    west = padded[1:-1, :-2, :]
    east = padded[1:-1, 2:, :]

    if channels == 1:
        # Single-channel (grayscale): use the channel itself as luma
        lC = center
        lN = north
        lS = south
        lW = west
        lE = east

        # Compute weights based on luminance difference
        wC = 1.0
        wN = tf.exp(-tf.abs(lC - lN))
        wS = tf.exp(-tf.abs(lC - lS))
        wW = tf.exp(-tf.abs(lC - lW))
        wE = tf.exp(-tf.abs(lC - lE))
        sum_w = wC + wN + wS + wW + wE + 1e-10

        # Weighted blend on the single channel
        result = (center * wC + north * wN + south * wS + west * wW + east * wE) / sum_w

    elif channels == 2:
        # Two-channel: [grayscale, alpha]
        lumC = center[..., 0:1]
        alpha = center[..., 1:2]
        lumN = north[..., 0:1]
        lumS = south[..., 0:1]
        lumW = west[..., 0:1]
        lumE = east[..., 0:1]

        # Compute weights from grayscale channel only
        wC = 1.0
        wN = tf.exp(-tf.abs(lumC - lumN))
        wS = tf.exp(-tf.abs(lumC - lumS))
        wW = tf.exp(-tf.abs(lumC - lumW))
        wE = tf.exp(-tf.abs(lumC - lumE))
        sum_w = wC + wN + wS + wW + wE + 1e-10

        # Blend only grayscale (first channel), keep alpha unchanged
        blended_lum = (lumC * wC + lumN * wN + lumS * wS + lumW * wW + lumE * wE) / sum_w
        result = tf.concat([blended_lum, alpha], axis=2)

    elif channels == 3:
        # Three-channel (RGB): compute luminance as NTSC weights
        weights = tf.constant([0.299, 0.587, 0.114], dtype=tf.float32)

        lC = tf.reduce_sum(center * weights, axis=-1, keepdims=True)
        lN = tf.reduce_sum(north * weights, axis=-1, keepdims=True)
        lS = tf.reduce_sum(south * weights, axis=-1, keepdims=True)
        lW = tf.reduce_sum(west * weights, axis=-1, keepdims=True)
        lE = tf.reduce_sum(east * weights, axis=-1, keepdims=True)

        # Compute weights from luminance differences
        wC = 1.0
        wN = tf.exp(-tf.abs(lC - lN))
        wS = tf.exp(-tf.abs(lC - lS))
        wW = tf.exp(-tf.abs(lC - lW))
        wE = tf.exp(-tf.abs(lC - lE))
        sum_w = wC + wN + wS + wW + wE + 1e-10

        # Blend RGB channels using those weights
        result = (center * wC + north * wN + south * wS + west * wW + east * wE) / sum_w

    elif channels == 4:
        # Four-channel (RGBA): separate RGB and alpha
        rgbC = center[..., 0:3]
        alpha = center[..., 3:4]
        rgbN = north[..., 0:3]
        rgbS = south[..., 0:3]
        rgbW = west[..., 0:3]
        rgbE = east[..., 0:3]

        # Compute luminance from RGB channels
        weights = tf.constant([0.299, 0.587, 0.114], dtype=tf.float32)

        lC = tf.reduce_sum(rgbC * weights, axis=-1, keepdims=True)
        lN = tf.reduce_sum(rgbN * weights, axis=-1, keepdims=True)
        lS = tf.reduce_sum(rgbS * weights, axis=-1, keepdims=True)
        lW = tf.reduce_sum(rgbW * weights, axis=-1, keepdims=True)
        lE = tf.reduce_sum(rgbE * weights, axis=-1, keepdims=True)

        # Compute weights from luminance differences
        wC = 1.0
        wN = tf.exp(-tf.abs(lC - lN))
        wS = tf.exp(-tf.abs(lC - lS))
        wW = tf.exp(-tf.abs(lC - lW))
        wE = tf.exp(-tf.abs(lC - lE))
        sum_w = wC + wN + wS + wW + wE + 1e-10

        # Blend only RGB channels; keep alpha unchanged
        blended_rgb = (rgbC * wC + rgbN * wN + rgbS * wS + rgbW * wW + rgbE * wE) / sum_w
        result = tf.concat([blended_rgb, alpha], axis=2)

    else:
        # Unexpected channel count: no-op
        result = tensor

    return result
