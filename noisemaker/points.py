"""Point cloud library for Noisemaker. Used for Voronoi and DLA functions."""

from __future__ import annotations

import math
from typing import Any

import noisemaker.masks as masks
import noisemaker.rng as rng
import noisemaker.simplex as simplex
from noisemaker.constants import PointDistribution, ValueMask


def point_cloud(
    freq: int,
    distrib: PointDistribution | ValueMask = PointDistribution.random,
    shape: list[int] | None = None,
    corners: bool = False,
    generations: int = 1,
    drift: float = 0.0,
    time: float = 0.0,
    speed: float = 1.0,
) -> tuple[list[Any], list[Any]] | None:
    """
    Generate a point cloud for Voronoi diagrams or other point-based effects.

    Args:
        freq: Point frequency/density
        distrib: Point distribution method (PointDistribution or ValueMask)
        shape: Optional shape [height, width, channels]
        corners: If True, anchor points to corners instead of center
        generations: Number of generations for iterative distributions
        drift: Amount of random drift to apply to points
        time: Time parameter for animation
        speed: Animation speed multiplier

    Returns:
        Tuple of (x_coords, y_coords) lists, or None if freq is 0
    """

    if not freq:
        return None

    x: list[Any] = []
    y: list[Any] = []

    if shape is None:
        width = 1.0
        height = 1.0

    else:
        width = shape[1]
        height = shape[0]

    if isinstance(distrib, int):
        if any(d.value == distrib for d in PointDistribution):
            distrib = PointDistribution(distrib)
        else:
            distrib = ValueMask(distrib)

    elif isinstance(distrib, str):
        if any(d.name == distrib for d in PointDistribution):
            distrib = PointDistribution[distrib]
        else:
            distrib = ValueMask[distrib]

    if distrib in ValueMask.procedural_members():
        raise Exception("Procedural ValueMask can't be used as a PointDistribution.")

    point_func: Any = rand

    range_x = width * 0.5
    range_y = height * 0.5

    #
    seen: set[tuple[Any, Any]] = set()
    active_set: set[tuple[Any, Any, int]] = set()

    if isinstance(distrib, PointDistribution):
        if PointDistribution.is_grid(distrib):
            point_func = square_grid

        elif distrib == PointDistribution.spiral:
            point_func = spiral

        elif PointDistribution.is_circular(distrib):
            point_func = circular

        if PointDistribution.is_grid(distrib):
            active_set.add((0.0, 0.0, 1))

        else:
            active_set.add((range_y, range_x, 1))

    else:
        # Use a ValueMask as a PointDistribution!
        if shape is None:
            raise ValueError("shape must be provided when using ValueMask as PointDistribution")

        mask: Any = masks.Masks[distrib]
        mask_shape = masks.mask_shape(distrib)

        x_space = shape[1] / mask_shape[1]
        y_space = shape[0] / mask_shape[0]

        x_margin = x_space * 0.5
        y_margin = y_space * 0.5

        for _x in range(mask_shape[1]):
            for _y in range(mask_shape[0]):
                pixel = mask[_y][_x]

                if isinstance(pixel, list):
                    pixel = sum(p for p in pixel)

                if drift:
                    x_drift = simplex.random(time, speed=speed) * drift / freq * shape[1]
                    y_drift = simplex.random(time, speed=speed) * drift / freq * shape[0]
                else:
                    x_drift = 0
                    y_drift = 0

                if pixel != 0:
                    x.append(int(x_margin + _x * x_space + x_drift))
                    y.append(int(y_margin + _y * y_space + y_drift))

        return x, y

    seen.update((x, y) for x, y, _ in active_set)

    while active_set:
        x_point, y_point, generation = active_set.pop()

        if generation <= generations:
            multiplier = max(2 * (generation - 1), 1)

            _x, _y = point_func(
                freq=freq,
                distrib=distrib,
                corners=corners,
                center_x=x_point,
                center_y=y_point,
                range_x=range_x / multiplier,
                range_y=range_y / multiplier,
                width=width,
                height=height,
                generation=generation,
                time=time,
                speed=speed * 0.1,
            )

            for i in range(len(_x)):
                x_point = _x[i]
                y_point = _y[i]

                if (x_point, y_point) in seen:
                    continue

                seen.add((x_point, y_point))

                active_set.add((x_point, y_point, generation + 1))

                if drift:
                    x_drift = simplex.random(time, speed=speed) * drift
                    y_drift = simplex.random(time, speed=speed) * drift

                else:
                    x_drift = 0
                    y_drift = 0

                if shape is None:
                    x_point = (x_point + x_drift / freq) % 1.0
                    y_point = (y_point + y_drift / freq) % 1.0

                else:
                    x_point = int(x_point + (x_drift / freq * shape[1])) % shape[1]
                    y_point = int(y_point + (y_drift / freq * shape[0])) % shape[0]

                x.append(x_point)
                y.append(y_point)

    return (x, y)


def cloud_points(count: int, seed: int | None = None) -> tuple[list[float], list[float]]:
    """Convenience wrapper for random point clouds.

    RNG: ``count * count * 2`` calls to :func:`rng.random` via :func:`rand`,
    ordered as x then y for each point.

    Args:
        count: Number of points per axis (total points = count * count).
        seed: Optional random seed for reproducible output.

    Returns:
        Tuple of (x_coords, y_coords) lists with normalized [0.0, 1.0] coordinates.
    """

    if seed is not None:
        rng.set_seed(seed)

    result = point_cloud(count, PointDistribution.random)
    if result is None:
        return ([], [])
    return result


def rand(
    freq: int = 2,
    center_x: float = 0.5,
    center_y: float = 0.5,
    range_x: float = 0.5,
    range_y: float = 0.5,
    width: float = 1.0,
    height: float = 1.0,
    **kwargs: Any,
) -> tuple[list[float], list[float]]:
    """Generate a random cloud of points within a specified region.

    RNG: ``freq * freq * 2`` calls to :func:`rng.random`, ordered as x then y.

    Args:
        freq: Number of points per axis (total points = freq * freq).
        center_x: Horizontal center of the distribution region [0.0, 1.0].
        center_y: Vertical center of the distribution region [0.0, 1.0].
        range_x: Horizontal radius from center [0.0, 1.0].
        range_y: Vertical radius from center [0.0, 1.0].
        width: Horizontal wrapping bounds.
        height: Vertical wrapping bounds.
        **kwargs: Unused; accepts additional parameters for compatibility.

    Returns:
        Tuple of (x_coords, y_coords) lists with coordinates wrapped to [0.0, width) and [0.0, height).
    """

    x = []
    y = []

    for i in range(freq * freq):
        _x = (center_x + (rng.random() * (range_x * 2.0) - range_x)) % width  # RNG[x]
        _y = (center_y + (rng.random() * (range_y * 2.0) - range_y)) % height  # RNG[y]

        x.append(_x)
        y.append(_y)

    return x, y


def square_grid(
    freq: float = 1.0,
    distrib: PointDistribution | None = None,
    corners: bool = False,
    center_x: float = 0.0,
    center_y: float = 0.0,
    range_x: float = 1.0,
    range_y: float = 1.0,
    width: float = 1.0,
    height: float = 1.0,
    **kwargs: Any,
) -> tuple[list[float], list[float]]:
    """Generate a square grid of points with optional distribution patterns.

    Supports various grid patterns including waffle, chess, and hexagonal layouts.

    Args:
        freq: Number of grid divisions per axis (total points = freq * freq).
        distrib: Optional point distribution pattern (waffle, chess, h_hex, v_hex).
        corners: If True, align grid to corners; otherwise center the grid.
        center_x: Horizontal center offset [0.0, 1.0].
        center_y: Vertical center offset [0.0, 1.0].
        range_x: Horizontal scale factor.
        range_y: Vertical scale factor.
        width: Horizontal wrapping bounds.
        height: Vertical wrapping bounds.
        **kwargs: Unused; accepts additional parameters for compatibility.

    Returns:
        Tuple of (x_coords, y_coords) lists with grid coordinates.
    """

    x = []
    y = []

    # Keep a node in the center of the image, or pin to corner:
    drift_amount = 0.5 / freq

    if (freq % 2) == 0:
        drift = 0.0 if not corners else drift_amount

    else:
        drift = drift_amount if not corners else 0.0

    #
    for a in range(int(freq)):
        for b in range(int(freq)):
            if distrib == PointDistribution.waffle and (b % 2) == 0 and (a % 2) == 0:
                continue

            if distrib == PointDistribution.chess and (a % 2) == (b % 2):
                continue

            #
            if distrib == PointDistribution.h_hex:
                x_drift = drift_amount if (b % 2) == 1 else 0

            else:
                x_drift = 0

            #
            if distrib == PointDistribution.v_hex:
                y_drift = 0 if (a % 2) == 1 else drift_amount

            else:
                y_drift = 0

            _x = (center_x + (((a / freq) + drift + x_drift) * range_x * 2)) % width
            _y = (center_y + (((b / freq) + drift + y_drift) * range_y * 2)) % height

            x.append(_x)
            y.append(_y)

    return x, y


def spiral(
    freq: float = 1.0,
    center_x: float = 0.0,
    center_y: float = 0.0,
    range_x: float = 1.0,
    range_y: float = 1.0,
    width: float = 1.0,
    height: float = 1.0,
    time: float = 0.0,
    speed: float = 1.0,
    **kwargs: Any,
) -> tuple[list[float], list[float]]:
    """Generate points along a spiral path with time-based rotation.

    RNG: 1 call to :func:`rng.random` for spiral kink factor.

    Args:
        freq: Number of points to generate.
        center_x: Horizontal center of the spiral [0.0, 1.0].
        center_y: Vertical center of the spiral [0.0, 1.0].
        range_x: Horizontal radius of the spiral.
        range_y: Vertical radius of the spiral.
        width: Horizontal wrapping bounds.
        height: Vertical wrapping bounds.
        time: Animation time parameter for rotation.
        speed: Animation speed multiplier.
        **kwargs: Unused; accepts additional parameters for compatibility.

    Returns:
        Tuple of (x_coords, y_coords) lists with spiral coordinates.
    """

    kink = 0.5 + rng.random() * 0.5

    x = []
    y = []

    count = freq * freq

    for i in range(int(count)):
        fract = i / count

        degrees = fract * 360.0 * math.radians(1) * kink

        x.append((center_x + math.sin(degrees) * fract * range_x) % width)
        y.append((center_y + math.cos(degrees) * fract * range_y) % height)

    return x, y


def circular(
    freq: float = 1.0,
    distrib: PointDistribution = PointDistribution.circular,
    center_x: float = 0.0,
    center_y: float = 0.0,
    range_x: float = 1.0,
    range_y: float = 1.0,
    width: float = 1.0,
    height: float = 1.0,
    generation: int = 1,
    time: float = 0.0,
    speed: float = 1.0,
    **kwargs: Any,
) -> tuple[list[float], list[float]]:
    """Generate points in concentric circular rings.

    Creates a center point surrounded by rings of evenly spaced points.

    Args:
        freq: Number of rings and points per ring.
        distrib: Point distribution method (circular or rotating).
        center_x: Horizontal center of the circular pattern [0.0, 1.0].
        center_y: Vertical center of the circular pattern [0.0, 1.0].
        range_x: Horizontal radius scale factor.
        range_y: Vertical radius scale factor.
        width: Horizontal wrapping bounds.
        height: Vertical wrapping bounds.
        generation: Ring generation parameter for pattern variation.
        time: Animation time parameter for rotation.
        speed: Animation speed multiplier.
        **kwargs: Unused; accepts additional parameters for compatibility.

    Returns:
        Tuple of (x_coords, y_coords) lists with circular coordinates.
    """

    x = []
    y = []

    ring_count = freq
    dot_count = freq

    x.append(center_x)
    y.append(center_y)

    rotation = (1 / dot_count) * 360.0 * math.radians(1)

    kink = 0.5 + rng.random() * 0.5

    for i in range(1, int(ring_count) + 1):
        dist_fract = i / ring_count

        for j in range(1, int(dot_count) + 1):
            rads = j * rotation

            if distrib == PointDistribution.circular:
                rads += rotation * 0.5 * i

            if distrib == PointDistribution.rotating:
                rads += rotation * dist_fract * kink

            x_point = center_x + math.sin(rads) * dist_fract * range_x
            y_point = center_y + math.cos(rads) * dist_fract * range_y

            x.append(x_point % width)
            y.append(y_point % height)

    return x, y
