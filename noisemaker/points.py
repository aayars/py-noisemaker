from enum import Enum

import math
import random


class PointDistribution(Enum):
    """
    Point cloud distribution, used by Voronoi and DLA
    """

    none = 0

    random = 1

    square = 2

    horizontal_hex = 3

    vertical_hex = 4

    spiral = 5

    circular = 6

    concentric = 7

    @staticmethod
    def is_grid(member):
        if isinstance(member, PointDistribution):
            member = member.value

        return member in (2, 3, 4)

    @staticmethod
    def is_circular(member):
        if isinstance(member, PointDistribution):
            member = member.value

        return member in (6, 7)


def point_cloud(freq, distrib=PointDistribution.random, shape=None, center=True):
    """
    """

    if not freq:
        return

    x = []
    y = []

    if shape is None:
        width = 1.0
        height = 1.0

    else:
        width = shape[1]
        height = shape[0]

    half_width = width * .5
    half_height = height * .5

    if isinstance(distrib, PointDistribution):
        distrib = distrib.value

    count = freq * freq

    point_func = rand

    if PointDistribution.is_grid(distrib):
        point_func = hex

    elif distrib == PointDistribution.spiral.value:
        point_func = spiral

    elif PointDistribution.is_circular(distrib):
        point_func = circular

    _x, _y = point_func(freq, distrib, half_width, half_height, half_width, half_height)

    x += _x
    y += _y

    return (x, y)


def rand(freq, distrib, center_x, center_y, half_width, half_height):
    """
    """

    for i in range(count):
        _x = random.random() * half_width * 2.0
        _y = random.random() * half_height * 2.0

        x.append(_x)
        y.append(_y)


def hex(freq, distrib, center_x, center_y, half_width, half_height):
    """
    """

    # Keep a node in the center of the image, or pin to corner:
    drift_amount = .5 / freq

    if (count % 2) == 0:
        drift = 0.0 if center else drift_amount

    else:
        drift = drift_amount if center else 0.0

    #
    for a in range(freq):
        for b in range(freq):
            if distrib == PointDistribution.horizontal_hex.value:
                x_drift = drift_amount if (b % 2) == 1 else 0

            else:
                x_drift = 0

            if distrib == PointDistribution.vertical_hex.value:
                y_drift = 0 if (a % 2) == 1 else drift_amount

            else:
                y_drift = 0

            _x = (((a / freq) + drift + x_drift) * half_width * 2) % (half_width * 2.0)
            _y = (((b / freq) + drift + y_drift) * half_height * 2) % (half_height * 2.0)

            x.append(_x)
            y.append(_y)


def spiral(freq, distrib, center_x, center_y, half_width, half_height):
    kink = random.random() * 12.5 - 25

    x = []
    y = []

    for i in range(count):
        fract = i / count

        degrees = fract * 360.0 * math.radians(1) * kink

        x.append((half_width + math.sin(degrees) * fract * half_width) % (half_width * 2.0))
        y.append((half_height + math.cos(degrees) * fract * half_height) % (half_height * 2.0))

    return x, y


def circular(freq, distrib, center_x, center_y, half_width, half_height):
    """
    """

    x = []
    y = []

    ring_count = freq
    dot_count = freq

    x.append(center_x)
    y.append(center_y)

    rotation = (1 / dot_count) * 360.0 * math.radians(1)

    for i in range(1, ring_count + 1):
        dist_fract = i / ring_count

        for j in range(1, dot_count + 1):
            degrees = j * rotation

            if distrib == PointDistribution.circular.value and (i % 2) == 0:
                degrees += rotation * .5

            x.append((center_x + math.sin(degrees) * dist_fract * center_x) % (half_width * 2.0))
            y.append((center_y + math.cos(degrees) * dist_fract * center_y) % (half_height * 2.0))

    return x, y