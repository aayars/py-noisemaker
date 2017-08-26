from collections import deque
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


def point_cloud(freq, distrib=PointDistribution.random, shape=None, center=True, generations=1):
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
        point_func = square_grid

    elif distrib == PointDistribution.spiral.value:
        point_func = spiral

    elif PointDistribution.is_circular(distrib):
        point_func = circular

    stack = deque()
    stack.append((half_width, half_height, 1))

    while stack:
        x_point, y_point, generation = stack.popleft()

        if generation <= generations:
            _x, _y = point_func(freq, distrib, center, x_point, y_point, half_width / generation, half_height / generation, width, height)

            for i in range(len(_x)):
                x_point = _x[i]
                y_point = _y[i]

                stack.append((x_point, y_point, generation + 1))

                x.append(x_point)
                y.append(y_point)

    return (x, y)


def rand(freq, distrib, center, center_x, center_y, half_width, half_height, width, height):
    """
    """

    x = []
    y = []

    for i in range(freq * freq):
        _x = (center_x + (random.random() * (half_width * 2.0) - half_width)) % width
        _y = (center_y + (random.random() * (half_height * 2.0) - half_height)) % height

        x.append(_x)
        y.append(_y)

    return x, y


def square_grid(freq, distrib, center, center_x, center_y, half_width, half_height, width, height):
    """
    """

    x = []
    y = []

    # Keep a node in the center of the image, or pin to corner:
    drift_amount = .5 / freq

    if ((freq * freq) % 2) == 0:
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

            _x = (center_x + (((a / freq) + drift + x_drift) * half_width * 2)) % width
            _y = (center_y + (((b / freq) + drift + y_drift) * half_height * 2)) % height

            x.append(_x)
            y.append(_y)

    return x, y


def spiral(freq, distrib, center, center_x, center_y, half_width, half_height, width, height):
    kink = random.random() * 12.5 - 25

    x = []
    y = []

    count = freq * freq

    for i in range(count):
        fract = i / count

        degrees = fract * 360.0 * math.radians(1) * kink

        x.append((center_x + math.sin(degrees) * fract * half_width) % width)
        y.append((center_y + math.cos(degrees) * fract * half_height) % height)

    return x, y


def circular(freq, distrib, center, center_x, center_y, half_width, half_height, width, height):
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

            x.append((center_x + math.sin(degrees) * dist_fract * half_width) % width)
            y.append((center_y + math.cos(degrees) * dist_fract * half_height) % height)

    return x, y