import math
import random

from noisemaker.constants import PointDistribution


def point_cloud(freq, distrib=PointDistribution.random, shape=None, corners=False, generations=1, drift=0.0):
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

    range_x = width * .5
    range_y = height * .5

    if isinstance(distrib, int):
        distrib = PointDistribution(distrib)

    elif isinstance(distrib, str):
        distrib = PointDistribution[distrib]

    point_func = rand

    if PointDistribution.is_grid(distrib):
        point_func = square_grid

    elif distrib == PointDistribution.spiral:
        point_func = spiral

    elif PointDistribution.is_circular(distrib):
        point_func = circular

    #
    seen = set()
    active_set = set()

    if PointDistribution.is_grid(distrib):
        active_set.add((0.0, 0.0, 1))

    else:
        active_set.add((range_x, range_y, 1))

    seen.update(active_set)

    while active_set:
        x_point, y_point, generation = active_set.pop()

        if generation <= generations:
            multiplier = max(2 * (generation - 1), 1)

            next = point_func(freq=freq, distrib=distrib, corners=corners,
                              center_x=x_point, center_y=y_point, range_x=range_x / multiplier, range_y=range_y / multiplier,
                              width=width, height=height, generation=generation)

            _x, _y = next

            for i in range(len(_x)):
                x_point = _x[i]
                y_point = _y[i]

                if drift:
                    x_drift = random.random() * drift - drift * .5
                    y_drift = random.random() * drift - drift * .5

                else:
                    x_drift = 0
                    y_drift = 0

                if shape is None:
                    x_point = (x_point + x_drift / freq) % 1.0
                    y_point = (y_point + y_drift / freq) % 1.0

                else:
                    x_point = int(x_point + (x_drift / freq * shape[1])) % shape[1]
                    y_point = int(y_point + (y_drift / freq * shape[0])) % shape[0]

                    if (x_point, y_point) in seen:
                        continue

                    seen.add((x_point, y_point))

                active_set.add((x_point, y_point, generation + 1))

                x.append(x_point)
                y.append(y_point)

    return (x, y)


def rand(freq=1.0, center_x=0.0, center_y=0.0, range_x=1.0, range_y=1.0, width=1.0, height=1.0, **kwargs):
    """
    """

    x = []
    y = []

    for i in range(freq * freq):
        _x = (center_x + (random.random() * (range_x * 2.0) - range_x)) % width
        _y = (center_y + (random.random() * (range_y * 2.0) - range_y)) % height

        x.append(_x)
        y.append(_y)

    return x, y


def square_grid(freq=1.0, distrib=None, corners=False, center_x=0.0, center_y=0.0, range_x=1.0, range_y=1.0, width=1.0, height=1.0, **kwargs):
    """
    """

    x = []
    y = []

    # Keep a node in the center of the image, or pin to corner:
    drift_amount = .5 / freq

    if (freq % 2) == 0:
        drift = 0.0 if not corners else drift_amount

    else:
        drift = drift_amount if not corners else 0.0

    #
    for a in range(freq):
        for b in range(freq):
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


def spiral(freq=1.0, center_x=0.0, center_y=0.0, range_x=1.0, range_y=1.0, width=1.0, height=1.0, **kwargs):
    """
    """

    kink = random.random() * 5.0 - 2.5

    x = []
    y = []

    count = freq * freq

    for i in range(count):
        fract = i / count

        degrees = fract * 360.0 * math.radians(1) * kink

        x.append((center_x + math.sin(degrees) * fract * range_x) % width)
        y.append((center_y + math.cos(degrees) * fract * range_y) % height)

    return x, y


def circular(freq=1.0, distrib=1.0, center_x=0.0, center_y=0.0, range_x=1.0, range_y=1.0, width=1.0, height=1.0, generation=1, **kwargs):
    """
    """

    x = []
    y = []

    ring_count = freq
    dot_count = freq

    x.append(center_x)
    y.append(center_y)

    rotation = (1 / dot_count) * 360.0 * math.radians(1)

    kink = random.random() * 100 - 50

    for i in range(1, ring_count + 1):
        dist_fract = i / ring_count

        for j in range(1, dot_count + 1):
            rads = j * rotation

            if distrib == PointDistribution.circular:
                rads += rotation * .5 * i

            if distrib == PointDistribution.rotating:
                rads += rotation * dist_fract * kink

            x_point = (center_x + math.sin(rads) * dist_fract * range_x)
            y_point = (center_y + math.cos(rads) * dist_fract * range_y)

            x.append(x_point % width)
            y.append(y_point % height)

    return x, y
