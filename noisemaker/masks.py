import math
import random

from noisemaker.constants import ValueMask


#: Hard-coded masks
Masks = {
    ValueMask.chess: {
        "shape": [2, 2, 1],
        "values": [[0.0, 1.0], [1.0, 0.0]]
    },

    ValueMask.waffle: {
        "shape": [2, 2, 1],
        "values": [[0.0, 1.0], [1.0, 1.0]]
    },

    ValueMask.h_hex: {
        "shape": [6, 4, 1],
        "values": [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    },

    ValueMask.v_hex: {
        "shape": [4, 6, 1],
        "values": [
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ]
    },

    ValueMask.h_tri: {
        "shape": [4, 2, 1],
        "values": [
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0]
        ]
    },

    ValueMask.v_tri: {
        "shape": [2, 4, 1],
        "values": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ]
    },

    ValueMask.square: {
        "shape": [4, 4, 1],
        "values": [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ]
    },

}


# Procedural masks, also mapping to keys in the ValueMask enum.


def sparse(*args):
    return 1.0 if random.random() < .15 else 0.0


def invaders_shape():
    return (random.randint(5, 7), random.randint(6, 12))


def invaders_square_shape():
    return (6, 6)


def invaders(*args):
    return _invaders(*args)


def invaders_square(*args):
    return _invaders(*args)


def _invaders(x, y, row, shape, *args):
    # Inspired by http://www.complexification.net/gallery/machines/invaderfractal/
    height = shape[0]
    width = shape[1]

    if y % height == 0 or x % width == 0:
        return 0.0

    elif x % width > width / 2:
        return row[x - int(((x % width) - width / 2) * 2)]

    else:
        return random.randint(0, 1) * 1.0


def matrix_shape():
    return (6, 4)


def matrix(x, y, row, shape, *args):
    height = shape[0]
    width = shape[1]

    if y % height == 0 or x % width == 0:
        return 0.0

    return random.randint(0, 1) * 1.0


def letters_shape():
    return (random.randint(3, 4) * 2 + 1, random.randint(3, 4) * 2 + 1)


def letters(x, y, row, shape, *args):
    # Inspired by https://www.shadertoy.com/view/4lscz8
    height = shape[0]
    width = shape[1]

    if any(n == 0 for n in (x % width, y % height)):
        return 0.0

    if any(n == 1 for n in (width - (x % width), height - (y % height))):
        return 0.0

    if all(n % 2 == 0 for n in (x % width, y % height)):
        return 0.0

    if x % 2 == 0 or y % 2 == 0:
        return random.random() > .25

    return random.random() > .75


def iching_shape():
    return (14, 8)


def iching(x, y, row, shape, *args):
    height = shape[0]
    width = shape[1]

    if any(n == 0 for n in (x % width, y % height)):
        return 0.0

    if any(n == 1 for n in (width - (x % width), height - (y % height))):
        return 0.0

    if y % 2 == 0:
        return 0.0

    if x % 2 == 1 and x % width not in (3, 4):
        return 1.0

    if x % 2 == 0:
        return row[x - 1]

    return random.randint(0, 1)


def ideogram_shape():
    return (random.randint(4, 6) * 2, ) * 2


def ideogram(x, y, row, shape, *args):
    height = shape[0]
    width = shape[1]

    if any(n == 0 for n in (x % width, y % height)):
        return 0.0

    if any(n == 1 for n in (width - (x % width), height - (y % height))):
        return 0.0

    if all(n % 2 == 1 for n in (x % width, y % height)):
        return 0.0

    return random.random() > .5


def script_shape():
    return (random.randint(7, 9), random.randint(12, 24))


def script(x, y, row, shape, *args):
    height = shape[0]
    width = shape[1]

    x_step = x % width
    y_step = y % height

    if x > 0 and (x + y) % 2 == 1:
        return row[x - 1]

    if y_step == 0:
        return 0.0

    if y_step in (1, 3, 6):
        return random.random() > .25

    if y_step in (2, 4, 5):
        return random.random() > .9

    if x_step == 0:
        return 0.0

    if any(n == 0 for n in (width - x_step, height - y_step)):
        return 0.0

    if all(n % 2 == 0 for n in (x_step, y_step)):
        return 0.0

    if y_step == height - 1:
        return 0.0

    return random.random() > .5