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
    return (random.randint(5, 7), random.randint(6,12))


def invaders_square_shape():
    return (random.randint(3, 5) * 2, ) * 2


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
