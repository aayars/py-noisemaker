import random

import numpy as np
import tensorflow as tf

from noisemaker.constants import ValueMask


#: Hard-coded masks
Masks = {
    ValueMask.chess: {
        "shape": [2, 2, 1],
        "values": [[0, 1], [1, 0]]
    },

    ValueMask.waffle: {
        "shape": [2, 2, 1],
        "values": [[0, 1], [1, 1]]
    },

    ValueMask.h_hex: {
        "shape": [6, 4, 1],
        "values": [
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ]
    },

    ValueMask.v_hex: {
        "shape": [4, 6, 1],
        "values": [
            [0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0]
        ]
    },

    ValueMask.h_tri: {
        "shape": [4, 2, 1],
        "values": [
            [0, 1],
            [0, 0],
            [1, 0],
            [0, 0]
        ]
    },

    ValueMask.v_tri: {
        "shape": [2, 4, 1],
        "values": [
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ]
    },

    ValueMask.square: {
        "shape": [4, 4, 1],
        "values": [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ]
    },

    ValueMask.zero: {
        "shape": [6, 6, 1],
        "values": [
            [ 0, 0, 0, 0, 0, 0 ],
            [ 0, 1, 1, 1, 0, 0 ],
            [ 1, 0, 0, 1, 1, 0 ],
            [ 1, 0, 1, 0, 1, 0 ],
            [ 1, 1, 0, 0, 1, 0 ],
            [ 0, 1, 1, 1, 0, 0 ]
        ]
    },

    ValueMask.one: {
        "shape": [6, 6, 1],
        "values": [
            [ 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 1, 0, 0, 0 ],
            [ 0, 1, 1, 0, 0, 0 ],
            [ 0, 0, 1, 0, 0, 0 ],
            [ 0, 0, 1, 0, 0, 0 ],
            [ 0, 1, 1, 1, 0, 0 ]
        ]
    },

    ValueMask.two: {
        "shape": [6, 6, 1],
        "values": [
            [ 0, 0, 0, 0, 0, 0 ],
            [ 0, 1, 1, 1, 0, 0 ],
            [ 0, 0, 0, 0, 1, 0 ],
            [ 0, 1, 1, 1, 0, 0 ],
            [ 1, 0, 0, 0, 0, 0 ],
            [ 1, 1, 1, 1, 1, 0 ]
        ]
    },

    ValueMask.three: {
        "shape": [6, 6, 1],
        "values": [
            [ 0, 0, 0, 0, 0, 0 ],
            [ 1, 1, 1, 1, 0, 0 ],
            [ 0, 0, 0, 0, 1, 0 ],
            [ 0, 1, 1, 1, 0, 0 ],
            [ 0, 0, 0, 0, 1, 0 ],
            [ 1, 1, 1, 1, 0, 0 ]
        ]
    },

    ValueMask.four: {
        "shape": [6, 6, 1],
        "values": [
            [ 0, 0, 0, 0, 0, 0 ],
            [ 1, 0, 0, 1, 0, 0 ],
            [ 1, 0, 0, 1, 0, 0 ],
            [ 1, 1, 1, 1, 1, 0 ],
            [ 0, 0, 0, 1, 0, 0 ],
            [ 0, 0, 0, 1, 0, 0 ]
        ]
    },

    ValueMask.five: {
        "shape": [6, 6, 1],
        "values": [
            [ 0, 0, 0, 0, 0, 0 ],
            [ 1, 1, 1, 1, 0, 0 ],
            [ 1, 0, 0, 0, 0, 0 ],
            [ 1, 1, 1, 1, 0, 0 ],
            [ 0, 0, 0, 0, 1, 0 ],
            [ 1, 1, 1, 1, 0, 0 ]
        ]
    },

    ValueMask.six: {
        "shape": [6, 6, 1],
        "values": [
            [ 0, 0, 0, 0, 0, 0 ],
            [ 0, 1, 1, 1, 0, 0 ],
            [ 1, 0, 0, 0, 0, 0 ],
            [ 1, 1, 1, 1, 0, 0 ],
            [ 1, 0, 0, 0, 1, 0 ],
            [ 0, 1, 1, 1, 0, 0 ]
        ]
    },

    ValueMask.seven: {
        "shape": [6, 6, 1],
        "values": [
            [ 0, 0, 0, 0, 0, 0 ],
            [ 1, 1, 1, 1, 1, 0 ],
            [ 0, 0, 0, 0, 1, 0 ],
            [ 0, 0, 0, 1, 0, 0 ],
            [ 0, 0, 1, 0, 0, 0 ],
            [ 0, 1, 0, 0, 0, 0 ]
        ]
    },

    ValueMask.eight: {
        "shape": [6, 6, 1],
        "values": [
            [ 0, 0, 0, 0, 0, 0 ],
            [ 0, 1, 1, 1, 0, 0 ],
            [ 1, 0, 0, 0, 1, 0 ],
            [ 0, 1, 1, 1, 0, 0 ],
            [ 1, 0, 0, 0, 1, 0 ],
            [ 0, 1, 1, 1, 0, 0 ]
        ]
    },

    ValueMask.nine: {
        "shape": [6, 6, 1],
        "values": [
            [ 0, 0, 0, 0, 0, 0 ],
            [ 0, 1, 1, 1, 0, 0 ],
            [ 1, 0, 0, 0, 1, 0 ],
            [ 0, 1, 1, 1, 1, 0 ],
            [ 0, 0, 0, 0, 1, 0 ],
            [ 0, 1, 1, 1, 0, 0 ]
        ]
    },

    ValueMask.a: {
        "shape": [6, 6, 1],
        "values": [
            [ 0, 0, 0, 0, 0, 0 ],
            [ 0, 1, 1, 1, 0, 0 ],
            [ 1, 0, 0, 0, 1, 0 ],
            [ 1, 1, 1, 1, 1, 0 ],
            [ 1, 0, 0, 0, 1, 0 ],
            [ 1, 0, 0, 0, 1, 0 ]
        ]
    },

    ValueMask.b: {
        "shape": [6, 6, 1],
        "values": [
            [ 0, 0, 0, 0, 0, 0 ],
            [ 1, 1, 1, 1, 0, 0 ],
            [ 1, 0, 0, 0, 1, 0 ],
            [ 1, 1, 1, 1, 0, 0 ],
            [ 1, 0, 0, 0, 1, 0 ],
            [ 1, 1, 1, 1, 0, 0 ]
        ]
    },

    ValueMask.c: {
        "shape": [6, 6, 1],
        "values": [
            [ 0, 0, 0, 0, 0, 0 ],
            [ 0, 1, 1, 1, 1, 0 ],
            [ 1, 0, 0, 0, 0, 0 ],
            [ 1, 0, 0, 0, 0, 0 ],
            [ 1, 0, 0, 0, 0, 0 ],
            [ 0, 1, 1, 1, 1, 0 ]
        ]
    },

    ValueMask.d: {
        "shape": [6, 6, 1],
        "values": [
            [ 0, 0, 0, 0, 0, 0 ],
            [ 1, 1, 1, 1, 0, 0 ],
            [ 1, 0, 0, 0, 1, 0 ],
            [ 1, 0, 0, 0, 1, 0 ],
            [ 1, 0, 0, 0, 1, 0 ],
            [ 1, 1, 1, 1, 0, 0 ]
        ]
    },

    ValueMask.e: {
        "shape": [6, 6, 1],
        "values": [
            [ 0, 0, 0, 0, 0, 0 ],
            [ 1, 1, 1, 1, 1, 0 ],
            [ 1, 0, 0, 0, 0, 0 ],
            [ 1, 1, 1, 1, 0, 0 ],
            [ 1, 0, 0, 0, 0, 0 ],
            [ 1, 1, 1, 1, 1, 0 ]
        ]
    },

    ValueMask.f: {
        "shape": [6, 6, 1],
        "values": [
            [ 0, 0, 0, 0, 0, 0 ],
            [ 1, 1, 1, 1, 1, 0 ],
            [ 1, 0, 0, 0, 0, 0 ],
            [ 1, 1, 1, 1, 0, 0 ],
            [ 1, 0, 0, 0, 0, 0 ],
            [ 1, 0, 0, 0, 0, 0 ]
        ]
    },

    ValueMask.tromino_i: {
        "shape": [4, 4, 1],
        "values": [
            [ 1, 0, 0, 0 ],
            [ 1, 0, 0, 0 ],
            [ 1, 0, 0, 0 ],
            [ 1, 0, 0, 0 ]
        ]
    },

    ValueMask.tromino_l: {
        "shape": [4, 4, 1],
        "values": [
            [ 0, 0, 0, 0 ],
            [ 0, 1, 0, 0 ],
            [ 0, 1, 0, 0 ],
            [ 0, 1, 1, 0 ]
        ]
    },

    ValueMask.tromino_o: {
        "shape": [4, 4, 1],
        "values": [
            [ 0, 0, 0, 0 ],
            [ 0, 1, 1, 0 ],
            [ 0, 1, 1, 0 ],
            [ 0, 0, 0, 0 ]
        ]
    },

    ValueMask.tromino_s: {
        "shape": [4, 4, 1],
        "values": [
            [ 0, 0, 0, 0 ],
            [ 0, 1, 1, 0 ],
            [ 1, 1, 0, 0 ],
            [ 0, 0, 0, 0 ]
        ]
    },

    ValueMask.halftone_0: {
        "shape": [4, 4, 1],
        "values": [
            [ 0, 0, 0, 0 ],
            [ 0, 0, 0, 0 ],
            [ 0, 0, 0, 0 ],
            [ 0, 0, 0, 0 ]
        ]
    },

    ValueMask.halftone_1: {
        "shape": [4, 4, 1],
        "values": [
            [ 0, 0, 0, 0 ],
            [ 0, 1, 0, 0 ],
            [ 0, 0, 0, 0 ],
            [ 0, 0, 0, 0 ]
        ]
    },

    ValueMask.halftone_2: {
        "shape": [4, 4, 1],
        "values": [
            [ 0, 0, 0, 0 ],
            [ 0, 1, 0, 0 ],
            [ 0, 0, 0, 0 ],
            [ 0, 0, 0, 1 ]
        ]
    },

    ValueMask.halftone_3: {
        "shape": [4, 4, 1],
        "values": [
            [ 0, 0, 0, 0 ],
            [ 0, 1, 0, 1 ],
            [ 0, 0, 0, 0 ],
            [ 0, 1, 0, 1 ],
        ]
    },

    ValueMask.halftone_4: {
        "shape": [4, 4, 1],
        "values": [
            [ 1, 0, 1, 0 ],
            [ 0, 1, 0, 1 ],
            [ 1, 0, 1, 0 ],
            [ 0, 1, 0, 1 ],
        ]
    },

    ValueMask.halftone_5: {
        "shape": [4, 4, 1],
        "values": [
            [ 0, 1, 0, 1 ],
            [ 1, 0, 1, 0 ],
            [ 0, 1, 0, 1 ],
            [ 1, 0, 1, 0 ]
        ]
    },

    ValueMask.halftone_6: {
        "shape": [4, 4, 1],
        "values": [
            [ 1, 1, 1, 1 ],
            [ 1, 0, 1, 0 ],
            [ 1, 1, 1, 1 ],
            [ 1, 0, 1, 0 ]
        ]
    },

    ValueMask.halftone_7: {
        "shape": [4, 4, 1],
        "values": [
            [ 1, 1, 1, 1 ],
            [ 1, 0, 1, 1 ],
            [ 1, 1, 1, 1 ],
            [ 1, 1, 1, 0 ]
        ]
    },

    ValueMask.halftone_8: {
        "shape": [4, 4, 1],
        "values": [
            [ 1, 1, 1, 1 ],
            [ 1, 0, 1, 1 ],
            [ 1, 1, 1, 1 ],
            [ 1, 1, 1, 1 ]
        ]
    },

    ValueMask.halftone_9: {
        "shape": [4, 4, 1],
        "values": [
            [ 1, 1, 1, 1 ],
            [ 1, 1, 1, 1 ],
            [ 1, 1, 1, 1 ],
            [ 1, 1, 1, 1 ]
        ]
    },

    ValueMask.lcd_0: {
        "shape": [5, 8, 1],
        "values": [
            [ 0, 1, 1, 0, 0 ],
            [ 1, 0, 0, 1, 0 ],
            [ 1, 0, 0, 1, 0 ],
            [ 0, 0, 0, 0, 0 ],
            [ 1, 0, 0, 1, 0 ],
            [ 1, 0, 0, 1, 0 ],
            [ 0, 1, 1, 0, 0 ],
            [ 0, 0, 0, 0, 0 ],
        ]
    },

    ValueMask.lcd_1: {
        "shape": [5, 8, 1],
        "values": [
            [ 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 1, 0 ],
            [ 0, 0, 0, 1, 0 ],
            [ 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 1, 0 ],
            [ 0, 0, 0, 1, 0 ],
            [ 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0 ],
        ]
    },

    ValueMask.lcd_2: {
        "shape": [5, 8, 1],
        "values": [
            [ 0, 1, 1, 0, 0 ],
            [ 0, 0, 0, 1, 0 ],
            [ 0, 0, 0, 1, 0 ],
            [ 0, 1, 1, 0, 0 ],
            [ 1, 0, 0, 0, 0 ],
            [ 1, 0, 0, 0, 0 ],
            [ 0, 1, 1, 0, 0 ],
            [ 0, 0, 0, 0, 0 ],
        ]
    },

    ValueMask.lcd_3: {
        "shape": [5, 8, 1],
        "values": [
            [ 0, 1, 1, 0, 0 ],
            [ 0, 0, 0, 1, 0 ],
            [ 0, 0, 0, 1, 0 ],
            [ 0, 1, 1, 0, 0 ],
            [ 0, 0, 0, 1, 0 ],
            [ 0, 0, 0, 1, 0 ],
            [ 0, 1, 1, 0, 0 ],
            [ 0, 0, 0, 0, 0 ],
        ]
    },

    ValueMask.lcd_4: {
        "shape": [5, 8, 1],
        "values": [
            [ 0, 0, 0, 0, 0 ],
            [ 1, 0, 0, 1, 0 ],
            [ 1, 0, 0, 1, 0 ],
            [ 0, 1, 1, 0, 0 ],
            [ 0, 0, 0, 1, 0 ],
            [ 0, 0, 0, 1, 0 ],
            [ 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0 ],
        ]
    },

    ValueMask.lcd_5: {
        "shape": [5, 8, 1],
        "values": [
            [ 0, 1, 1, 0, 0 ],
            [ 1, 0, 0, 0, 0 ],
            [ 1, 0, 0, 0, 0 ],
            [ 0, 1, 1, 0, 0 ],
            [ 0, 0, 0, 1, 0 ],
            [ 0, 0, 0, 1, 0 ],
            [ 0, 1, 1, 0, 0 ],
            [ 0, 0, 0, 0, 0 ],
        ]
    },

    ValueMask.lcd_6: {
        "shape": [5, 8, 1],
        "values": [
            [ 0, 0, 0, 0, 0 ],
            [ 1, 0, 0, 0, 0 ],
            [ 1, 0, 0, 0, 0 ],
            [ 0, 1, 1, 0, 0 ],
            [ 1, 0, 0, 1, 0 ],
            [ 1, 0, 0, 1, 0 ],
            [ 0, 1, 1, 0, 0 ],
            [ 0, 0, 0, 0, 0 ],
        ]
    },

    ValueMask.lcd_7: {
        "shape": [5, 8, 1],
        "values": [
            [ 0, 1, 1, 0, 0 ],
            [ 0, 0, 0, 1, 0 ],
            [ 0, 0, 0, 1, 0 ],
            [ 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 1, 0 ],
            [ 0, 0, 0, 1, 0 ],
            [ 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0 ],
        ]
    },

    ValueMask.lcd_8: {
        "shape": [5, 8, 1],
        "values": [
            [ 0, 1, 1, 0, 0 ],
            [ 1, 0, 0, 1, 0 ],
            [ 1, 0, 0, 1, 0 ],
            [ 0, 1, 1, 0, 0 ],
            [ 1, 0, 0, 1, 0 ],
            [ 1, 0, 0, 1, 0 ],
            [ 0, 1, 1, 0, 0 ],
            [ 0, 0, 0, 0, 0 ],
        ]
    },

    ValueMask.lcd_9: {
        "shape": [5, 8, 1],
        "values": [
            [ 0, 1, 1, 0, 0 ],
            [ 1, 0, 0, 1, 0 ],
            [ 1, 0, 0, 1, 0 ],
            [ 0, 1, 1, 0, 0 ],
            [ 0, 0, 0, 1, 0 ],
            [ 0, 0, 0, 1, 0 ],
            [ 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0 ],
        ]
    },
}


# Procedural masks, corresponding to keys in constants.ValueMask

def bake_procedural(mask, channel_shape, uv_noise=None, atlas=None, inverse=False):
    """
    """

    mask_function = globals().get(mask.name)
    mask_shape = globals().get("{0}_shape".format(mask.name), lambda: None)() or channel_shape

    mask_values = []

    uv_shape = [int(channel_shape[0] / mask_shape[0]) or 1, int(channel_shape[1] / mask_shape[1]) or 1]

    if uv_noise is None:
        uv_noise = np.random.uniform(size=uv_shape)

    sum = 0

    for y in range(channel_shape[0]):
        uv_y = int((y  / channel_shape[0]) * uv_shape[0])

        mask_row = []
        mask_values.append(mask_row)

        for x in range(channel_shape[1]):
            uv_x = int((x / channel_shape[1]) * uv_shape[1])

            pixel = mask_function(x=x, y=y, row=mask_row, shape=mask_shape, uv_x=uv_x, uv_y=uv_y, uv_noise=uv_noise,
                                  atlas=atlas) * 1.0

            if inverse:
                pixel = 1.0 - pixel

            mask_row.append(pixel)
            sum += pixel

    return mask_values, sum


def sparse_shape():
    return (5, 5)


def sparse(**kwargs):
    return 1 if random.random() < .15 else 0


def invaders_shape():
    return (random.randint(5, 7), random.randint(6, 12))


def invaders_square_shape():
    return (6, 6)


def invaders(**kwargs):
    return _invaders(**kwargs)


def invaders_square(**kwargs):
    return _invaders(**kwargs)


def white_bear(**kwargs):
    return _invaders(**kwargs)


def white_bear_shape():
    return (4, 4)


def _invaders(x, y, row, shape, **kwargs):
    # Inspired by http://www.complexification.net/gallery/machines/invaderfractal/
    height = shape[0]
    width = shape[1]

    if y % height == 0 or x % width == 0:
        return 0

    elif x % width > width / 2:
        return row[x - int(((x % width) - width / 2) * 2)]

    else:
        return random.randint(0, 1)


def matrix_shape():
    return (6, 4)


def matrix(x, y, row, shape, **kwargs):
    height = shape[0]
    width = shape[1]

    if y % height == 0 or x % width == 0:
        return 0

    return random.randint(0, 1)


def letters_shape():
    return (random.randint(3, 4) * 2 + 1, random.randint(3, 4) * 2 + 1)


def letters(x, y, row, shape, **kwargs):
    # Inspired by https://www.shadertoy.com/view/4lscz8
    height = shape[0]
    width = shape[1]

    if any(n == 0 for n in (x % width, y % height)):
        return 0

    if any(n == 1 for n in (width - (x % width), height - (y % height))):
        return 0

    if all(n % 2 == 0 for n in (x % width, y % height)):
        return 0

    if x % 2 == 0 or y % 2 == 0:
        return random.random() > .25

    return random.random() > .75


def iching_shape():
    return (14, 8)


def iching(x, y, row, shape, **kwargs):
    height = shape[0]
    width = shape[1]

    if any(n == 0 for n in (x % width, y % height)):
        return 0

    if any(n == 1 for n in (width - (x % width), height - (y % height))):
        return 0

    if y % 2 == 0:
        return 0

    if x % 2 == 1 and x % width not in (3, 4):
        return 1

    if x % 2 == 0:
        return row[x - 1]

    return random.randint(0, 1)


def ideogram_shape():
    return (random.randint(4, 6) * 2, ) * 2


def ideogram(x, y, row, shape, **kwargs):
    height = shape[0]
    width = shape[1]

    if any(n == 0 for n in (x % width, y % height)):
        return 0

    if any(n == 1 for n in (width - (x % width), height - (y % height))):
        return 0

    if all(n % 2 == 1 for n in (x % width, y % height)):
        return 0

    return random.random() > .5


def script_shape():
    return (random.randint(7, 9), random.randint(12, 24))


def script(x, y, row, shape, **kwargs):
    height = shape[0]
    width = shape[1]

    x_step = x % width
    y_step = y % height

    if x > 0 and (x + y) % 2 == 1:
        return row[x - 1]

    if y_step == 0:
        return 0

    if y_step in (1, 3, 6):
        return random.random() > .25

    if y_step in (2, 4, 5):
        return random.random() > .9

    if x_step == 0:
        return 0

    if any(n == 0 for n in (width - x_step, height - y_step)):
        return 0

    if all(n % 2 == 0 for n in (x_step, y_step)):
        return 0

    if y_step == height - 1:
        return 0

    return random.random() > .5


def binary_shape():
    return (6, 6)


def binary(x, y, row, shape, uv_x, uv_y, uv_noise, **kwargs):
    glyph = Masks[ValueMask.zero] if uv_noise[uv_y][uv_x] < .5 else Masks[ValueMask.one]

    return glyph["values"][y % shape[0]][x % shape[1]]


def tromino_shape():
    return (4, 4)


def tromino(x, y, row, shape, uv_x, uv_y, uv_noise, **kwargs):
    atlas = [Masks[g]["values"] for g in Masks if g.name.startswith("tromino")]

    tex_x = x % shape[1]
    tex_y = y % shape[0]

    uv_value = uv_noise[uv_y][uv_x] * (len(atlas) - 1)
    uv_floor = int(uv_value)
    uv_fract = uv_value - uv_floor

    float2 = uv_noise[(uv_y + int(shape[0] * .5)) % shape[0]][uv_x]
    float3 = uv_noise[uv_y][(uv_x + int(shape[1] * .5)) % shape[1]]

    if uv_fract < .5:
        _x = tex_x
        tex_x = tex_y
        tex_y = _x

    if float2 < .5:
        tex_x = shape[1] - tex_x - 1

    if float3 < .5:
        tex_y = shape[0] - tex_y - 1

    return atlas[uv_floor][tex_x][tex_y]


def numeric_shape():
    return (6, 6)


def numeric(x, y, row, shape, uv_x, uv_y, uv_noise, **kwargs):
    return _glyph_from_atlas_range(x, y, shape, uv_x, uv_y, uv_noise, ValueMask.zero.value, ValueMask.nine.value)


def hex_shape():
    return (6, 6)


def hex(x, y, row, shape, uv_x, uv_y, uv_noise, **kwargs):
    return _glyph_from_atlas_range(x, y, shape, uv_x, uv_y, uv_noise, ValueMask.zero.value, ValueMask.f.value)


def truetype_shape():
    return (15, 15)


def truetype(x, y, row, shape, uv_x, uv_y, uv_noise, atlas, **kwargs):
    glyph = atlas[int(uv_noise[uv_y][uv_x] * (len(atlas) - 1))]

    return glyph[y % shape[0]][x % shape[1]]


def halftone_shape():
    return (4, 4)


def halftone(x, y, row, shape, uv_x, uv_y, uv_noise, **kwargs):
    return _glyph_from_atlas_range(x, y, shape, uv_x, uv_y, uv_noise, ValueMask.halftone_0.value, ValueMask.halftone_9.value)


def lcd_shape():
    return (8, 5)


def lcd(x, y, row, shape, uv_x, uv_y, uv_noise, **kwargs):
    return _glyph_from_atlas_range(x, y, shape, uv_x, uv_y, uv_noise, ValueMask.lcd_0.value, ValueMask.lcd_9.value)


def _glyph_from_atlas_range(x, y, shape, uv_x, uv_y, uv_noise, min_value, max_value):
    atlas = [Masks[g]["values"] for g in Masks if g.value >= min_value and g.value <= max_value]

    return atlas[int(uv_noise[uv_y][uv_x] * (len(atlas) - 1))][y % shape[0]][x % shape[1]]

