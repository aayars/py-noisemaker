import math
import random

import numpy as np
import tensorflow as tf

from noisemaker.generators import gaussian, multires

import noisemaker.effects as effects


def glitch(freq, width, height, channels, clut=None):
    """
    """

    # noisemaker multires --freq 3 --octaves 2 --spline-order 0 --channels 1 --sort 0 --refract 5 --clut hello.png --clut-range .1 --clut-horizontal

    base = multires(freq, width, height, 1, octaves=3, spline_order=0, refract_range=10)

    tensor = effects.color_map(base, clut, horizontal=True, displacement=.5)

    sort_axis = 0

    tensor = effects.normalize(tensor).eval()
    # _sorted = np.flip(np.sort(tensor, axis=sort_axis), axis=sort_axis)

    x_offset = int(random.random() * width * 2)
    # y_offset = int((random.random() * 2 - 1) * len(tensor[:,:,0]))
    hue_offset = random.random() * 2 - 1

    colored = tf.image.adjust_hue(tensor, hue_offset).eval()
    # colored[:,:,0] = np.roll(colored[:,:,0], y_offset, 0)
    colored[:,:,0] = np.roll(colored[:,:,0], x_offset, 1)
    colored = tf.image.adjust_hue(colored, -hue_offset)
    # temp = np.rot90(colored[:,:,0])
    # temp = np.roll(temp, int(random.random() * offset))
    # colored[:,:,0] = np.rot90(temp, 3)

    colored = tf.image.convert_image_dtype(effects.normalize(colored), tf.uint8, saturate=True)
    data = tf.image.encode_jpeg(colored, quality=5)
    colored = tf.image.decode_jpeg(data)
    data = tf.image.encode_jpeg(colored, quality=1)
    colored = tf.image.decode_jpeg(data)
    colored = tf.image.convert_image_dtype(colored, tf.float32, saturate=True)
    colored = effects.convolve(effects.ConvKernel.sharpen, effects.normalize(colored))

    colored = tf.image.adjust_saturation(colored, 2.5)

    colored = (1 - base) * colored
    tensor = base * tensor

    # tensor = effects.normalize(tensor + colored)
    tensor = tf.maximum(tensor, colored)

    tensor = tensor.eval()
    tensor = np.roll(tensor, shift=int(width * .1), axis=1)

    return tensor
