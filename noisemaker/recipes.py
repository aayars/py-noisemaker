import math
import random

import numpy as np
import tensorflow as tf

from noisemaker.generators import basic, multires

import noisemaker.effects as effects


def glitch(tensor):
    """
    """

    shape = tf.shape(tensor).eval()
    height, width, channels = shape

    tensor = effects.normalize(tensor)

    base = multires(int(random.random() * 2 + 2), width, height, channels, octaves=int(random.random() * 1) + 1, spline_order=0, refract_range=random.random())
    stylized = effects.normalize(effects.color_map(base, tensor, horizontal=True, displacement=2.5)).eval()

    base2 = multires(int(random.random() * 4 + 2), width, height, channels, octaves=int(random.random() * 3) + 2, spline_order=0, refract_range=random.random())
    jpegged = effects.normalize(effects.color_map(base2, stylized, horizontal=True, displacement=2.5)).eval()
    jpegged = tf.image.convert_image_dtype(jpegged, tf.uint8, saturate=True)

    x_offset = int(random.random() * width * 2)
    channel = int(random.random() * channels)
    stylized[:,:,channel] = np.roll(stylized[:,:,channel], x_offset)

    data = tf.image.encode_jpeg(jpegged, quality=random.random() * 5 + 10)
    jpegged = tf.image.decode_jpeg(data)
    data = tf.image.encode_jpeg(jpegged, quality=random.random() * 5)
    jpegged = tf.image.decode_jpeg(data)
    jpegged = tf.image.convert_image_dtype(jpegged, tf.float32, saturate=True)
    jpegged = effects.normalize(effects.convolve(effects.ConvKernel.sharpen, effects.normalize(jpegged)))

    combined = effects.blend(jpegged, tf.multiply(stylized, 1.0), tf.maximum(base2 * 2 - 1, 0))
    combined = effects.blend(tensor, combined, tf.maximum(base * 2 - 1, 0))

    m = basic(12, width, height, 1)
    index = m
    index -= .5
    index = tf.maximum(index, 0)
    index *= index
    index = tf.image.convert_image_dtype(index, tf.float32, saturate=True)
    m *= index
    m = effects.normalize(m)
    m = m.eval()

    combined = combined.eval()

    noise = basic(int(height * .25), width, height, 1).eval()

    for y in range(height):
        amount = m[0,y,0]  # this is wrong

        if not amount:
            continue

        fract = np.minimum(amount * 3, 1.0)

        row = combined[y] * (1 - fract)
        noise_row = noise[y] * fract

        combined[y] = np.roll(noise_row + row, int(amount * width * .5), axis=0)

    combined = np.roll(combined, int(width * .25), axis=1)

    return combined
