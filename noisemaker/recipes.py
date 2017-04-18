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

    base = multires(int(random.random() * 2 + 2), width, height, channels, octaves=int(random.random() * 1) + 1, spline_order=0, refract_range=random.random())
    stylized = effects.normalize(effects.color_map(base, tensor, horizontal=True, displacement=5)).eval()

    base2 = multires(int(random.random() * 4 + 2), width, height, channels, octaves=int(random.random() * 3) + 2, spline_order=0, refract_range=random.random())
    jpegged = effects.normalize(effects.color_map(base2, stylized, horizontal=True, displacement=5)).eval()
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

    combined = tf.maximum(jpegged, base)

    return tf.minimum(combined, stylized)
