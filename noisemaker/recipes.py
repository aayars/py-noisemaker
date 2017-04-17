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

    base = multires(2, width, height, channels, octaves=int(random.random() * 3) + 1, spline_order=0, refract_range=random.random())

    stylized = effects.normalize(effects.color_map(base, tensor, horizontal=True, displacement=5)).eval()
    jpegged = tf.image.convert_image_dtype(stylized, tf.uint8, saturate=True)

    x_offset = int(random.random() * width * 2)
    stylized[:,:,0] = np.roll(stylized[:,:,0], x_offset, axis=1)

    data = tf.image.encode_jpeg(jpegged, quality=random.random() * 5 + 10)
    jpegged = tf.image.decode_jpeg(data)
    data = tf.image.encode_jpeg(jpegged, quality=random.random() * 5)
    jpegged = tf.image.decode_jpeg(data)
    jpegged = tf.image.convert_image_dtype(jpegged, tf.float32, saturate=True)

    jpegged = effects.normalize(effects.convolve(effects.ConvKernel.sharpen, effects.normalize(jpegged)))

    return tf.maximum(jpegged * base * 2, stylized)


    tensor = tensor.eval()
    tensor = np.roll(tensor, shift=int(width * .1), axis=1)

    return tensor
