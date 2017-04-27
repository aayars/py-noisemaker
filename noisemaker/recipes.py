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

    base = multires(int(random.random() * 2 + 2), [height, width, channels], octaves=int(random.random() * 1) + 1, spline_order=0, refract_range=random.random())
    stylized = effects.normalize(effects.color_map(base, tensor, horizontal=True, displacement=2.5)).eval()

    base2 = multires(int(random.random() * 4 + 2), [height, width, channels], octaves=int(random.random() * 3) + 2, spline_order=0, refract_range=random.random())
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

    # Shitty VHS Tracking
    grad = tf.maximum(basic([int(random.random() * 10) + 5, 1], [height, width, 1]) - .5, 0)
    grad *= grad
    grad = tf.image.convert_image_dtype(grad, tf.float32, saturate=True)
    grad = effects.normalize(grad).eval()

    identity = effects._xy_index(tensor).eval()
    identity[:,:,1] = identity[:,:,1] - grad[:,:,0] * width * .25

    combined = effects.blend(combined, basic(int(height * .75), [height, width, 1]), grad)

    combined = tf.gather_nd(combined, identity % width)
    combined = tf.image.convert_image_dtype(combined, tf.float32, saturate=True)

    return combined