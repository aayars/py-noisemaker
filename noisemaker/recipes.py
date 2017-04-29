import math
import random

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
    stylized = effects.normalize(effects.color_map(base, tensor, horizontal=True, displacement=2.5))

    base2 = multires(int(random.random() * 4 + 2), [height, width, channels], octaves=int(random.random() * 3) + 2, spline_order=0, refract_range=random.random())
    jpegged = effects.normalize(effects.color_map(base2, stylized, horizontal=True, displacement=2.5))
    jpegged = tf.image.convert_image_dtype(jpegged, tf.uint8, saturate=True)

    x_index = (effects._row_index(tensor) + int(random.random() * width * 2)) % width
    separated = [stylized[:,:,i] for i in range(channels)]

    channel = int(random.random() * channels)
    identity = tf.cast(tf.stack([effects._column_index(tensor), x_index], 2), tf.int32) % width
    separated[channel] = effects.normalize(tf.gather_nd(separated[channel], identity) % random.random() * .5)

    stylized = tf.stack(separated, 2)

    data = tf.image.encode_jpeg(jpegged, quality=random.random() * 5 + 10)
    jpegged = tf.image.decode_jpeg(data)
    data = tf.image.encode_jpeg(jpegged, quality=random.random() * 5)
    jpegged = tf.image.decode_jpeg(data)
    jpegged = tf.image.convert_image_dtype(jpegged, tf.float32, saturate=True)

    combined = effects.blend(tf.multiply(stylized, 1.0), jpegged, tf.maximum(base2 * 2 - 1, 0))
    combined = effects.blend(combined, tensor, tf.maximum(base * 2 - 1, 0))

    return combined


def vhs(tensor):
    """
    """

    shape = tf.shape(tensor).eval()
    height, width, channels = shape

    scan_noise = tf.reshape(basic([int(height * .5) + 1, int(width * .01) + 1], [height, width, 1]), [height, width])
    white_noise = basic([int(height * .5) + 1, int(width * .1) + 1], [height, width, 1], spline_order=0)

    # Create horizontal offsets
    grad = tf.maximum(basic([int(random.random() * 10) + 5, 1], [height, width, 1]) - .5, 0)
    grad *= grad
    grad = tf.image.convert_image_dtype(grad, tf.float32, saturate=True)
    grad = effects.normalize(grad)
    grad = tf.reshape(grad, [height, width])

    tensor = effects.blend(tensor, white_noise, tf.reshape(grad, [height, width, 1]) * .75)

    x_index = effects._row_index(tensor) - grad * width * .25 + (scan_noise * width * .5 * grad * grad)
    identity = tf.cast(tf.stack([effects._column_index(tensor), x_index], 2), tf.int32) % width

    tensor = tf.gather_nd(tensor, identity)
    tensor = tf.image.convert_image_dtype(tensor, tf.float32, saturate=True)

    return tensor