import math
import random

import tensorflow as tf

from noisemaker.generators import basic, multires

import noisemaker.effects as effects


def post_process(tensor, shape, with_glitch, with_vhs):
    """
    Apply complex post-processing recipes.

    :param Tensor tensor:
    :param list[int] shape:
    :param bool with_glitch: Glitch effect (Bit shit)
    :param bool with_vhs: VHS effect (Shitty tracking)
    :return: Tensor
    """

    if with_glitch:
        tensor = glitch(tensor, shape)

    if with_vhs:
        tensor = vhs(tensor, shape)

    return tensor


def glitch(tensor, shape):
    """
    Apply a glitch effect.

    :param Tensor tensor:
    :param list[int] shape:
    :return: Tensor
    """

    height, width, channels = shape

    tensor = effects.normalize(tensor)

    base = multires(2, [height, width, channels], octaves=int(random.random() * 2) + 1, spline_order=0, refract_range=random.random())
    stylized = effects.normalize(effects.color_map(base, tensor, shape, horizontal=True, displacement=2.5))

    base2 = multires(int(random.random() * 4 + 2), [height, width, channels], octaves=int(random.random() * 3) + 2, spline_order=0, refract_range=random.random())

    jpegged = effects.jpeg_decimate(effects.color_map(base2, stylized, shape, horizontal=True, displacement=2.5))

    # Offset a single color channel
    separated = [stylized[:,:,i] for i in range(channels)]
    x_index = (effects._row_index(tensor, shape) + int(random.random() * width)) % width
    offset_index = tf.cast(tf.stack([effects._column_index(tensor, shape), x_index], 2), tf.int32)

    channel = int(random.random() * channels)
    separated[channel] = effects.normalize(tf.gather_nd(separated[channel], offset_index) % random.random())
    stylized = tf.stack(separated, 2)

    combined = effects.blend(tf.multiply(stylized, 1.0), jpegged, tf.maximum(base2 * 2 - 1, 0))
    combined = effects.blend(tensor, combined, tf.maximum(base * 2 - 1, 0))

    return combined


def vhs(tensor, shape):
    """
    Apply a bad VHS tracking effect.

    :param Tensor tensor:
    :param list[int] shape:
    :return: Tensor
    """

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

    x_index = effects._row_index(tensor, shape) - tf.cast(grad * width * .25 + (scan_noise * width * .5 * grad * grad), tf.int32)
    identity = tf.stack([effects._column_index(tensor, shape), x_index], 2) % width

    tensor = tf.gather_nd(tensor, identity)
    tensor = tf.image.convert_image_dtype(tensor, tf.float32, saturate=True)

    return tensor