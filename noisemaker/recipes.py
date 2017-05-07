import math
import random

import tensorflow as tf

from noisemaker.generators import basic, multires

import noisemaker.effects as effects


def post_process(tensor, shape, with_glitch, with_vhs, with_crt):
    """
    Apply complex post-processing recipes.

    :param Tensor tensor:
    :param list[int] shape:
    :param bool with_glitch: Glitch effect (Bit shit)
    :param bool with_vhs: VHS effect (Shitty tracking)
    :param bool with_crt: Vintage TV effect
    :return: Tensor
    """

    if with_glitch:
        tensor = glitch(tensor, shape)

    if with_vhs:
        tensor = vhs(tensor, shape)

    if with_crt:
        tensor = crt(tensor, shape)

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
    x_index = (effects.row_index(shape) + int(random.random() * width)) % width
    index = tf.cast(tf.stack([effects.column_index(shape), x_index], 2), tf.int32)

    channel = int(random.random() * channels)
    separated[channel] = effects.normalize(tf.gather_nd(separated[channel], index) % random.random())
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

    x_index = effects.row_index(shape) - tf.cast(grad * width * .125 + (scan_noise * width * .25 * grad * grad), tf.int32)
    identity = tf.stack([effects.column_index(shape), x_index], 2) % width

    tensor = tf.gather_nd(tensor, identity)
    tensor = tf.image.convert_image_dtype(tensor, tf.float32, saturate=True)

    return tensor


def crt(tensor, shape):
    """
    Apply vintage CRT snow and scanlines.

    :param Tensor tensor:
    :param list[int] shape:
    """

    height, width, channels = shape

    distortion = basic(2, [height, width, 1])
    distortion_amount = .25

    white_noise = basic(int(height * .75), [height, width, 1], spline_order=0) - .5
    white_noise = effects.center_mask(white_noise, effects.refract(white_noise, shape, distortion_amount, reference=distortion), shape)

    white_noise2 = basic([int(height * .5), int(width * .25)], [height, width, 1], spline_order=3)
    white_noise2 = effects.center_mask(white_noise2, effects.refract(white_noise2, shape, distortion_amount, reference=distortion), shape)

    tensor = effects.blend(tensor, white_noise, white_noise2 * .25)

    scan_noise = tf.tile(basic([2, 1], [2, 1, 1]), [int(height * .333), width, 1])
    scan_noise = effects.resample(scan_noise, shape)
    scan_noise = effects.center_mask(scan_noise, effects.refract(scan_noise, shape, distortion_amount, reference=distortion), shape)
    tensor = effects.blend(tensor, scan_noise, 0.25)

    if channels <= 2:
        return tensor

    tensor = tf.image.random_hue(tensor, .125)
    tensor = tf.image.adjust_saturation(tensor, 1.25)
    # tensor = tf.image.adjust_contrast(tensor, 1.5)

    y_index = effects.column_index(shape)
    x_index = effects.row_index(shape)

    separated = []

    for i in range(channels):
        _x_index = (x_index + int(random.random() * 16 - 8)) % width

        separated.append(tf.gather_nd(tensor[:,:,i], tf.cast(tf.stack([y_index, _x_index], 2), tf.int32)))

    tensor = tf.stack(separated, 2)

    return tensor