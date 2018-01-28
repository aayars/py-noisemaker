import random

import tensorflow as tf

from noisemaker.constants import ValueDistribution
from noisemaker.generators import basic, multires

import noisemaker.effects as effects


def post_process(tensor, freq=3, shape=None, with_glitch=False, with_vhs=False, with_crt=False, with_scan_error=False, with_snow=False, with_dither=False,
                 with_false_color=False):
    """
    Apply complex post-processing recipes.

    :param Tensor tensor:
    :param int freq:
    :param list[int] shape:
    :param bool with_glitch: Glitch effect (Bit shit)
    :param bool with_vhs: VHS effect (Shitty tracking)
    :param bool with_crt: Vintage TV effect
    :param bool with_scan_error: Horizontal scan error
    :param float with_snow: Analog broadcast snow
    :param float with_dither: Per-pixel brightness jitter
    :param bool with_false_color: Swap colors
    :return: Tensor
    """

    if with_false_color:
        tensor = false_color(tensor, shape)

    if with_glitch:
        tensor = glitch(tensor, shape)

    if with_dither:
        tensor = dither(tensor, shape, with_dither)

    if with_snow:
        tensor = snow(tensor, shape, with_snow)

    if with_scan_error:
        tensor = scanline_error(tensor, shape)

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

    base = multires(2, shape, octaves=random.randint(2, 5), spline_order=0, refract_range=random.random())
    stylized = effects.normalize(effects.color_map(base, tensor, shape, horizontal=True, displacement=2.5))

    jpegged = effects.jpeg_decimate(effects.color_map(base, stylized, shape, horizontal=True, displacement=2.5), shape)

    # Offset a single color channel
    separated = [stylized[:, :, i] for i in range(channels)]
    x_index = (effects.row_index(shape) + random.randint(1, width)) % width
    index = tf.cast(tf.stack([effects.column_index(shape), x_index], 2), tf.int32)

    channel = random.randint(0, channels - 1)
    separated[channel] = effects.normalize(tf.gather_nd(separated[channel], index) % random.random())

    channel = random.randint(0, channels - 1)
    top, _ = tf.nn.top_k(effects.value_map(tensor, shape), k=width)
    separated[channel] += top

    stylized = tf.stack(separated, 2)

    combined = effects.blend(tf.multiply(stylized, 1.0), jpegged, base)
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

    tensor = effects.blend_cosine(tensor, white_noise, tf.reshape(grad, [height, width, 1]) * .75)

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

    value_shape = [height, width, 1]

    distortion = basic(3, value_shape)
    distortion_amount = .25

    white_noise = basic(int(height * .75), value_shape, spline_order=0) - .5
    white_noise = effects.center_mask(white_noise, effects.refract(white_noise, shape, distortion_amount, reference_x=distortion), shape)

    white_noise2 = basic([int(height * .5), int(width * .25)], value_shape)
    white_noise2 = effects.center_mask(white_noise2, effects.refract(white_noise2, shape, distortion_amount, reference_x=distortion), shape)

    tensor = effects.blend_cosine(tensor, white_noise, white_noise2 * .25)

    scan_noise = tf.tile(basic([2, 1], [2, 1, 1]), [int(height * .333), width, 1])
    scan_noise = effects.resample(scan_noise, shape)
    scan_noise = effects.center_mask(scan_noise, effects.refract(scan_noise, shape, distortion_amount, reference_x=distortion), shape)
    tensor = effects.blend_cosine(tensor, scan_noise, 0.25)

    if channels <= 2:
        return tensor

    tensor = tf.image.random_hue(tensor, .125)
    tensor = tf.image.adjust_saturation(tensor, 1.25)

    return tensor


def scanline_error(tensor, shape):
    """
    """

    height, width, channels = shape

    value_shape = [height, width, 1]
    error_line = tf.maximum(basic([int(height * .75), 1], value_shape, distrib=ValueDistribution.exp) - .5, 0)
    error_swerve = tf.maximum(basic([int(height * .01), 1], value_shape, distrib=ValueDistribution.exp) - .5, 0)

    error_line *= error_swerve

    error_swerve *= 2

    white_noise = basic([int(height * .75), 1], value_shape)
    white_noise = effects.blend(0, white_noise, error_swerve)

    error = error_line + white_noise

    y_index = effects.column_index(shape)
    x_index = (effects.row_index(shape) - tf.cast(effects.value_map(error, value_shape) * width * .025, tf.int32)) % width

    return tf.minimum(tf.gather_nd(tensor, tf.stack([y_index, x_index], 2)) + error_line * white_noise * 4, 1)


def snow(tensor, shape, amount):
    """
    """

    height, width, channels = shape

    white_noise_1 = basic([height, width], [height, width, 1], wavelet=True, refract_range=10)
    white_noise_2 = tf.maximum(basic([int(height * .75), int(width * .75)], [height, width, 1]) - (1 - amount), 0) * 2

    return effects.blend(tensor, white_noise_1, white_noise_2)


def dither(tensor, shape, amount):
    """
    """

    height, width, channels = shape

    white_noise = basic([height, width], [height, width, 1])

    return effects.blend(tensor, white_noise, amount)


def false_color(tensor, shape, horizontal=False, displacement=1.0, corners=True, **basic_kwargs):
    """
    """

    clut = basic(random.randint(2, 4), shape, corners=True, **basic_kwargs)

    return effects.color_map(tensor, clut, shape, horizontal=horizontal, displacement=displacement)