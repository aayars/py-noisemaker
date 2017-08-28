import math
import random

import tensorflow as tf

from noisemaker.generators import Distribution, basic, multires

import noisemaker.effects as effects


def post_process(tensor, shape, freq, with_glitch, with_vhs, with_crt, with_scan_error, with_snow, with_dither):
    """
    Apply complex post-processing recipes.

    :param Tensor tensor:
    :param list[int] shape:
    :param int freq:
    :param bool with_glitch: Glitch effect (Bit shit)
    :param bool with_vhs: VHS effect (Shitty tracking)
    :param bool with_crt: Vintage TV effect
    :param bool with_scan_error: Horizontal scan error
    :param float with_snow: Analog broadcast snow
    :param float with_dither: Per-pixel brightness jitter
    :return: Tensor
    """

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

    # tensor = pop(tensor, shape)

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

    channel = int(random.random() * channels)
    top, _ = tf.nn.top_k(effects.value_map(tensor, shape), k=width)
    separated[channel] += top

    stylized = tf.stack(separated, 2)

    combined = effects.blend_cosine(tf.multiply(stylized, 1.0), jpegged, tf.maximum(base2 * 2 - 1, 0))
    combined = effects.blend_cosine(tensor, combined, tf.maximum(base * 2 - 1, 0))

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
    error_line = tf.maximum(basic([int(height * .75), 1], value_shape, distrib=Distribution.exponential) - .5, 0)
    error_swerve = tf.maximum(basic([int(height * .01), 1], value_shape, distrib=Distribution.exponential) - .5, 0)

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


def pop(tensor, shape):
    freq = 2

    tensor = tf.image.random_hue(tensor, .5)

    tensor = effects.inner_tile(tensor, shape, freq)

    tensor = effects.posterize(tensor, 3)

    tensor = tensor % basic([freq, freq], shape, spline_order=0)

    # tensor = effects.normalize(tensor)

    # tensor = tf.image.adjust_brightness(tensor, 1.125)

    return tensor


def light_leak(tensor, shape, alpha=.25):
    """
    """

    x, y = effects.point_cloud(6, distrib=effects.PointDistribution.v_hex, shape=shape)

    leak = effects.voronoi(tensor, shape, diagram_type=effects.VoronoiDiagramType.color_regions, xy=(x, y, len(x)))
    leak = effects.wormhole(leak, shape, kink=1.0, input_stride=.25)

    leak = effects.bloom(leak, shape, 1.0)
    leak = effects.convolve(effects.ConvKernel.blur, leak, shape)
    leak = effects.convolve(effects.ConvKernel.blur, leak, shape)
    leak = effects.convolve(effects.ConvKernel.blur, leak, shape)

    leak = 1 - ((1 - tensor) * (1 - leak))

    leak = effects.center_mask(tensor, leak, shape)
    leak = effects.center_mask(tensor, leak, shape)

    return effects.blend(tensor, leak, alpha)