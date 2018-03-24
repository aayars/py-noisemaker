import random

import tensorflow as tf

from noisemaker.constants import ValueDistribution
from noisemaker.generators import basic, multires

import noisemaker.effects as effects


def post_process(tensor, freq=3, shape=None, with_glitch=False, with_vhs=False, with_crt=False, with_scan_error=False, with_snow=False, with_dither=False,
                 with_false_color=False, with_interference=False, with_frame=False, with_fibers=False, with_stray_hair=False, with_grime=False,
                 with_watermark=False):
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
    :param bool with_frame: Shitty instant camera effect
    :param bool with_false_color: Swap colors with basic noise
    :param bool with_interference: CRT-like moire effect
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

    if with_interference:
        tensor = interference(tensor, shape)

    if with_watermark:
        tensor = watermark(tensor, shape)

    if with_frame:
        tensor = frame(tensor, shape)

    if with_grime:
        tensor = grime(tensor, shape)

    if with_fibers:
        tensor = fibers(tensor, shape)

    if with_stray_hair:
        tensor = stray_hair(tensor, shape)
        tensor = stray_hair(tensor, shape)

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


def interference(tensor, shape):
    """
    """

    height, width, channels = shape

    value_shape = [height, width, 1]

    distortion = basic(2, value_shape, corners=True)

    scan_noise = basic([2, 1], [2, 1, 1])
    scan_noise = tf.tile(scan_noise, [random.randint(32, 128), width, 1])
    scan_noise = effects.resample(scan_noise, value_shape, spline_order=0)
    scan_noise = effects.refract(scan_noise, value_shape, 1, reference_x=distortion, reference_y=distortion)

    tensor = 1.0 - (1.0 - tensor) * scan_noise

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
    white_noise = effects.center_mask(white_noise, effects.refract(white_noise, value_shape, distortion_amount, reference_x=distortion), value_shape)

    white_noise2 = basic([int(height * .5), int(width * .25)], value_shape)
    white_noise2 = effects.center_mask(white_noise2, effects.refract(white_noise2, value_shape, distortion_amount, reference_x=distortion), value_shape)

    tensor = effects.blend_cosine(tensor, white_noise, white_noise2 * .25)

    scan_noise = tf.tile(basic([2, 1], [2, 1, 1]), [int(height * .333), width, 1])
    scan_noise = effects.resample(scan_noise, value_shape)
    scan_noise = effects.center_mask(scan_noise, effects.refract(scan_noise, value_shape, distortion_amount, reference_x=distortion), value_shape)
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


def false_color(tensor, shape, horizontal=False, displacement=.5, **basic_kwargs):
    """
    """

    clut = basic(2, shape, **basic_kwargs)

    return effects.normalize(effects.color_map(tensor, clut, shape, horizontal=horizontal, displacement=displacement))


def fibers(tensor, shape):
    """
    """

    value_shape = [shape[0], shape[1], 1]

    for i in range(4):
        mask = basic(4, value_shape,
                     with_worms=4,
                     worms_alpha=1,
                     worms_density=.05 + random.random() * .00125,
                     worms_duration=1,
                     worms_kink=random.randint(5, 10),
                     worms_stride=.75,
                     worms_stride_deviation=.125,
                     )

        brightness = basic(128, shape, saturation=2.0)

        tensor = effects.blend(tensor, brightness, mask * .5)

    return tensor


def stray_hair(tensor, shape):
    """
    """

    value_shape = [shape[0], shape[1], 1]

    mask = basic(4, value_shape,
                 with_worms=4,
                 worms_alpha=1,
                 worms_density=.0025 + random.random() * .00125,
                 worms_duration=random.randint(8, 16),
                 worms_kink=random.randint(5, 50),
                 worms_stride=.5,
                 worms_stride_deviation=.25,
                 )

    brightness = basic(32, value_shape)

    return effects.blend(tensor, brightness * .333, mask * .666)


def grime(tensor, shape):
    """
    """

    value_shape = [shape[0], shape[1], 1]

    mask = multires(5, value_shape, distrib="exp", octaves=8, refract_range=1.0, deriv=3, deriv_alpha=.5)

    dusty = effects.blend(tensor, .25, tf.square(mask) * .125)

    specks = basic([int(shape[0] * .25), int(shape[1] * .25)], value_shape, distrib="exp", refract_range=.1)
    specks = 1.0 - tf.sqrt(effects.normalize(tf.maximum(specks - .5, 0.0)))

    dusty = effects.blend(dusty, basic([shape[0], shape[1]], value_shape), .125) * specks

    return effects.blend(tensor, dusty, mask)


def frame(tensor, shape):
    """
    """

    half_shape = [int(shape[0] * .5), int(shape[1] * .5), shape[2]]
    half_value_shape = [half_shape[0], half_shape[1], 1]

    noise = multires(64, half_value_shape, octaves=8)

    black = tf.zeros(half_value_shape)
    white = tf.ones(half_value_shape)

    mask = effects.singularity(None, half_value_shape, 1, dist_func=3, inverse=True)
    mask = effects.normalize(mask + noise * .005)
    mask = effects.blend_layers(tf.sqrt(mask), half_value_shape, 0.0125, white, black, black, black)

    faded = effects._downsample(tensor, shape, half_shape)
    faded = tf.image.adjust_brightness(faded, .1)
    faded = tf.image.adjust_contrast(faded, .75)
    faded = effects.light_leak(faded, half_shape, .125)
    faded = effects.vignette(faded, half_shape, 0.05, .75)

    edge_texture = white * .9 + effects.shadow(noise, half_value_shape, 1.0) * .1

    out = effects.blend(faded, edge_texture, mask)
    out = effects.aberration(out, half_shape, .00666)
    out = grime(out, half_shape)

    out = tf.image.adjust_saturation(out, .5)
    out = tf.image.random_hue(out, .05)

    out = effects.resample(out, shape)
    out = stray_hair(out, shape)

    return out


def watermark(tensor, shape):
    """
    """

    value_shape = [int(shape[0] * .5), int(shape[1] * .5), 1]
    value_shape = [shape[0], shape[1], 1]

    mask = basic(240, value_shape, spline_order=0, distrib=ValueDistribution.ones, mask="numeric")

    mask = crt(mask, value_shape)

    mask = effects.warp(mask, value_shape, [2, 4], octaves=1, displacement=.5)

    mask *= tf.square(basic(2, value_shape))

    value_shape = [shape[0], shape[1], 1]

    brightness = basic(16, value_shape)

    return effects.blend(tensor, brightness, mask * .125)