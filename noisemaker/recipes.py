"""High-level effect recipes for Noisemaker. May call out to generators or effects."""

import random

import tensorflow as tf

from noisemaker.constants import PointDistribution, ValueDistribution, ValueMask
from noisemaker.generators import basic, multires
from noisemaker.points import point_cloud

import noisemaker.effects as effects
import noisemaker.masks as masks
import noisemaker.simplex as simplex


def post_process(tensor, freq=3, shape=None, with_glitch=False, with_vhs=False, with_crt=False, with_scan_error=False, with_snow=False, with_dither=False,
                 with_nebula=False, with_false_color=False, with_interference=False, with_frame=False, with_scratches=False, with_fibers=False,
                 with_stray_hair=False, with_grime=False, with_watermark=False, with_ticker=False, with_texture=False,
                 with_pre_spatter=False, with_spatter=False, with_clouds=False, time=0.0, simplex_displacement=1.0, **_):
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
    :param bool with_nebula: Add clouds
    :param bool with_false_color: Swap colors with basic noise
    :param bool with_interference: CRT-like moire effect
    :param bool with_watermark: Stylized digital watermark effect
    :param bool with_ticker: With spooky ticker effect
    :param bool with_scratches: Scratched film effect
    :param bool with_fibers: Old-timey paper fibers
    :param bool with_texture: Bumpy canvas
    :param bool with_pre_spatter: Spatter mask (early pass)
    :param bool with_spatter: Spatter mask
    :param bool with_clouds: Cloud cover
    :param float time: Input value for Z axis (simplex only)
    :return: Tensor
    """

    if with_pre_spatter:
        tensor = spatter(tensor, shape, time=time, simplex_displacement=simplex_displacement)

    if with_nebula:
        tensor = nebula(tensor, shape, time=time, simplex_displacement=simplex_displacement)

    if with_false_color:
        tensor = false_color(tensor, shape, time=time, simplex_displacement=simplex_displacement)

    if with_glitch:
        tensor = glitch(tensor, shape, time=time, simplex_displacement=simplex_displacement)

    if with_dither:
        tensor = dither(tensor, shape, with_dither, time=time, simplex_displacement=simplex_displacement)

    if with_snow:
        tensor = snow(tensor, shape, with_snow, time=time, simplex_displacement=simplex_displacement)

    if with_scan_error:
        tensor = scanline_error(tensor, shape, time=time, simplex_displacement=simplex_displacement)

    if with_vhs:
        tensor = vhs(tensor, shape, time=time, simplex_displacement=simplex_displacement)

    if with_crt:
        tensor = crt(tensor, shape, time=time, simplex_displacement=simplex_displacement)

    if with_interference:
        tensor = interference(tensor, shape, time=time, simplex_displacement=simplex_displacement)

    if with_watermark:
        tensor = watermark(tensor, shape, time=time, simplex_displacement=simplex_displacement)

    if with_frame:
        tensor = frame(tensor, shape, time=time, simplex_displacement=simplex_displacement)

    if with_grime:
        tensor = grime(tensor, shape, time=time, simplex_displacement=simplex_displacement)

    if with_fibers:
        tensor = fibers(tensor, shape, time=time, simplex_displacement=simplex_displacement)

    if with_scratches:
        tensor = scratches(tensor, shape, time=time, simplex_displacement=simplex_displacement)

    if with_texture:
        tensor = texture(tensor, shape, time=time, simplex_displacement=simplex_displacement)

    if with_ticker:
        tensor = spooky_ticker(tensor, shape)

    if with_stray_hair:
        tensor = stray_hair(tensor, shape, time=time, simplex_displacement=simplex_displacement)

    if with_spatter:
        tensor = spatter(tensor, shape, time=time, simplex_displacement=simplex_displacement)

    if with_clouds:
        tensor = clouds(tensor, shape, time=time, simplex_displacement=simplex_displacement)

    return tensor


def glitch(tensor, shape, time=0.0, simplex_displacement=1.0):
    """
    Apply a glitch effect.

    :param Tensor tensor:
    :param list[int] shape:
    :return: Tensor
    """

    height, width, channels = shape

    tensor = effects.normalize(tensor)

    base = multires(2, shape, time=time, simplex_displacement=simplex_displacement, distrib=ValueDistribution.simplex, octaves=random.randint(2, 5), spline_order=0, refract_range=random.random())
    stylized = effects.normalize(effects.color_map(base, tensor, shape, horizontal=True, displacement=2.5))

    jpegged = effects.color_map(base, stylized, shape, horizontal=True, displacement=2.5)

    if channels in (1, 3):
        jpegged = effects.jpeg_decimate(jpegged, shape)

    # Offset a single color channel
    separated = [stylized[:, :, i] for i in range(channels)]
    x_index = (effects.row_index(shape) + random.randint(1, width)) % width
    index = tf.cast(tf.stack([effects.column_index(shape), x_index], 2), tf.int32)

    channel = random.randint(0, channels - 1)
    separated[channel] = effects.normalize(tf.gather_nd(separated[channel], index) % random.random())

    stylized = tf.stack(separated, 2)

    combined = effects.blend(tf.multiply(stylized, 1.0), jpegged, base)
    combined = effects.blend(tensor, combined, tf.maximum(base * 2 - 1, 0))
    combined = effects.blend(combined, effects.pixel_sort(combined, shape), 1.0 - base)

    combined = tf.image.adjust_contrast(combined, 1.75)

    return combined


def vhs(tensor, shape, time=0.0, simplex_displacement=1.0):
    """
    Apply a bad VHS tracking effect.

    :param Tensor tensor:
    :param list[int] shape:
    :return: Tensor
    """

    height, width, channels = shape

    # Generate scan noise
    scan_noise = basic([int(height * .5) + 1, int(width * .05) + 1], [height, width, 1], time=time,
                       simplex_displacement=simplex_displacement, spline_order=1, distrib=ValueDistribution.simplex)

    # Create horizontal offsets
    grad = basic([int(random.random() * 10) + 5, 1], [height, width, 1], time=time,
                 simplex_displacement=simplex_displacement, distrib=ValueDistribution.simplex)
    grad = tf.maximum(grad - .5, 0)
    grad = tf.minimum(grad * 2, 1)

    x_index = effects.row_index(shape)
    x_index -= tf.squeeze(tf.cast(scan_noise * width * tf.square(grad), tf.int32))
    x_index = x_index % width

    tensor = effects.blend(tensor, scan_noise, grad)

    identity = tf.stack([effects.column_index(shape), x_index], 2)

    tensor = tf.gather_nd(tensor, identity)

    return tensor


def interference(tensor, shape, time=0.0, simplex_displacement=1.0):
    """
    """

    height, width, channels = shape

    value_shape = [height, width, 1]

    distortion = basic(2, value_shape, time=0.0, simplex_displacement=simplex_displacement, distribution=ValueDistribution.simplex, corners=True)

    scan_noise = basic([2, 1], [2, 1, 1], time=0.0, simplex_displacement=simplex_displacement, distribution=ValueDistribution.simplex)
    scan_noise = tf.tile(scan_noise, [random.randint(32, 128), width, 1])
    scan_noise = effects.resample(scan_noise, value_shape, spline_order=0)
    scan_noise = effects.refract(scan_noise, value_shape, 1, reference_x=distortion, reference_y=distortion)

    tensor = 1.0 - (1.0 - tensor) * scan_noise

    return tensor


def crt(tensor, shape, time=0.0, simplex_displacement=1.0):
    """
    Apply vintage CRT snow and scanlines.

    :param Tensor tensor:
    :param list[int] shape:
    """

    height, width, channels = shape

    value_shape = [height, width, 1]

    # CRT lens shape
    mask = tf.pow(effects.singularity(None, value_shape), 4)  # Power of 4 to obscure center pinch

    # Displacement values to make it wavy towards the edges
    distortion_x = basic(2, value_shape, time=time, simplex_displacement=simplex_displacement, distrib=ValueDistribution.simplex, spline_order=2) * mask
    distortion_y = basic(2, value_shape, time=time, simplex_displacement=simplex_displacement, distrib=ValueDistribution.simplex, spline_order=2) * mask

    # Horizontal scanlines
    scan_noise = tf.tile(basic([2, 1], [2, 1, 1], time=time, simplex_displacement=simplex_displacement, distrib=ValueDistribution.simplex, spline_order=0), [int(height * .125) or 1, width, 1])
    scan_noise = effects.resample(scan_noise, value_shape)

    distortion_amount = .125
    scan_noise = effects.refract(scan_noise, value_shape, distortion_amount,
                                 reference_x=distortion_x, reference_y=distortion_y, extend_range=False)

    tensor = effects.normalize(effects.blend(tensor, (tensor + scan_noise) * scan_noise, 0.05))

    if channels == 3:
        tensor = effects.aberration(tensor, shape, .0075 + random.random() * .0075)
        tensor = tf.image.random_hue(tensor, .125)
        tensor = tf.image.adjust_saturation(tensor, 1.25)

    tensor = tf.image.adjust_contrast(tensor, 1.25)

    tensor = effects.vignette(tensor, shape, brightness=0, alpha=random.random() * .175)

    return tensor


def scanline_error(tensor, shape, time=0.0, simplex_displacement=1.0):
    """
    """

    height, width, channels = shape

    value_shape = [height, width, 1]
    error_line = tf.maximum(basic([int(height * .75), 1], value_shape, time=time,
                                  simplex_displacement=simplex_displacement, distrib=ValueDistribution.simplex_exp) - .5, 0)
    error_swerve = tf.maximum(basic([int(height * .01), 1], value_shape, time=time,
                                    simplex_displacement=simplex_displacement, distrib=ValueDistribution.simplex_exp) - .5, 0)

    error_line *= error_swerve

    error_swerve *= 2

    white_noise = basic([int(height * .75), 1], value_shape, time=time,
                        simplex_displacement=simplex_displacement, distrib=ValueDistribution.simplex)
    white_noise = effects.blend(0, white_noise, error_swerve)

    error = error_line + white_noise

    y_index = effects.column_index(shape)
    x_index = (effects.row_index(shape) - tf.cast(effects.value_map(error, value_shape) * width * .025, tf.int32)) % width

    return tf.minimum(tf.gather_nd(tensor, tf.stack([y_index, x_index], 2)) + error_line * white_noise * 4, 1)


def snow(tensor, shape, amount, time=0.0, simplex_displacement=1.0):
    """
    """

    height, width, channels = shape

    static = basic([height, width], [height, width, 1], time=time, simplex_displacement=simplex_displacement,
                   distrib=ValueDistribution.simplex, spline_order=0)
    static_limiter = basic([height, width], [height, width, 1], time=time, simplex_displacement=simplex_displacement,
                           distrib=ValueDistribution.simplex_exp, spline_order=0) * amount

    return effects.blend(tensor, static, static_limiter)


def dither(tensor, shape, amount, time=0.0, simplex_displacement=1.0):
    """
    """

    height, width, channels = shape

    white_noise = basic([height, width], [height, width, 1], time=time, simplex_displacement=simplex_displacement,
                        distrib=ValueDistribution.simplex)

    return effects.blend(tensor, white_noise, amount)


def false_color(tensor, shape, horizontal=False, displacement=.5, time=0.0, simplex_displacement=1.0, **basic_kwargs):
    """
    """

    clut = basic(2, shape, time=time, simplex_displacement=simplex_displacement, distrib=ValueDistribution.simplex, **basic_kwargs)

    return effects.normalize(effects.color_map(tensor, clut, shape, horizontal=horizontal, displacement=displacement))


def fibers(tensor, shape, time=0.0, simplex_displacement=1.0):
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
                     time=time,
                     simplex_displacement=simplex_displacement,
                     distrib=ValueDistribution.simplex,
                     )

        brightness = basic(128, shape, time=time, simplex_displacement=simplex_displacement,
                           distrib=ValueDistribution.simplex, saturation=2.0)

        tensor = effects.blend(tensor, brightness, mask * .5)

    return tensor


def scratches(tensor, shape, time=0.0, simplex_displacement=1.0):
    """
    """

    value_shape = [shape[0], shape[1], 1]

    for i in range(4):
        mask = basic(random.randint(2, 4), value_shape,
                     with_worms=[1, 3][random.randint(0, 1)],
                     worms_alpha=1,
                     worms_density=.25 + random.random() * .25,
                     worms_duration=2 + random.random() * 2,
                     worms_kink=.125 + random.random() * .125,
                     worms_stride=.75,
                     worms_stride_deviation=.5,
                     time=time,
                     simplex_displacement=simplex_displacement,
                     distrib=ValueDistribution.simplex,
                     )

        mask -= basic(random.randint(2, 4), value_shape, time=time, simplex_displacement=simplex_displacement,
                      distrib=ValueDistribution.simplex) * 2.0

        mask = tf.maximum(mask, 0.0)

        tensor = tf.maximum(tensor, mask * 8.0)

        tensor = tf.minimum(tensor, 1.0)

    return tensor


def stray_hair(tensor, shape, time=0.0, simplex_displacement=1.0):
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
                 time=time,
                 simplex_displacement=simplex_displacement,
                 distrib=ValueDistribution.simplex,
                 )

    brightness = basic(32, value_shape, time=time, simplex_displacement=simplex_displacement,
                       distrib=ValueDistribution.simplex)

    return effects.blend(tensor, brightness * .333, mask * .666)


def grime(tensor, shape, time=0.0, simplex_displacement=1.0):
    """
    """

    value_shape = [shape[0], shape[1], 1]

    mask = multires(5, value_shape, time=time, simplex_displacement=simplex_displacement,
                    distrib=ValueDistribution.simplex_exp, octaves=8, refract_range=1.0, deriv=3, deriv_alpha=.5)

    dusty = effects.blend(tensor, .25, tf.square(mask) * .125)

    specks = basic([int(shape[0] * .25), int(shape[1] * .25)], value_shape, time=time,
                   simplex_displacement=simplex_displacement, distrib=ValueDistribution.simplex_exp, refract_range=.1)
    specks = 1.0 - tf.sqrt(effects.normalize(tf.maximum(specks - .5, 0.0)))

    dusty = effects.blend(dusty, basic([shape[0], shape[1]], value_shape, time=time,
                          simplex_displacement=simplex_displacement, distrib=ValueDistribution.simplex), .125) * specks

    return effects.blend(tensor, dusty, mask)


def frame(tensor, shape, time=0.0, simplex_displacement=1.0):
    """
    """

    half_shape = [int(shape[0] * .5), int(shape[1] * .5), shape[2]]
    half_value_shape = [half_shape[0], half_shape[1], 1]

    noise = multires(64, half_value_shape, time=time, simplex_displacement=simplex_displacement, distrib=ValueDistribution.simplex, octaves=8)

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

    out = scratches(out, shape)

    out = stray_hair(out, shape)

    return out


def texture(tensor, shape, time=0.0, simplex_displacement=1.0):
    """
    """

    value_shape = [shape[0], shape[1], 1]

    noise = multires(64, value_shape, time=time, simplex_displacement=simplex_displacement,
                     distrib=ValueDistribution.simplex, octaves=8, ridges=True)

    return tensor * (tf.ones(value_shape) * .95 + effects.shadow(noise, value_shape, 1.0) * .05)


def watermark(tensor, shape, time=0.0, simplex_displacement=1.0):
    """
    """

    value_shape = [int(shape[0] * .5), int(shape[1] * .5), 1]
    value_shape = [shape[0], shape[1], 1]

    mask = basic(240, value_shape, spline_order=0, distrib=ValueDistribution.ones, mask="numeric")

    mask = crt(mask, value_shape)

    mask = effects.warp(mask, value_shape, [2, 4], octaves=1, displacement=.5, time=time,
                        simplex_displacement=simplex_displacement)

    mask *= tf.square(basic(2, value_shape, time=time, simplex_displacement=simplex_displacement,
                            distrib=ValueDistribution.simplex))

    value_shape = [shape[0], shape[1], 1]

    brightness = basic(16, value_shape, time=time, simplex_displacement=simplex_displacement,
                       distrib=ValueDistribution.simplex)

    return effects.blend(tensor, brightness, mask * .125)


def spooky_ticker(tensor, shape):
    """
    """

    if random.random() > .75:
        tensor = on_screen_display(tensor, shape)

    _masks = [
        ValueMask.arecibo_nucleotide,
        ValueMask.arecibo_num,
        ValueMask.bank_ocr,
        ValueMask.bar_code,
        ValueMask.bar_code_short,
        ValueMask.emoji,
        ValueMask.fat_lcd_hex,
        ValueMask.hex,
        ValueMask.iching,
        ValueMask.ideogram,
        ValueMask.invaders,
        ValueMask.lcd,
        ValueMask.letters,
        ValueMask.matrix,
        ValueMask.numeric,
        ValueMask.script,
        ValueMask.white_bear,
    ]

    bottom_padding = 2

    rendered_mask = tf.zeros(shape)

    for _ in range(random.randint(1, 3)):
        mask = _masks[random.randint(0, len(_masks) - 1)]
        mask_shape = masks.mask_shape(mask)

        multiplier = 1 if mask != ValueMask.script and (mask_shape[1] == 1 or mask_shape[1] >= 10) else 2

        width = int(shape[1] / multiplier) or 1
        width = mask_shape[1] * int(width / mask_shape[1])  # Make sure the mask divides evenly into width

        freq = [mask_shape[0], width]

        this_mask = basic(freq, [mask_shape[0], width, 1], corners=True, spline_order=0, distrib=ValueDistribution.ones, mask=mask)

        this_mask = effects.resample(this_mask, [mask_shape[0] * multiplier, shape[1]], spline_order=1)

        rendered_mask += tf.pad(this_mask, tf.stack([[shape[0] - mask_shape[0] * multiplier - bottom_padding, bottom_padding], [0, 0], [0, 0]]))

        bottom_padding += mask_shape[0] * multiplier + 2

    alpha = .5 + random.random() * .25

    # shadow
    tensor = effects.blend(tensor, tensor * 1.0 - effects.offset(rendered_mask, shape, -1, -1), alpha * .333)

    return effects.blend(tensor, tf.maximum(rendered_mask, tensor), alpha)


def on_screen_display(tensor, shape):
    glyph_count = random.randint(3, 6)

    _masks = [
        ValueMask.bank_ocr,
        ValueMask.hex,
        ValueMask.numeric,
    ]

    mask = _masks[random.randint(0, len(_masks) - 1)]
    mask_shape = masks.mask_shape(mask)

    width = int(shape[1] / 24)

    width = mask_shape[1] * int(width / mask_shape[1])  # Make sure the mask divides evenly
    height = mask_shape[0] * int(width / mask_shape[1])

    width *= glyph_count

    freq = [mask_shape[0], mask_shape[1] * glyph_count]

    this_mask = basic(freq, [height, width, shape[2]], corners=True, spline_order=0, distrib=ValueDistribution.ones, mask=mask)

    rendered_mask = tf.pad(this_mask, tf.stack([[25, shape[0] - height - 25], [shape[1] - width - 25, 25], [0, 0]]))

    alpha = .5 + random.random() * .25

    return effects.blend(tensor, tf.maximum(rendered_mask, tensor), alpha)


def nebula(tensor, shape, time=0.0, simplex_displacement=1.0):
    overlay = multires(random.randint(2, 4), shape, time=time, simplex_displacement=simplex_displacement,
                       distrib=ValueDistribution.simplex_exp, ridges=True, octaves=6)

    overlay -= multires(random.randint(2, 4), shape, time=time, simplex_displacement=simplex_displacement,
                        distrib=ValueDistribution.simplex, ridges=True, octaves=4)

    overlay = tf.maximum(overlay, 0)

    return tf.maximum(tensor, overlay * .25)


def spatter(tensor, shape, time=0.0, simplex_displacement=1.0):
    """
    """

    value_shape = [shape[0], shape[1], 1]

    # Generate a smear
    smear = multires(random.randint(2, 4), value_shape, time=time,
                     simplex_displacement=simplex_displacement, distrib=ValueDistribution.simplex_exp,
                     ridges=True, octaves=6, spline_order=3)

    smear = effects.warp(smear, value_shape, [random.randint(2, 3), random.randint(1, 3)],
                         octaves=random.randint(1, 2), displacement=.5 + random.random() * .5,
                         spline_order=3, time=time, simplex_displacement=simplex_displacement)

    # Add spatter dots
    smear = tf.maximum(smear, multires(random.randint(25, 50), value_shape, time=time,
                                       simplex_displacement=simplex_displacement, distrib=ValueDistribution.simplex_exp,
                                       post_brightness=-.25, post_contrast=4, octaves=4, spline_order=1))

    smear = tf.maximum(smear, multires(random.randint(200, 250), value_shape, time=time,
                                       simplex_displacement=simplex_displacement, distrib=ValueDistribution.simplex_exp,
                                       post_brightness=-.25, post_contrast=4, octaves=4, spline_order=1))

    # Remove some of it
    smear = tf.maximum(0.0, smear - multires(random.randint(2, 3), value_shape, time=time,
                                             simplex_displacement=simplex_displacement, distrib=ValueDistribution.simplex_exp,
                                             ridges=True, octaves=3, spline_order=1) * .75)

    #
    if shape[2] == 3:
        splash = tf.image.random_hue(tf.ones(shape) * tf.stack([.875, 0.125, 0.125]), .5)

    else:
        splash = tf.zeros(shape)

    return effects.blend_layers(effects.normalize(smear), shape, .005, tensor, splash)


def clouds(tensor, shape, time=0.0, simplex_displacement=1.0):
    """Top-down cloud cover effect"""

    pre_shape = [int(shape[0] * .25) or 1, int(shape[1] * .25) or 1, 1]

    control_kwargs = {
        "freq": random.randint(2, 4),
        "lattice_drift": 1,
        "octaves": 8,
        "ridges": True,
        "simplex_displacement": simplex_displacement,
        "shape": pre_shape,
        "time": time,
        "warp_freq": 3,
        "warp_range": .125,
        "warp_octaves": 2,
        "distrib": ValueDistribution.simplex,
    }

    control = multires(**control_kwargs)

    layer_0 = tf.ones(pre_shape)
    layer_1 = tf.zeros(pre_shape)

    combined = effects.blend_layers(control, pre_shape, 1.0, layer_0, layer_1)

    shadow = effects.offset(combined, pre_shape, random.randint(-15, 15), random.randint(-15, 15))
    shadow = tf.minimum(shadow * 2.5, 1.0)

    for _ in range(3):
        shadow = effects.convolve(ValueMask.conv2d_blur, shadow, pre_shape)

    post_shape = [shape[0], shape[1], 1]

    shadow = effects.resample(shadow, post_shape)
    combined = effects.resample(combined, post_shape)

    tensor = effects.blend(tensor, tf.zeros(shape), shadow * .75)
    tensor = effects.blend(tensor, tf.ones(shape), combined)

    tensor = effects.shadow(tensor, shape, alpha=.5)

    return tensor
