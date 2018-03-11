import numpy as np
import tensorflow as tf

from noisemaker.constants import ValueDistribution, ValueMask

import noisemaker.effects as effects
import noisemaker.masks as masks


def values(freq, shape, distrib=ValueDistribution.normal, corners=False, mask=None, mask_inverse=False,
           spline_order=3, seed=None, wavelet=False):
    """
    """

    initial_shape = freq + [shape[-1]]
    channel_shape = freq + [1]

    if isinstance(distrib, int):
        distrib = ValueDistribution(distrib)

    elif isinstance(distrib, str):
        distrib = ValueDistribution[distrib]

    if isinstance(mask, int):
        mask = ValueMask(mask)

    elif isinstance(mask, str):
        mask = ValueMask[mask]

    if distrib == ValueDistribution.ones:
        tensor = tf.ones(initial_shape)

    elif distrib == ValueDistribution.mids:
        tensor = tf.ones(initial_shape) * .5

    elif distrib == ValueDistribution.normal:
        tensor = tf.random_normal(initial_shape, seed=seed)

    elif distrib == ValueDistribution.uniform:
        tensor = tf.random_uniform(initial_shape, seed=seed)

    elif distrib == ValueDistribution.exp:
        tensor = tf.cast(tf.stack(np.random.exponential(size=initial_shape)), tf.float32)

    elif distrib == ValueDistribution.laplace:
        tensor = tf.cast(tf.stack(np.random.laplace(size=initial_shape)), tf.float32)

    elif distrib == ValueDistribution.lognormal:
        tensor = tf.cast(tf.stack(np.random.lognormal(size=initial_shape)), tf.float32)

    if mask:
        if mask in masks.Masks:
            mask_values = effects.expand_tile(tf.cast(masks.Masks[mask]["values"], tf.float32),
                                              masks.Masks[mask]["shape"],
                                              [channel_shape[0], channel_shape[1]])

        else:
            atlas = None

            if mask == ValueMask.truetype:
                from noisemaker.glyphs import load_glyphs

                atlas = load_glyphs(masks.truetype_shape())

                if not atlas:
                    mask = ValueMask.numeric  # Fall back to canned values

            mask_values, _ = masks.bake_procedural(mask, channel_shape, atlas=atlas, inverse=mask_inverse)

        tensor *= tf.reshape(mask_values, channel_shape)

    if wavelet:
        tensor = effects.wavelet(tensor, initial_shape)

    tensor = effects.resample(tensor, shape, spline_order=spline_order)

    if (not corners and (freq[0] % 2) == 0) or (corners and (freq[0] % 2) == 1):
        tensor = effects.offset(tensor, shape, x=int((shape[1] / freq[1]) * .5), y=int((shape[0] / freq[0]) * .5))

    return tensor


def basic(freq, shape, ridges=False, sin=0.0, wavelet=False, spline_order=3, seed=None,
          distrib=ValueDistribution.normal, corners=False, mask=None, mask_inverse=False, lattice_drift=0.0,
          rgb=False, hue_range=.125, hue_rotation=None, saturation=1.0, brightness_distrib=None, saturation_distrib=None,
          **post_process_args):
    """
    Generate a single layer of scaled noise.

    .. image:: images/gaussian.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param int|list[int] freq: Base noise frequency. Int, or list of ints for each spatial dimension
    :param list[int]: Shape of noise. For 2D noise, this is [height, width, channels]
    :param bool ridges: "Crease" at midpoint values: (1 - abs(n * 2 - 1))
    :param float sin: Apply sin function to noise basis
    :param bool wavelet: Maybe not wavelets this time?
    :param int spline_order: Spline point count. 0=Constant, 1=Linear, 2=Cosine, 3=Bicubic
    :param int|str|ValueDistribution distrib: Type of noise distribution. See :class:`ValueDistribution` enum
    :param bool corners: If True, pin values to corners instead of image center
    :param None|ValueMask mask:
    :param bool mask_inverse:
    :param float lattice_drift: Push away from underlying lattice
    :param int seed: Random seed for reproducible output. Ineffective with exp
    :param bool rgb: Disable HSV
    :param float hue_range: HSV hue range
    :param float|None hue_rotation: HSV hue bias
    :param float saturation: HSV saturation
    :param None|int|str|ValueDistribution saturation_distrib: Override ValueDistribution for saturation
    :param None|int|str|ValueDistribution brightness_distrib: Override ValueDistribution for brightness
    :return: Tensor

    Additional keyword args will be sent to :py:func:`noisemaker.effects.post_process`
    """

    if isinstance(freq, int):
        freq = effects.freq_for_shape(freq, shape)

    tensor = values(freq, shape, distrib=distrib, corners=corners, mask=mask, mask_inverse=mask_inverse,
                    spline_order=spline_order, seed=seed, wavelet=wavelet)

    if lattice_drift:
        displacement = lattice_drift / min(freq[0], freq[1])

        tensor = effects.refract(tensor, shape, displacement=displacement, warp_freq=freq, spline_order=spline_order)

    tensor = effects.post_process(tensor, shape, freq, spline_order=spline_order, **post_process_args)

    if shape[-1] == 3 and not rgb:
        if hue_rotation is None:
            hue_rotation = tf.random_normal([])

        h = (tensor[:, :, 0] * hue_range + hue_rotation) % 1.0

        if saturation_distrib:
            s = tf.squeeze(values(freq, [shape[0], shape[1], 1], distrib=saturation_distrib, corners=corners,
                                  mask=mask, mask_inverse=mask_inverse, spline_order=spline_order, seed=seed,
                                  wavelet=wavelet))

        else:
            s = tensor[:, :, 1]

        s *= saturation

        if brightness_distrib:
            v = tf.squeeze(values(freq, [shape[0], shape[1], 1], distrib=brightness_distrib, corners=corners,
                                  mask=mask, mask_inverse=mask_inverse, spline_order=spline_order, seed=seed,
                                  wavelet=wavelet))

        else:
            v = tensor[:, :, 2]

        if ridges:
            v = effects.crease(v)

        if sin:
            v = effects.normalize(tf.sin(sin * v))

        tensor = tf.image.hsv_to_rgb([tf.stack([h, s, v], 2)])[0]

    elif ridges:
        tensor = effects.crease(tensor)

    if sin and rgb:
        tensor = tf.sin(sin * tensor)

    return tensor


def multires(freq=3, shape=None, octaves=4, ridges=False, post_ridges=False, sin=0.0, wavelet=False, spline_order=3, seed=None,
             reflect_range=0.0, refract_range=0.0, reindex_range=0.0, distrib=ValueDistribution.normal, corners=False,
             mask=None, mask_inverse=False, deriv=False, deriv_func=0, deriv_alpha=1.0, lattice_drift=0.0,
             post_reindex_range=0.0, post_reflect_range=0.0, post_refract_range=0.0, post_deriv=False,
             with_reverb=None, reverb_iterations=1,
             rgb=False, hue_range=.125, hue_rotation=None, saturation=1.0, saturation_distrib=None, brightness_distrib=None,
             **post_process_args):
    """
    Generate multi-resolution value noise. For each octave: freq increases, amplitude decreases.

    .. image:: images/multires.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param int|list[int] freq: Bottom layer frequency. Int, or list of ints for each spatial dimension
    :param list[int]: Shape of noise. For 2D noise, this is [height, width, channels]
    :param int octaves: Octave count. Number of multi-res layers. Typically 1-8
    :param bool ridges: Per-octave "crease" at midpoint values: (1 - abs(n * 2 - 1))
    :param bool post_ridges: Post-reduce "crease" at midpoint values: (1 - abs(n * 2 - 1))
    :param float sin: Apply sin function to noise basis
    :param bool wavelet: Maybe not wavelets this time?
    :param int spline_order: Spline point count. 0=Constant, 1=Linear, 2=Cosine, 3=Bicubic
    :param int seed: Random seed for reproducible output. Ineffective with exponential
    :param float reflect_range: Per-octave derivative-based distort range (0..1+)
    :param float refract_range: Per-octave self-distort range (0..1+)
    :param float reindex_range: Per-octave self-reindexing range (0..1+)
    :param None|int with_reverb: Post-reduce tiled octave count
    :param int reverb_iterations: Re-reverberate N times
    :param int|ValueDistribution distrib: Type of noise distribution. See :class:`ValueDistribution` enum
    :param bool corners: If True, pin values to corners instead of image center
    :param None|ValueMask mask:
    :param bool mask_inverse:
    :param bool deriv: Extract derivatives from noise
    :param DistanceFunction|int deriv_func: Derivative distance function
    :param float deriv_alpha: Derivative alpha blending amount
    :param float lattice_drift: Push away from underlying lattice
    :param float post_reindex_range: Reduced self-reindexing range (0..1+)
    :param float post_reflect_range: Reduced derivative-based distort range (0..1+)
    :param float post_refract_range: Reduced self-distort range (0..1+)
    :param bool post_deriv: Reduced derivatives
    :param bool rgb: Disable HSV
    :param float hue_range: HSV hue range
    :param float|None hue_rotation: HSV hue bias
    :param float saturation: HSV saturation
    :param None|ValueDistribution saturation_distrib: Override ValueDistribution for HSV saturation
    :param None|ValueDistribution brightness_distrib: Override ValueDistribution for HSV brightness
    :return: Tensor

    Additional keyword args will be sent to :py:func:`noisemaker.effects.post_process`
    """

    tensor = tf.zeros(shape)

    if isinstance(freq, int):
        freq = effects.freq_for_shape(freq, shape)

    for octave in range(1, octaves + 1):
        multiplier = 2 ** octave

        base_freq = [int(f * .5 * multiplier) for f in freq]

        if all(base_freq[i] > shape[i] for i in range(len(base_freq))):
            break

        layer = basic(base_freq, shape, ridges=ridges, sin=sin, wavelet=wavelet, spline_order=spline_order, seed=seed,
                      reflect_range=reflect_range / multiplier, refract_range=refract_range / multiplier, reindex_range=reindex_range / multiplier,
                      distrib=distrib, corners=corners, mask=mask, mask_inverse=mask_inverse, deriv=deriv, deriv_func=deriv_func, deriv_alpha=deriv_alpha,
                      lattice_drift=lattice_drift, rgb=rgb, hue_range=hue_range, hue_rotation=hue_rotation, saturation=saturation,
                      brightness_distrib=brightness_distrib, saturation_distrib=saturation_distrib,
                      )

        tensor += layer / multiplier

    tensor = effects.post_process(tensor, shape, freq, ridges_hint=ridges and rgb, spline_order=spline_order,
                                  reindex_range=post_reindex_range, reflect_range=post_reflect_range, refract_range=post_refract_range,
                                  with_reverb=with_reverb, reverb_iterations=reverb_iterations,
                                  deriv=post_deriv, deriv_func=deriv_func, with_crease=post_ridges, **post_process_args)

    return tensor
