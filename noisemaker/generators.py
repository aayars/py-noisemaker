"""Low-level value noise generators for Noisemaker"""

import random
import re
import string

import numpy as np
import tensorflow as tf

from noisemaker.constants import OctaveBlending, ValueDistribution, ValueMask

import noisemaker.effects as effects
import noisemaker.fastnoise as fastnoise
import noisemaker.masks as masks
import noisemaker.simplex as simplex


def set_seed(seed):
    """
    """

    if seed is not None:
        random.seed(seed)

        np.random.seed(seed)

        tf.random.set_seed(seed)

        simplex._seed = seed


def values(freq, shape, distrib=ValueDistribution.normal, corners=False,
           mask=None, mask_inverse=False, mask_static=False,
           spline_order=3, wavelet=False, time=0.0, speed=1.0):
    """
    """

    initial_shape = freq + [shape[-1]]

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
        tensor = tf.random.normal(initial_shape, mean=0.5, stddev=0.25)
        tensor = tf.math.minimum(tf.math.maximum(tensor, 0.0), 1.0)

    elif distrib == ValueDistribution.uniform:
        tensor = tf.random.uniform(initial_shape)

    elif distrib == ValueDistribution.exp:
        tensor = tf.cast(tf.stack(np.random.exponential(size=initial_shape)), tf.float32)

    elif distrib == ValueDistribution.laplace:
        tensor = tf.cast(tf.stack(np.random.laplace(size=initial_shape)), tf.float32)

    elif distrib == ValueDistribution.lognormal:
        tensor = tf.cast(tf.stack(np.random.lognormal(size=initial_shape)), tf.float32)

    elif distrib == ValueDistribution.column_index:
        tensor = tf.expand_dims(effects.normalize(tf.cast(effects.column_index(initial_shape), tf.float32)), -1) * tf.ones(initial_shape, tf.float32)

    elif distrib == ValueDistribution.row_index:
        tensor = tf.expand_dims(effects.normalize(tf.cast(effects.row_index(initial_shape), tf.float32)), -1) * tf.ones(initial_shape, tf.float32)

    elif distrib == ValueDistribution.simplex:
        tensor = simplex.simplex(initial_shape, time=time, speed=speed)

    elif distrib == ValueDistribution.simplex_exp:
        tensor = tf.math.pow(simplex.simplex(initial_shape, time=time, speed=speed), 4)

    elif distrib == ValueDistribution.fastnoise:
        tensor = fastnoise.fastnoise(shape, freq, seed=simplex._seed, time=time, speed=speed)

    elif distrib == ValueDistribution.fastnoise_exp:
        tensor = tf.math.pow(fastnoise.fastnoise(shape, freq, seed=simplex._seed, time=time, speed=speed), 4)

    else:
        raise ValueError("%s (%s) is not a ValueDistribution" % (distrib, type(distrib)))

    # Skip the below post-processing for fastnoise, since it's generated at a different size.
    if distrib not in (ValueDistribution.fastnoise, ValueDistribution.fastnoise_exp):
        if mask:
            atlas = None

            if mask == ValueMask.truetype:
                from noisemaker.glyphs import load_glyphs

                atlas = load_glyphs([15, 15, 1])

                if not atlas:
                    mask = ValueMask.alphanum_numeric  # Fall back to canned values

            elif ValueMask.is_procedural(mask):
                base_name = re.sub(r'_[a-z]+$', '', mask.name)

                if mask.name.endswith("_binary"):
                    atlas = [masks.Masks[ValueMask[f"{base_name}_0"]], masks.Masks[ValueMask[f"{base_name}_1"]]]

                elif mask.name.endswith("_numeric"):
                    atlas = [masks.Masks[ValueMask[f"{base_name}_{i}"]] for i in string.digits]

                elif mask.name.endswith("_hex"):
                    atlas = [masks.Masks[g] for g in masks.Masks if re.match(f"^{base_name}_[0-9a-f]$", g.name)]

                else:
                    atlas = [masks.Masks[g] for g in masks.Masks
                                 if g.name.startswith(f"{mask.name}_") and not callable(masks.Masks[g])]

            glyph_shape = freq + [1]

            mask_values, _ = masks.mask_values(mask, glyph_shape, atlas=atlas, inverse=mask_inverse,
                                               time=0 if mask_static else time, speed=speed)

            if shape[2] == 2:
                tensor = tf.stack([tensor[:, :, 0], tf.stack(mask_values)[:, :, 0]], 2)

            elif shape[2] == 4:
                tensor = tf.stack([tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2], tf.stack(mask_values)[:, :, 0]], 2)

            else:
                tensor *= mask_values

        if wavelet:
            tensor = effects.wavelet(tensor, initial_shape)

        tensor = effects.resample(tensor, shape, spline_order=spline_order)

        if (not corners and (freq[0] % 2) == 0) or (corners and (freq[0] % 2) == 1):
            tensor = effects.offset(tensor, shape, x=int((shape[1] / freq[1]) * .5), y=int((shape[0] / freq[0]) * .5))

    return tensor


def basic(freq, shape, ridges=False, sin=0.0, wavelet=False, spline_order=3,
          distrib=ValueDistribution.normal, corners=False,mask=None, mask_inverse=False, mask_static=False,
          lattice_drift=0.0, rgb=False, hue_range=.125, hue_rotation=None, saturation=1.0,
          hue_distrib=None, brightness_distrib=None, brightness_freq=None, saturation_distrib=None,
          speed=1.0, time=0.0, **post_process_args):
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
    :param bool mask_static: If True, don't animate the mask
    :param float lattice_drift: Push away from underlying lattice
    :param bool rgb: Disable HSV
    :param float hue_range: HSV hue range
    :param float|None hue_rotation: HSV hue bias
    :param float saturation: HSV saturation
    :param None|int|str|ValueDistribution hue_distrib: Override ValueDistribution for hue
    :param None|int|str|ValueDistribution saturation_distrib: Override ValueDistribution for saturation
    :param None|int|str|ValueDistribution brightness_distrib: Override ValueDistribution for brightness
    :param None|int|list[int] brightness_freq: Override frequency for brightness
    :param float speed: Displacement range for Z/W axis (simplex only)
    :param float time: Time argument for Z/W axis (simplex only)
    :return: Tensor

    Additional keyword args will be sent to :py:func:`noisemaker.effects.post_process`
    """

    if isinstance(freq, int):
        freq = effects.freq_for_shape(freq, shape)

    tensor = values(freq, shape, distrib=distrib, corners=corners,
                    mask=mask, mask_inverse=mask_inverse, mask_static=mask_static,
                    spline_order=spline_order, wavelet=wavelet, speed=speed, time=time)

    if lattice_drift:
        displacement = lattice_drift / min(freq[0], freq[1])

        tensor = effects.refract(tensor, shape, time=time, speed=speed,
                                 displacement=displacement, warp_freq=freq, spline_order=spline_order,
                                 signed_range=False)

    tensor = effects.post_process(tensor, shape, freq, time=time, speed=speed,
                                  spline_order=spline_order, rgb=rgb, **post_process_args)

    if shape[-1] >= 3 and not rgb:
        if hue_distrib:
            h = tf.squeeze(values(freq, [shape[0], shape[1], 1], distrib=hue_distrib, corners=corners,
                                  mask=mask, mask_inverse=mask_inverse, mask_static=mask_static,
                                  spline_order=spline_order, wavelet=wavelet, time=time, speed=speed))

        else:
            if hue_rotation is None:
                hue_rotation = tf.random.normal([])

            h = (tensor[:, :, 0] * hue_range + hue_rotation) % 1.0

        if saturation_distrib:
            s = tf.squeeze(values(freq, [shape[0], shape[1], 1], distrib=saturation_distrib, corners=corners,
                                  mask=mask, mask_inverse=mask_inverse, mask_static=mask_static,
                                  spline_order=spline_order, wavelet=wavelet, time=time, speed=speed))

        else:
            s = tensor[:, :, 1]

        s *= saturation

        if brightness_distrib or brightness_freq:
            if isinstance(brightness_freq, int):
                brightness_freq = effects.freq_for_shape(brightness_freq, shape)

            v = tf.squeeze(values(brightness_freq or freq, [shape[0], shape[1], 1],
                                  distrib=brightness_distrib or ValueDistribution.normal,
                                  corners=corners, mask=mask, mask_inverse=mask_inverse, mask_static=mask_static,
                                  spline_order=spline_order, wavelet=wavelet, time=time,
                                  speed=speed))

        else:
            v = tensor[:, :, 2]

        if ridges and spline_order:  # ridges don't work well when not interpolating values
            v = effects.crease(v)

        if sin:
            v = effects.normalize(tf.sin(sin * v))

        # Preserve the alpha channel before conversion to RGB
        if shape[2] == 4:
            a = tensor[:, :, 3]

        tensor = tf.image.hsv_to_rgb([tf.stack([h, s, v], 2)])[0]

        if shape[2] == 4:
            tensor = tf.stack([tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2], a], 2)

    elif ridges and spline_order:
        tensor = effects.crease(tensor)

    if sin and rgb:
        tensor = tf.sin(sin * tensor)

    return tensor


def multires(freq=3, shape=None, octaves=4, ridges=False, post_ridges=False, sin=0.0, wavelet=False, spline_order=3,
             reflect_range=0.0, refract_range=0.0, reindex_range=0.0, distrib=ValueDistribution.normal, corners=False,
             mask=None, mask_inverse=False, mask_static=False,
             deriv=False, deriv_metric=0, deriv_alpha=1.0, lattice_drift=0.0,
             post_reindex_range=0.0, post_reflect_range=0.0, post_refract_range=0.0, post_refract_y_from_offset=True,
             post_deriv=False, with_reverb=None, reverb_iterations=1,
             rgb=False, hue_range=.125, hue_rotation=None, saturation=1.0,
             hue_distrib=None, saturation_distrib=None, brightness_distrib=None, brightness_freq=None,
             octave_blending=OctaveBlending.falloff, time=0.0, speed=1.0, **post_process_args):
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
    :param float reflect_range: Per-octave derivative-based distort range (0..1+)
    :param float refract_range: Per-octave self-distort range (0..1+)
    :param float reindex_range: Per-octave self-reindexing range (0..1+)
    :param None|int with_reverb: Post-reduce tiled octave count
    :param int reverb_iterations: Re-reverberate N times
    :param int|ValueDistribution distrib: Type of noise distribution. See :class:`ValueDistribution` enum
    :param bool corners: If True, pin values to corners instead of image center
    :param None|ValueMask mask:
    :param bool mask_inverse:
    :param bool mask_static: If True, don't animate the mask
    :param bool deriv: Extract derivatives from noise
    :param DistanceFunction|int deriv_metric: Derivative distance metric
    :param float deriv_alpha: Derivative alpha blending amount
    :param float lattice_drift: Push away from underlying lattice
    :param float post_reindex_range: Reduced self-reindexing range (0..1+)
    :param float post_reflect_range: Reduced derivative-based distort range (0..1+)
    :param float post_refract_range: Reduced self-distort range (0..1+)
    :param float post_refract_y_from_offset: Derive Y offsets from offset image
    :param bool post_deriv: Reduced derivatives
    :param bool rgb: Disable HSV
    :param float hue_range: HSV hue range
    :param float|None hue_rotation: HSV hue bias
    :param float saturation: HSV saturation
    :param None|ValueDistribution hue_distrib: Override ValueDistribution for HSV hue
    :param None|ValueDistribution saturation_distrib: Override ValueDistribution for HSV saturation
    :param None|ValueDistribution brightness_distrib: Override ValueDistribution for HSV brightness
    :param None|int|list[int] brightness_freq: Override frequency for HSV brightness
    :param OctaveBlendingMethod|int octave_blending: Method for flattening octave values
    :param float speed: Displacement range for Z/W axis (simplex only)
    :param float time: Time argument for Z/W axis (simplex only)
    :return: Tensor

    Additional keyword args will be sent to :py:func:`noisemaker.effects.post_process`
    """

    # Normalize input

    if isinstance(freq, int):
        freq = effects.freq_for_shape(freq, shape)

    if isinstance(octave_blending, int):
        octave_blending = OctaveBlending(octave_blending)

    elif isinstance(octave_blending, str):
        octave_blending = OctaveBlending[octave_blending]

    original_shape = shape.copy()

    if octave_blending == OctaveBlending.alpha and shape[2] in (1, 3):  # Make sure there's an alpha channel
        shape[2] += 1

    # Make some noise

    tensor = tf.zeros(shape)

    for octave in range(1, octaves + 1):
        multiplier = 2 ** octave

        base_freq = [int(f * .5 * multiplier) for f in freq]

        if all(base_freq[i] > shape[i] for i in range(len(base_freq))):
            break

        layer = basic(base_freq, shape, ridges=ridges, sin=sin, wavelet=wavelet, spline_order=spline_order,
                      reflect_range=reflect_range / multiplier, refract_range=refract_range / multiplier, reindex_range=reindex_range / multiplier,
                      refract_y_from_offset=post_process_args.get("refract_y_from_offset", False),
                      distrib=distrib, corners=corners, mask=mask, mask_inverse=mask_inverse, mask_static=mask_static,
                      deriv=deriv, deriv_metric=deriv_metric, deriv_alpha=deriv_alpha,
                      lattice_drift=lattice_drift, rgb=rgb, hue_range=hue_range, hue_rotation=hue_rotation, saturation=saturation,
                      hue_distrib=hue_distrib, brightness_distrib=brightness_distrib, brightness_freq=brightness_freq,
                      saturation_distrib=saturation_distrib, time=time, speed=speed,
                      )

        if octave_blending == OctaveBlending.reduce_max:
            tensor = tf.maximum(tensor, layer)

        elif octave_blending == OctaveBlending.alpha:
            a = tf.expand_dims(layer[:, :, -1], -1)

            tensor = (tensor * (1.0 - a)) + layer * a

        else:  # falloff
            tensor += layer / multiplier

    # If the original shape did not include an alpha channel, reduce masked values to 0 (black)
    if octave_blending == OctaveBlending.alpha and original_shape[2] in (1, 3):
        a = tensor[:, :, -1]

        if original_shape[2] == 1:
            tensor = tf.expand_dims(tensor[:, :, 0] * a, -1)

        elif original_shape[2] == 3:
            tensor = tf.stack([tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]], 2) * tf.expand_dims(a, -1)

        shape = original_shape

    post_process_args['refract_signed_range'] = False
    post_process_args.pop("refract_y_from_offset", None)

    tensor = effects.post_process(tensor, shape, freq, time=time, speed=speed,
                                  ridges_hint=ridges and rgb, spline_order=spline_order,
                                  reindex_range=post_reindex_range, reflect_range=post_reflect_range,
                                  refract_range=post_refract_range, refract_y_from_offset=post_refract_y_from_offset,
                                  with_reverb=with_reverb, reverb_iterations=reverb_iterations,
                                  deriv=post_deriv, deriv_metric=deriv_metric, with_crease=post_ridges, rgb=rgb,
                                  **post_process_args)

    return tensor
