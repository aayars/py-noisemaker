"""Noise generation interface for Noisemaker"""

import tensorflow as tf

from noisemaker.constants import InterpolationType, OctaveBlending, ValueDistribution

import noisemaker.effects as effects
import noisemaker.value as value


def basic(freq, shape, ridges=False, sin=0.0, spline_order=InterpolationType.bicubic,
          distrib=ValueDistribution.normal, corners=False, mask=None, mask_inverse=False, mask_static=False,
          lattice_drift=0.0, rgb=False, hue_range=.125, hue_rotation=None, saturation=1.0,
          hue_distrib=None, brightness_distrib=None, brightness_freq=None, saturation_distrib=None,
          speed=1.0, time=0.0, octave_effects=None, **post_process_args):
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
    :param float speed: Displacement range for Z/W axis (simplex and periodic only)
    :param float time: Time argument for Z/W axis (simplex and periodic only)
    :return: Tensor

    Additional keyword args will be sent to :py:func:`noisemaker.effects.post_process`
    """

    if isinstance(freq, int):
        freq = value.freq_for_shape(freq, shape)

    common_value_params = {
        "corners": corners,
        "mask": mask,
        "mask_inverse": mask_inverse,
        "mask_static": mask_static,
        "speed": speed,
        "spline_order": spline_order,
        "time": time,
    }

    tensor = value.values(freq=freq, shape=shape, distrib=distrib, **common_value_params)

    # Use 1 channel for per-channel noise generation, if any
    common_value_params["shape"] = [shape[0], shape[1], 1]

    if lattice_drift:
        displacement = lattice_drift / min(freq[0], freq[1])

        tensor = effects.refract(tensor, shape, time=time, speed=speed,
                                 displacement=displacement, warp_freq=freq, spline_order=spline_order,
                                 signed_range=False)

    if octave_effects is not None:
        for effect in octave_effects:
            tensor = effect(tensor=tensor, shape=shape, time=time, speed=speed)

    else:
        tensor = effects.post_process(tensor, shape, freq, time=time, speed=speed, spline_order=spline_order, rgb=rgb, **post_process_args)

    if shape[-1] >= 3 and not rgb:
        if hue_distrib:
            h = tf.squeeze(value.values(freq=freq, distrib=hue_distrib, **common_value_params))

        else:
            if hue_rotation is None:
                hue_rotation = tf.random.normal([])

            h = (tensor[:, :, 0] * hue_range + hue_rotation) % 1.0

        if saturation_distrib:
            s = tf.squeeze(value.values(freq=freq, distrib=saturation_distrib, **common_value_params))

        else:
            s = tensor[:, :, 1]

        s *= saturation

        if brightness_distrib or brightness_freq:
            if isinstance(brightness_freq, int):
                brightness_freq = value.freq_for_shape(brightness_freq, shape)

            v = tf.squeeze(value.values(freq=brightness_freq or freq, distrib=brightness_distrib or ValueDistribution.normal,
                                        **common_value_params))

        else:
            v = tensor[:, :, 2]

        if ridges and spline_order:  # ridges don't work well when not interpolating values
            v = value.ridge(v)

        # Preserve the alpha channel before conversion to RGB
        if shape[2] == 4:
            a = tensor[:, :, 3]

        tensor = tf.image.hsv_to_rgb([tf.stack([h, s, v], 2)])[0]

        if shape[2] == 4:
            tensor = tf.stack([tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2], a], 2)

    elif ridges and spline_order:
        tensor = value.ridge(tensor)

    return tensor


def multires(freq=3, shape=None, octaves=4, ridges=False, spline_order=InterpolationType.bicubic,
             distrib=ValueDistribution.normal, corners=False,
             mask=None, mask_inverse=False, mask_static=False, lattice_drift=0.0,
             rgb=False, hue_range=.125, hue_rotation=None, saturation=1.0,
             hue_distrib=None, saturation_distrib=None, brightness_distrib=None, brightness_freq=None,
             octave_blending=OctaveBlending.falloff,
             octave_effects=None, post_effects=None, time=0.0, speed=1.0):
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
    :param int spline_order: Spline point count. 0=Constant, 1=Linear, 2=Cosine, 3=Bicubic
    :param int|ValueDistribution distrib: Type of noise distribution. See :class:`ValueDistribution` enum
    :param bool corners: If True, pin values to corners instead of image center
    :param None|ValueMask mask:
    :param bool mask_inverse:
    :param bool mask_static: If True, don't animate the mask
    :param float lattice_drift: Push away from underlying lattice
    :param bool rgb: Disable HSV
    :param float hue_range: HSV hue range
    :param float|None hue_rotation: HSV hue bias
    :param float saturation: HSV saturation
    :param None|ValueDistribution hue_distrib: Override ValueDistribution for HSV hue
    :param None|ValueDistribution saturation_distrib: Override ValueDistribution for HSV saturation
    :param None|ValueDistribution brightness_distrib: Override ValueDistribution for HSV brightness
    :param None|int|list[int] brightness_freq: Override frequency for HSV brightness
    :param OctaveBlendingMethod|int octave_blending: Method for flattening octave values
    :param list[callable] octave_effects: A list of composer lambdas to invoke per-octave
    :param list[callable] post_effects: A list of composer lambdas to invoke after flattening layers
    :param float speed: Displacement range for Z/W axis (simplex and periodic only)
    :param float time: Time argument for Z/W axis (simplex and periodic only)
    :return: Tensor

    Additional keyword args will be sent to :py:func:`noisemaker.effects.post_process`
    """

    # Normalize input

    if isinstance(freq, int):
        freq = value.freq_for_shape(freq, shape)

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

        layer = basic(base_freq, shape, ridges=ridges, spline_order=spline_order, corners=corners,
                      mask=mask, mask_inverse=mask_inverse, mask_static=mask_static, lattice_drift=lattice_drift, rgb=rgb,
                      hue_range=hue_range, hue_rotation=hue_rotation, saturation=saturation, hue_distrib=hue_distrib,
                      brightness_distrib=brightness_distrib, brightness_freq=brightness_freq,
                      saturation_distrib=saturation_distrib, octave_effects=octave_effects, time=time, speed=speed)

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

    for effect_or_preset in post_effects:
        tensor = _apply_post_effect_or_preset(effect_or_preset, tensor, shape, time, speed)

    return tensor


def multires_old(freq=3, shape=None, octaves=4, ridges=False, sin=0.0, spline_order=InterpolationType.bicubic,
                 distrib=ValueDistribution.normal, corners=False,
                 mask=None, mask_inverse=False, mask_static=False, octave_effects=None, post_effects=None, time=0.0, speed=1.0,
                 rgb=False, hue_range=.125, hue_rotation=None, saturation=1.0,
                 hue_distrib=None, saturation_distrib=None, brightness_distrib=None, brightness_freq=None,
                 octave_blending=OctaveBlending.falloff,
                 post_ridges=False, reflect_range=0.0, refract_range=0.0, reindex_range=0.0,
                 deriv=False, deriv_metric=0, deriv_alpha=1.0, lattice_drift=0.0,
                 post_reindex_range=0.0, post_reflect_range=0.0, post_refract_range=0.0, post_refract_y_from_offset=True,
                 post_deriv=False, with_reverb=None, reverb_iterations=1, **post_process_args):
    """
    This method is deprecated. Please use multires() instead.

    :param int|list[int] freq: Bottom layer frequency. Int, or list of ints for each spatial dimension
    :param list[int]: Shape of noise. For 2D noise, this is [height, width, channels]
    :param int octaves: Octave count. Number of multi-res layers. Typically 1-8
    :param bool ridges: Per-octave "crease" at midpoint values: (1 - abs(n * 2 - 1))
    :param bool post_ridges: Post-reduce "crease" at midpoint values: (1 - abs(n * 2 - 1))
    :param float sin: Apply sin function to noise basis
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
    :param float speed: Displacement range for Z/W axis (simplex and periodic only)
    :param float time: Time argument for Z/W axis (simplex and periodic only)
    :param list[callable] octave_effects: A list of composer lambdas to invoke per-octave, rather than calling post_process.
    :return: Tensor

    Additional keyword args will be sent to :py:func:`noisemaker.effects.post_process`
    """

    # Normalize input

    if isinstance(freq, int):
        freq = value.freq_for_shape(freq, shape)

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

        layer = basic(base_freq, shape, ridges=ridges, sin=sin, spline_order=spline_order,
                      reflect_range=reflect_range / multiplier, refract_range=refract_range / multiplier, reindex_range=reindex_range / multiplier,
                      refract_y_from_offset=post_process_args.get("refract_y_from_offset", False),
                      distrib=distrib, corners=corners, mask=mask, mask_inverse=mask_inverse, mask_static=mask_static,
                      deriv=deriv, deriv_metric=deriv_metric, deriv_alpha=deriv_alpha,
                      lattice_drift=lattice_drift, rgb=rgb, hue_range=hue_range, hue_rotation=hue_rotation, saturation=saturation,
                      hue_distrib=hue_distrib, brightness_distrib=brightness_distrib, brightness_freq=brightness_freq,
                      saturation_distrib=saturation_distrib, time=time, speed=speed, octave_effects=octave_effects,
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

    if post_effects is not None:
        for effect_or_preset in post_effects:
            tensor = _apply_post_effect_or_preset(effect_or_preset, tensor, shape, time, speed)

    else:
        tensor = effects.post_process(tensor, shape, freq, time=time, speed=speed,
                                      ridges_hint=ridges and rgb, spline_order=spline_order,
                                      reindex_range=post_reindex_range, reflect_range=post_reflect_range,
                                      refract_range=post_refract_range, refract_y_from_offset=post_refract_y_from_offset,
                                      with_reverb=with_reverb, reverb_iterations=reverb_iterations,
                                      deriv=post_deriv, deriv_metric=deriv_metric, with_ridge=post_ridges, rgb=rgb,
                                      **post_process_args)

    return tensor


def _apply_post_effect_or_preset(effect_or_preset, tensor, shape, time, speed):
    """Helper function to either invoke a post effect or unroll a preset."""

    if callable(effect_or_preset):
        return effect_or_preset(tensor=tensor, shape=shape, time=time, speed=speed)

    else:  # Is a Preset. Unroll me.
        for e_or_p in effect_or_preset.post_effects:
            tensor = _apply_post_effect_or_preset(e_or_p, tensor, shape, time, speed)

        return tensor
