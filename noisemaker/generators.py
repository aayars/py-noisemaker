"""Noise generation interface for Noisemaker"""

import os
from functools import partial

import tempfile

from noisemaker.constants import (
    ColorSpace,
    InterpolationType,
    OctaveBlending,
    ValueDistribution
)

import noisemaker.ai as ai
import noisemaker.effects as effects
import noisemaker.oklab as oklab
import noisemaker.simplex as simplex
import noisemaker.util as util
import noisemaker.value as value
import tensorflow as tf

def basic(freq, shape, ridges=False, sin=0.0, spline_order=InterpolationType.bicubic,
          distrib=ValueDistribution.uniform, corners=False, mask=None, mask_inverse=False, mask_static=False,
          lattice_drift=0.0, color_space=ColorSpace.hsv, hue_range=.125, hue_rotation=None, saturation=1.0,
          hue_distrib=None, brightness_distrib=None, brightness_freq=None, saturation_distrib=None,
          speed=1.0, time=0.0, octave_effects=None, octave=1):
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
    :param ColorSpace color_space:
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
    """

    if isinstance(freq, int):
        freq = value.freq_for_shape(freq, shape)

    color_space = value.coerce_enum(color_space, ColorSpace)

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

    if lattice_drift:
        tensor = value.refract(tensor, shape, time=time, speed=speed,
                               displacement=lattice_drift / min(freq[0], freq[1]),
                               warp_freq=freq, spline_order=spline_order, signed_range=False)

    if octave_effects is not None:
        for effect_or_preset in octave_effects:
            tensor = _apply_octave_effect_or_preset(effect_or_preset, tensor, shape, time, speed, octave)

    # Preserve alpha channel for color space conversions
    alpha = None
    if shape[2] == 4:
        alpha = tensor[:, :, 3]
        tensor = tf.stack([tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]], 2)
    elif shape[2] == 2:
        alpha = tensor[:, :, 1]
        tensor = tf.stack([tensor[:, :, 0]], 2)

    original_color_space = color_space

    if color_space == ColorSpace.oklab:
        L = tensor[:, :, 0]
        a = tensor[:, :, 1] * -.509 + .276
        b = tensor[:, :, 2] * -.509 + .198

        tensor = value.clamp01(oklab.oklab_to_rgb(tf.stack([L, a, b], 2)))
        color_space = ColorSpace.rgb

    if color_space == ColorSpace.rgb:
        tensor = tf.image.rgb_to_hsv([tensor])[0]
        color_space = ColorSpace.hsv

    if color_space == ColorSpace.hsv:
        # Use 1 channel for per-channel noise generation, if any
        common_value_params["shape"] = [shape[0], shape[1], 1]

        # tweak hue
        if hue_distrib:
            h = tf.squeeze(value.values(freq=freq, distrib=hue_distrib, **common_value_params))
        else:
            if original_color_space == ColorSpace.hsv:
                if hue_rotation is None:
                    hue_rotation = simplex.random(time=time, speed=speed)
            else:  # Avoid hard edges on color models that don't wrap hue from 1 to 0 naturally
                hue_range = 1.0
                hue_rotation = 0.0

            h = (tensor[:, :, 0] * hue_range + hue_rotation) % 1.0

        # tweak saturation
        if saturation_distrib:
            s = tf.squeeze(value.values(freq=freq, distrib=saturation_distrib, **common_value_params))
        else:
            s = tensor[:, :, 1]

        s *= saturation

        # tweak brightness
        if brightness_distrib or brightness_freq:
            if isinstance(brightness_freq, int):
                brightness_freq = value.freq_for_shape(brightness_freq, shape)

            v = tf.squeeze(value.values(freq=brightness_freq or freq,
                                        distrib=brightness_distrib or ValueDistribution.uniform,
                                        **common_value_params))
        else:
            v = tensor[:, :, 2]

        if ridges and spline_order:  # ridges don't work with spline_order == 0
            v = value.ridge(v)

        if sin:
            v = value.normalize(tf.sin(sin * v))

        tensor = tf.image.hsv_to_rgb([tf.stack([h, s, v], 2)])[0]

    if color_space == ColorSpace.grayscale:
        if ridges and spline_order:  # ridges don't work with spline_order == 0
            tensor = value.ridge(tensor)

        if sin:
            tensor = tf.sin(sin * tensor)

    # re-insert the alpha channel
    if shape[2] == 4:
        tensor = tf.stack([tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2], alpha], 2)
    elif shape[2] == 2:
        tensor = tf.stack([tensor[:, :, 0], alpha], 2)

    return tensor


def multires(preset, seed, freq=3, shape=None, octaves=1, ridges=False, sin=0.0,
             spline_order=InterpolationType.bicubic, distrib=ValueDistribution.uniform, corners=False,
             mask=None, mask_inverse=False, mask_static=False, lattice_drift=0.0, with_supersample=False,
             color_space=ColorSpace.hsv, hue_range=.125, hue_rotation=None, saturation=1.0,
             hue_distrib=None, saturation_distrib=None, brightness_distrib=None, brightness_freq=None,
             octave_blending=OctaveBlending.falloff, octave_effects=None, post_effects=None,
             with_alpha=False, with_ai=False, final_effects=None, with_upscale=False, with_fxaa=False,
             stability_model=None, style_filename=None, time=0.0, speed=1.0, tensor=None):
    """
    Generate multi-resolution value noise. For each octave: freq increases, amplitude decreases.

    .. image:: images/multires.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Preset preset: The Preset object being rendered
    :param int seed: The generation seed to use
    :param int|list[int] freq: Bottom layer frequency. Int, or list of ints for each spatial dimension
    :param list[int]: Shape of noise. For 2D noise, this is [height, width, channels]
    :param int octaves: Octave count. Number of multi-res layers. Typically 1-8
    :param bool ridges: Per-octave "crease" at midpoint values: (1 - abs(n * 2 - 1))
    :param bool post_ridges: Post-reduce "crease" at midpoint values: (1 - abs(n * 2 - 1))
    :param float sin: Apply sin function to noise basis
    :param int spline_order: Spline point count. 0=Constant, 1=Linear, 2=Cosine, 3=Bicubic
    :param int|ValueDistribution distrib: Type of noise distribution. See :class:`ValueDistribution` enum
    :param bool corners: If True, pin values to corners instead of image center
    :param None|ValueMask mask:
    :param bool mask_inverse:
    :param bool mask_static: If True, don't animate the mask
    :param float lattice_drift: Push away from underlying lattice
    :param bool with_supersample: Use x2 supersampling
    :param ColorSpace color_space:
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
    :param bool with_alpha: Include alpha channel
    :param bool with_ai: AI: Apply image-to-image before the final effects pass
    :param list[callable] final_effects: A list of composer lambdas to invoke after everything else
    :param bool with_upscale: AI: x2 upscale final results
    :param bool with_fxaa: Apply FXAA to results
    :param str stability_model: AI: Override the default stability.ai model
    :param str|None style_filename: AI: Save the style reference, or load it if the file already exists.
    :param float speed: Displacement range for Z/W axis (simplex and periodic only)
    :param float time: Time argument for Z/W axis (simplex and periodic only)
    :return: Tensor
    """

    if seed:
        value.set_seed(seed)

    if with_ai and with_supersample:
        # Supersampling makes the input too large for current AI models
        raise Exception("--with-ai and --with-supersample may not be used together.")

    # Normalize input
    color_space = value.coerce_enum(color_space, ColorSpace)
    octave_blending = value.coerce_enum(octave_blending, OctaveBlending)

    original_shape = shape.copy()

    if shape[-1] is None:
        shape = util.shape_from_params(shape[1], shape[0], color_space, with_alpha)

    if isinstance(freq, int):
        freq = value.freq_for_shape(freq, shape)

    if with_supersample:
        shape[0] *= 2
        shape[1] *= 2

    if octave_blending == OctaveBlending.alpha and shape[2] in (1, 3):  # Make sure there's an alpha channel
        shape[2] += 1

    if tensor is None:
        tensor = tf.zeros(shape)

        for octave in range(1, octaves + 1):
            multiplier = 2 ** octave

            base_freq = [int(f * .5 * multiplier) for f in freq]

            if all(base_freq[i] > shape[i] for i in range(len(base_freq))):
                break

            layer = basic(base_freq, shape, ridges=ridges, sin=sin, spline_order=spline_order, corners=corners,
                          distrib=distrib, mask=mask, mask_inverse=mask_inverse, mask_static=mask_static,
                          lattice_drift=lattice_drift, color_space=color_space, hue_range=hue_range,
                          hue_rotation=hue_rotation, saturation=saturation, hue_distrib=hue_distrib,
                          brightness_distrib=brightness_distrib, brightness_freq=brightness_freq,
                          saturation_distrib=saturation_distrib, octave_effects=octave_effects, octave=octave,
                          time=time, speed=speed)

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

    else:
        for effect_or_preset in octave_effects:
            tensor = _apply_octave_effect_or_preset(effect_or_preset, tensor, shape, time, speed, 1)

    tensor = value.normalize(tensor)

    final = []

    if tensor.shape != shape:
        value.resample(tensor, shape)

    for effect_or_preset in post_effects:
        tensor, f = _apply_post_effect_or_preset(effect_or_preset, tensor, shape, time, speed)
        final += f

    if with_ai:
        tensor = value.normalize(tensor)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = f"{tmp}/temp.png"
            # tmp_path = "input.png"  # XXX

            util.save(tensor, tmp_path)

            try:
                # image-to-image as a style reference
                style_reference = None

                if style_filename:
                    if os.path.exists(style_filename):
                        style_reference = tf.image.convert_image_dtype(util.load(style_filename), tf.float32)

                    style_path = style_filename

                else:
                    style_path = f"{tmp}/temp-style.png"

                if style_reference is None:
                    style_reference = ai.apply(preset.ai_settings, seed, input_filename=tmp_path,
                                               stability_model=stability_model)

                style_reference = value.resample(style_reference, shape)

                util.save(style_reference, style_path)

                new_tensor = ai.apply_style(preset.ai_settings, seed, tmp_path, style_path)

                new_tensor = value.resample(new_tensor, shape)

                # tensor = new_tensor
                tensor = value.blend(tensor, new_tensor, 0.5)
                tensor = tf.image.adjust_contrast(tensor, 1.25)

                preset.ai_success = True

            except Exception as e:
                util.logger.error(f"ai.apply() failed: {e}\nSeed: {seed}")

    for effect_or_preset in final + final_effects:
        tensor = _apply_final_effect_or_preset(effect_or_preset, tensor, shape, time, speed)

    tensor = value.normalize(tensor)

    if with_fxaa:
        tensor = value.fxaa(tensor, shape)

    if with_supersample:
        tensor = value.proportional_downsample(tensor, shape, original_shape)

    if with_upscale:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = f"{tmp}/temp.png"

            util.save(tensor, tmp_path)

            try:
                tensor = ai.x4_upscale(tmp_path)

            except Exception as e:
                util.logger.error(f"preset.upscale() failed: {e}\nSeed: {seed}")

    return tensor


def _apply_octave_effect_or_preset(effect_or_preset, tensor, shape, time, speed, octave):
    """Helper function to either invoke a octave effect or unroll a preset."""
    if callable(effect_or_preset):
        if "displacement" in effect_or_preset.keywords:
            kwargs = dict(effect_or_preset.keywords)  # Be sure to copy, otherwise it modifies the original
            kwargs["displacement"] /= 2 ** octave
            effect_or_preset = partial(effect_or_preset.func, **kwargs)

        return effect_or_preset(tensor=tensor, shape=shape, time=time, speed=speed)

    else:  # Is a Preset. Unroll me.
        for e_or_p in effect_or_preset.octave_effects:
            tensor = _apply_octave_effect_or_preset(e_or_p, tensor, shape, time, speed, octave)

        return tensor

def _apply_post_effect_or_preset(effect_or_preset, tensor, shape, time, speed):
    """Helper function to either invoke a post effect or unroll a preset."""
    if callable(effect_or_preset):
        return effect_or_preset(tensor=tensor, shape=shape, time=time, speed=speed), []

    else:  # Is a Preset. Unroll me.
        final = []
        # Post effects may also define "final" effects. Collect them and return them so we
        # can tack them on at the end after everything is said and done
        final += effect_or_preset.final_effects

        for e_or_p in effect_or_preset.post_effects:
            tensor, f = _apply_post_effect_or_preset(e_or_p, tensor, shape, time, speed)
            final += f

        return tensor, final


def _apply_final_effect_or_preset(effect_or_preset, tensor, shape, time, speed):
    """Helper function to either invoke a final effect or unroll a preset."""
    if callable(effect_or_preset):
        return effect_or_preset(tensor=tensor, shape=shape, time=time, speed=speed)

    else:  # Is a Preset. Unroll me.
        for e_or_p in effect_or_preset.post_effects + effect_or_preset.final_effects:
            tensor = _apply_final_effect_or_preset(e_or_p, tensor, shape, time, speed)

        return tensor
