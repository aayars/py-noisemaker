"""Extremely high-level interface for composable noise presets. See `detailed docs <composer.html>`_."""

from functools import partial

import tensorflow as tf

from noisemaker.effects import EFFECTS
from noisemaker.generators import multires
from noisemaker.util import logger, save

DEFAULT_SHAPE = [1024, 1024, 3]

SETTINGS_KEY = 'settings'

ALLOWED_KEYS = ['extends', SETTINGS_KEY, 'generator', 'octaves', 'post']


class Preset:
    def __init__(self, preset_name, presets, settings=None):
        """
        """

        self.name = preset_name

        prototype = presets.get(preset_name)

        if prototype is None:
            raise ValueError(f"Preset named \"{preset_name}\" was not found among the available presets.")

        if not isinstance(prototype, dict):
            raise ValueError(f"Preset \"{preset_name}\" should be a dict, not \"{type(prototype)}\"")

        for key in prototype:
            if key not in ALLOWED_KEYS:
                raise ValueError(f"Not sure what to do with key \"{key}\" in preset \"{preset_name}\". Typo?")

        # The "settings" dict provides overridable args to generator, octaves, and post
        self.settings = _rollup(preset_name, SETTINGS_KEY, {}, presets, None)

        if settings:  # Inline overrides from caller
            self.settings.update(settings)

        # These args will be sent to generators.multires() to create the noise basis
        self.generator_kwargs = _rollup(preset_name, 'generator', {}, presets, self.settings)

        # A list of callable effects functions, to be applied per-octave, in order
        self.octave_effects = _rollup(preset_name, 'octaves', [], presets, self.settings)

        # A list of callable effects functions, to be applied post-reduce, in order
        self.post_effects = _rollup(preset_name, 'post', [], presets, self.settings)

    def __str__(self):
        return f"<Preset \"{self.name}\">"

    def render(self, shape=DEFAULT_SHAPE, name="art.png"):
        """Render the preset to an image file."""

        try:
            tensor = multires(shape=shape, octave_effects=self.octave_effects, post_effects=self.post_effects, **self.generator_kwargs)

            with tf.compat.v1.Session().as_default():
                save(tensor, name)

        except Exception as e:
            logger.error(f"Error rendering preset named {self.name}: {e}")

            raise


def Effect(effect_name, **kwargs):
    """Return a partial effects function. Invoke the wrapped function with params "tensor", "shape", "time", and "speed." """

    if effect_name not in EFFECTS:
        raise ValueError(f'"{effect_name}" is not a registered effect name.')

    for k in kwargs:
        if k not in EFFECTS[effect_name]:
            raise ValueError(f'Effect "{effect_name}" does not accept a parameter named "{k}"')

    return partial(EFFECTS[effect_name]["func"], **kwargs)


def _rollup(preset_name, key, default, presets, settings):
    """Recursively merge parent preset metadata into the named child."""

    evaluated_kwargs = presets[preset_name]

    # child_data represents the current preset's *evaluated* kwargs. The lambdas have been evaluated as per whatever the
    # current seed and random generator state is. Ancestor preset kwargs will get evaluated and merged into this.
    if key == SETTINGS_KEY:
        child_data = evaluated_kwargs.get(key, lambda: default)
    else:
        child_data = evaluated_kwargs.get(key, lambda _: default)

    if not callable(child_data):
        raise ValueError(f"Preset \"{preset_name}\" key \"{key}\" wasn't wrapped in a lambda. This can cause unexpected results for the given seed.")
    elif key == SETTINGS_KEY:
        child_data = child_data()
    else:
        child_data = child_data(settings)

    if not isinstance(child_data, type(default)):
        raise ValueError(f"Preset \"{preset_name}\" key \"{key}\" is a {type(child_data)}, but we were expecting a {type(default)}.")

    for base_preset_name in reversed(evaluated_kwargs.get('extends', [])):
        if base_preset_name not in presets:
            raise ValueError(f"Preset \"{preset_name}\"'s parent named \"{base_preset_name}\" was not found among the available presets.")

        # Data to be merged; just need to know how to merge it, based on type.
        parent_data = _rollup(base_preset_name, key, default, presets, settings)

        if callable(parent_data):
            if key == SETTINGS_KEY:
                parent_data = parent_data()
            else:
                parent_data = parent_data(settings)
        else:
            parent_data = parent_data.copy()

        if isinstance(parent_data, dict):
            child_data = dict(parent_data, **child_data)  # merge keys, overriding parent with child
        elif isinstance(parent_data, list):
            child_data = parent_data + child_data  # append (don't prepend)
        else:
            raise ValueError(f"Not sure how to roll up data of type {type(parent_data)} (key: \"{key}\")")

    return child_data
