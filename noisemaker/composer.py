"""Extremely high-level interface for composable noise presets."""

from functools import partial

import tensorflow as tf

from noisemaker.effects import EFFECTS
from noisemaker.generators import multires
from noisemaker.util import logger, save

DEFAULT_SHAPE = [1024, 1024, 3]

# PRESETS = {
#     # "settings", "generator", "octaves", and "post" are wrapped in lambda to enable re-evaluation if/when the seed value changes.
#     # For additional examples, see test/test_composer.py
#     "just-an-example": {
#         # A list of parent preset names, if any:
#        "extends": [],
#
#         # A dictionary of global args which may be re-used within the preset:
#         "settings": lambda: {},
#
#         # A dictionary of args to feed to the noise basis generator function:
#         "generator": lambda settings: {},
#
#         # A list of per-octave effects, to apply in order:
#         "octaves": lambda settings: [
#             # Effect(effect_name, args),  # Effect() returns a callable effect function
#             # ...
#         ],
#
#         # A list of post-reduce effects, to apply in order:
#         "post": lambda settings: [
#             # Effect(effect_name, args),
#             # ...
#         ],
#     },
#
#     # ...
# }

class Preset:
    def __init__(self, preset_name, presets, settings=None):
        """
        """

        self.name = preset_name

        # The "settings" dict provides overridable args to generator, octaves, and post
        self.settings = settings or _rollup(preset_name, 'settings', {}, presets, None)

        # These args will be sent to generators.multires() to create the noise basis
        self.generator_kwargs = _rollup(preset_name, 'generator', {}, presets, self.settings)

        # A list of callable effects functions, to be applied per-octave, in order
        self.octave_effects = _rollup(preset_name, 'octaves', [], presets, self.settings)

        # A list of callable effects functions, to be applied post-reduce, in order
        self.post_effects = _rollup(preset_name, 'post', [], presets, self.settings)

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
    child_data = presets[preset_name].get(key, default)

    if key == 'settings':
        child_data = child_data()
    elif callable(child_data):
        child_data = child_data(settings)

    if not isinstance(child_data, type(default)):
        raise ValueError(f"Preset {preset_name} key \"{key}\" is a {type(child_data)}, but we were expecting a {type(default)}.")

    for base_preset_name in evaluated_kwargs.get('extends', []):
        # Evaluate this particular key using the resolved "settings" dictionary
        print(f"Evaluating {base_preset_name} {key}")

        # Data to be merged; just need to know how to merge it, based on type.
        # parent_data = type(self)(base_preset_name, presets, self.settings)._rollup(key, default, presets)
        parent_data = _rollup(base_preset_name, key, default, presets, settings)

        # Evaluate this particular key using the resolved "settings" dictionary
        if callable(parent_data):
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
