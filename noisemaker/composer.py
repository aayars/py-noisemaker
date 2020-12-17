"""Extremely high-level interface for composable noise presets. See `detailed docs <composer.html>`_."""

from collections import UserDict
from enum import Enum, EnumMeta
from functools import partial

import json
import random

import tensorflow as tf

from noisemaker.effects import EFFECTS
from noisemaker.generators import multires
from noisemaker.util import logger, save

DEFAULT_SHAPE = [1024, 2048, 3]

SETTINGS_KEY = "settings"

ALLOWED_KEYS = ["layers", SETTINGS_KEY, "generator", "octaves", "post"]

# Don't raise an exception if the following keys are unused in settings
UNUSED_OKAY = ["speed", "palette_name"]

_STASH = {}


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
        self.settings = SettingsDict(_rollup(preset_name, SETTINGS_KEY, {}, presets, None))

        if settings:  # Inline overrides from caller
            self.settings.update(settings)

        # These args will be sent to generators.multires() to create the noise basis
        self.generator_kwargs = _rollup(preset_name, "generator", {}, presets, self.settings)

        # A list of callable effects functions, to be applied per-octave, in order
        self.octave_effects = _rollup(preset_name, "octaves", [], presets, self.settings)

        # A list of callable effects functions, to be applied post-reduce, in order
        self.post_effects = _rollup(preset_name, "post", [], presets, self.settings)

        # Make sure there's no dangling settings keys
        try:
            self.settings.raise_if_unaccessed(unused_okay=UNUSED_OKAY)

        except UnusedKeys as e:
            raise UnusedKeys(f"Preset \"{preset_name}\": {e}")

    def __str__(self):
        return f"<Preset \"{self.name}\">"

    def is_generator(self):
        return self.generator_kwargs or "voronoi" in self.settings

    def is_effect(self):
        return not self.is_generator() or self.settings.get("voronoi_refract")

    def render(self, tensor=None, shape=DEFAULT_SHAPE, name="art.png"):
        """Render the preset to an image file."""

        logger.debug("Rendering noise: "
                     + json.dumps(self.__dict__,
                                  default=lambda v: dict(v) if isinstance(v, SettingsDict) else str(v),
                                  indent=4))

        try:
            tensor = multires(tensor=tensor, shape=shape, octave_effects=self.octave_effects, post_effects=self.post_effects,
                              **self.generator_kwargs)

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

    for base_preset_name in reversed(evaluated_kwargs.get("layers", [])):
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


def random_member(*collections):
    """Return a random member from a collection, enum list, or enum. Ensures deterministic ordering."""

    collection = []

    for c in collections:
        if isinstance(collection, EnumMeta):
            collection += list(c)

        # maybe it's a list of enum members
        elif isinstance(next(iter(c), None), Enum):
            collection += [s[1] for s in sorted([(m.name if m is not None else "", m) for m in c])]

        else:
            # make sure order is deterministic
            collection += sorted(c)

    return collection[random.randint(0, len(collection) - 1)]


def coin_flip():
    return bool(random.randint(0, 1))


def enum_range(a, b):
    """Return a list of enum members within the specified inclusive numeric value range."""

    enum_class = type(a)

    members = []

    for i in range(a.value, b.value + 1):
        members.append(enum_class(i))

    return members


def stash(key, value=None):
    """Hold on to a variable for reference within the same lambda. Returns the stashed value if not given as arg."""

    global _STASH
    if value is not None:
        _STASH[key] = value
    return _STASH[key]


class UnusedKeys(Exception):
    """Exception raised when a preset has keys that aren't being used"""

    pass


class SettingsDict(UserDict):
    """dict, but it makes sure the caller eats everything on their plate."""

    def __init__(self, *args, **kwargs):
        self.__accessed__ = {}

        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        self.__accessed__[key] = True

        return super().__getitem__(key)

    def was_accessed(self, key):
        return key in self.__accessed__

    def raise_if_unaccessed(self, unused_okay=None):
        keys = []

        for key in self:
            if not self.was_accessed(key) and (unused_okay is None or key not in unused_okay):
                keys.append(key)

        if keys:
            if len(keys) == 1:
                raise UnusedKeys(f"Settings key \"{keys[0]}\" is unused. This is usually human error.")
            else:
                raise UnusedKeys(f"Settings keys {keys} are unused. This is usually human error.")
