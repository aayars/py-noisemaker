"""Extremely high-level interface for composable noise presets. See `detailed docs <composer.html>`_."""

from collections import UserDict
from enum import Enum, EnumMeta
from functools import partial

import random

import tensorflow as tf

from noisemaker.effects_registry import EFFECTS
from noisemaker.generators import multires
from noisemaker.util import logger, save

DEFAULT_SHAPE = [1024, 1024, 3]

SETTINGS_KEY = "settings"

ALLOWED_KEYS = ["layers", SETTINGS_KEY, "generator", "octaves", "post", "final", "ai", "unique"]

# These correspond to https://platform.stability.ai/docs/api-reference#tag/v1generation/operation/textToImage
ALLOWED_AI_KEYS = ["prompt", "image_strength", "cfg_scale", "style_preset", "model"]

# v1 models: stable-diffusion-v1-6, stable-diffusion-xl-1024-v1-0
# v2 models: sd3, core, ultra
AI_MODEL = "core"

# Don't raise an exception if the following keys are unused in settings
UNUSED_OKAY = ["ai", "angle", "palette_alpha", "palette_name", "speed"]

# Populated by reload_presets() after setting random seed
GENERATOR_PRESETS = {}
EFFECT_PRESETS = {}


_STASH = {}


class Preset:
    def __init__(self, preset_name, presets, settings=None):
        """
        """

        self.layers = presets[preset_name].get("layers", [])
        self.name = preset_name

        prototype = presets.get(preset_name)

        if prototype is None:
            raise ValueError(f"Preset \"{preset_name}\" was not found among the available presets.")

        if not isinstance(prototype, dict):
            raise ValueError(f"Preset \"{preset_name}\" should be a dict, not \"{type(prototype)}\"")

        # To avoid mistakes in presets, unknown top-level keys are disallowed.
        for key in prototype:
            if key not in ALLOWED_KEYS:
                raise ValueError(f"Preset \"{preset_name}\": Key \"{key}\" is not permitted. " \
                                 f"Allowed keys are: {ALLOWED_KEYS}")

        # Build a flat list of parent preset names, in topological order.
        self.flattened_layers = []
        _flatten_ancestors(self.name, presets, {}, self.flattened_layers)

        # self.settings provides overridable args which can be consumed by generator, octaves, post, ai, and final.
        # SettingsDict is a custom dict class that enforces no unused extra keys, to minimize human error.
        self.settings = SettingsDict(_flatten_ancestor_metadata(self, None, SETTINGS_KEY, {}, presets))

        if settings:  # Inline overrides from caller (such as from CLI)
            self.settings.update(settings)

        # These args will be sent to generators.multires() to create the noise basis
        self.generator_kwargs = _flatten_ancestor_metadata(self, self.settings, "generator", {}, presets)

        # A list of callable effects functions, to be applied per-octave, in order
        self.octave_effects = _flatten_ancestor_metadata(self, self.settings, "octaves", [], presets)

        # A list of callable effects functions, to be applied post-reduce, in order
        self.post_effects = _flatten_ancestor_metadata(self, self.settings, "post", [], presets)

        # A list of callable effects functions, to be applied in order after everything else
        self.final_effects = _flatten_ancestor_metadata(self, self.settings, "final", [], presets)

        try:
            # To avoid mistakes in presets, unused keys are disallowed.
            self.settings.raise_if_unaccessed(unused_okay=UNUSED_OKAY)
        except Exception as e:
            raise ValueError(f"Preset \"{preset_name}\": {e}")

        # AI post-processing settings are not currently inherited, but are specified on a per-preset basis.
        _ai_settings = prototype.get("ai", {})
        for k in _ai_settings:
            if k not in ALLOWED_AI_KEYS:
                raise ValueError(f"Preset \"{preset_name}\": Disallowed key in \"ai\" section: " \
                                 f"\"{k}\"")

        self.ai_settings = {
            "prompt": _ai_settings.get("prompt", self.name.replace('-', ' ') + ", abstract art"),
            "image_strength": _ai_settings.get("image_strength", 0.5),
            "cfg_scale": _ai_settings.get("cfg_scale", 15),
            "style_preset": _ai_settings.get("style_preset", "digital-art"),
            "model": _ai_settings.get("model", AI_MODEL)
        }

        self.ai_settings.update(prototype.get("ai_settings", {}))

        # This will be set to True if the call to Stable Diffusion succeeds
        self.ai_success = False

    def __str__(self):
        return f"<Preset \"{self.name}\">"

    def is_generator(self):
        return self.generator_kwargs

    def is_effect(self):
        return self.post_effects or self.final_effects

    def render(self, seed, tensor=None, shape=DEFAULT_SHAPE, time=0.0, speed=1.0, filename="art.png",
               with_alpha=False, with_supersample=False, with_fxaa=False, with_ai=False, with_upscale=False,
               stability_model=None):
        """Render the preset to an image file."""

        try:
            tensor = multires(self, seed, tensor=tensor, shape=shape, with_supersample=with_supersample,
                              octave_effects=self.octave_effects, post_effects=self.post_effects,
                              with_fxaa=with_fxaa, with_ai=with_ai, final_effects=self.final_effects, with_alpha=with_alpha,
                              with_upscale=with_upscale, stability_model=stability_model, time=time, speed=speed,
                              **self.generator_kwargs)

            save(tensor, filename)

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


def _flatten_ancestors(preset_name, presets, unique, ancestors):
    for ancestor_name in presets[preset_name].get("layers", []):
        if ancestor_name not in presets:
            raise ValueError(f"\"{ancestor_name}\" was not found among the available presets.")

        # "unique" layers may only be inherited once
        if ancestor_name in unique:
            continue

        if presets[ancestor_name].get("unique"):
            unique[ancestor_name] = True

        _flatten_ancestors(ancestor_name, presets, unique, ancestors)

    ancestors.append(preset_name)


def _flatten_ancestor_metadata(preset, settings, key, default, presets):
    """Flatten ancestor preset metadata"""

    if isinstance(default, dict):
        flattened_metadata = {}
    else:
        flattened_metadata = []

    for ancestor_name in preset.flattened_layers:
        if key == SETTINGS_KEY:
            ancestor = presets[ancestor_name].get(key, lambda: default)
        else:
            ancestor = presets[ancestor_name].get(key, lambda _: default)

        if callable(ancestor):
            try:
                if key == SETTINGS_KEY:
                    ancestor = ancestor()
                else:
                    ancestor = ancestor(settings)

            except Exception as e:
                if preset.name == ancestor_name:
                    raise
                else:
                    raise ValueError(f"In ancestor \"{ancestor_name}\": {e}")

        else:
            raise ValueError(f"{ancestor_name}: Key \"{key}\" wasn't wrapped in a lambda. " +
                              "This can cause unexpected results for the given seed.")

        if not isinstance(ancestor, type(default)):
            raise ValueError(f"{ancestor_name}: Key \"{key}\" should be {type(default)}, not {type(data)}.")

        if isinstance(ancestor, dict):
            flattened_metadata.update(ancestor)
        else:
            flattened_metadata += ancestor

    return flattened_metadata


def random_member(*collections):
    """Return a random member from a collection, enum list, or enum. Ensures deterministic ordering."""

    collection = []

    for c in collections:
        if not hasattr(c, "__iter__"):
            raise ValueError(f"random_member(arg) should be iterable (collection, enum list, or enum)")

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


def reload_presets(presets):
    """Re-evaluate presets after changing the interpreter's random seed."""

    GENERATOR_PRESETS.clear()
    EFFECT_PRESETS.clear()

    presets = presets()

    for preset_name in presets:
        try:
            preset = Preset(preset_name, presets)

            if preset.is_generator():
                GENERATOR_PRESETS[preset_name] = preset

            if preset.is_effect():
                EFFECT_PRESETS[preset_name] = preset

        except Exception as e:
            if f"Preset \"{preset_name}\"" in str(e):
                raise
            else:
                raise ValueError(f"Preset \"{preset_name}\": {e}")


def stash(key, value=None):
    """Hold on to a variable for reference within the same lambda. Returns the stashed value."""

    global _STASH

    if value is not None:
        _STASH[key] = value

    return _STASH[key]


class UnusedKeys(Exception):
    """Exception raised when a preset has keys that aren't being used."""

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
                raise UnusedKeys(f"Settings key \"{keys[0]}\" (value: {self[keys[0]]}) is unused. This is usually human error.")
            else:
                raise UnusedKeys(f"Settings keys {keys} are unused. This is usually human error.")
