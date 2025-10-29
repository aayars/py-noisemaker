"""Extremely high-level interface for composable noise presets. See `detailed docs <composer.html>`_."""

from __future__ import annotations

from collections import UserDict, defaultdict
from enum import Enum, EnumMeta
from functools import partial, lru_cache
from typing import Any, Callable
import inspect
import re

import noisemaker.rng as rng

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
UNUSED_OKAY = ["ai", "angle", "palette_alpha", "palette_name", "palette_on", "speed"]

# Populated by reload_presets() after setting random seed
GENERATOR_PRESETS: dict[str, Any] = {}
EFFECT_PRESETS: dict[str, Any] = {}


_STASH: dict[str, Any] = {}


class Preset:
    """
    A composable noise preset with layered effects and settings.

    This class represents a preset configuration that can generate procedural noise
    with various effects applied in stages (per-octave, post-reduce, and final).
    """

    def __init__(self, preset_name: str, presets: dict[str, Any], settings: dict[str, Any] | None = None):
        """
        Initialize a Preset from the presets dictionary.

        Args:
            preset_name: Name of the preset to load
            presets: Dictionary of all available presets
            settings: Optional settings overrides
        """

        self.name = preset_name

        layer_cache: dict[str, Any] = {}
        self.layers = list(_resolve_preset_layers(self.name, presets, layer_cache))

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
        self.flattened_layers: list[str] = []
        _flatten_ancestors(self.name, presets, {}, self.flattened_layers, layer_cache)

        # self.settings provides overridable args which can be consumed by generator, octaves, post, ai, and final.
        # SettingsDict is a custom dict class that enforces no unused extra keys, to minimize human error.
        self.settings = SettingsDict(
            _flatten_ancestor_metadata(self, None, SETTINGS_KEY, {}, presets)
        )

        if settings:  # Inline overrides from caller (such as from CLI)
            self.settings.update(settings)

        # These args will be sent to generators.multires() to create the noise basis
        self.generator_kwargs: dict[str, Any] = _flatten_ancestor_metadata(
            self, self.settings, "generator", {}, presets
        )

        # A list of callable effects functions, to be applied per-octave, in order
        self.octave_effects: list[Callable] = _flatten_ancestor_metadata(
            self, self.settings, "octaves", [], presets
        )

        # A list of callable effects functions, to be applied post-reduce, in order
        self.post_effects: list[Callable] = _flatten_ancestor_metadata(
            self, self.settings, "post", [], presets
        )

        # A list of callable effects functions, to be applied in order after everything else
        self.final_effects: list[Callable] = _flatten_ancestor_metadata(
            self, self.settings, "final", [], presets
        )

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

        self.ai_settings: dict[str, Any] = {
            "prompt": _ai_settings.get("prompt", self.name.replace('-', ' ') + ", abstract art"),
            "image_strength": _ai_settings.get("image_strength", 0.5),
            "cfg_scale": _ai_settings.get("cfg_scale", 15),
            "style_preset": _ai_settings.get("style_preset", "digital-art"),
            "model": _ai_settings.get("model", AI_MODEL)
        }

        self.ai_settings.update(prototype.get("ai_settings", {}))

        # This will be set to True if the call to Stable Diffusion succeeds
        self.ai_success: bool = False

    def __str__(self) -> str:
        return f"<Preset \"{self.name}\">"

    def is_generator(self) -> bool:
        """Check if this preset generates noise (has generator settings)."""
        return bool(self.generator_kwargs)

    def is_effect(self) -> bool:
        """Check if this preset applies effects (has post or final effects)."""
        return bool(self.post_effects or self.final_effects)

    def render(
        self,
        seed: int,
        tensor: tf.Tensor | None = None,
        shape: list[int] = DEFAULT_SHAPE,
        time: float = 0.0,
        speed: float = 1.0,
        filename: str = "art.png",
        with_alpha: bool = False,
        with_supersample: bool = False,
        with_fxaa: bool = False,
        with_ai: bool = False,
        with_upscale: bool = False,
        stability_model: str | None = None,
        style_filename: str | None = None,
        debug: bool = False,
    ) -> tf.Tensor | dict[str, Any]:
        """
        Render the preset to an image file or return execution graph if debug.

        Args:
            seed: Random seed for generation
            tensor: Optional input tensor to process
            shape: Output shape [height, width, channels]
            time: Time parameter for animation
            speed: Speed multiplier for animation
            filename: Output filename
            with_alpha: Include alpha channel
            with_supersample: Use 2x supersampling
            with_fxaa: Apply FXAA antialiasing
            with_ai: Use AI post-processing
            with_upscale: Use AI upscaling
            stability_model: Override AI model
            style_filename: AI style reference file
            debug: Return execution graph instead of rendering

        Returns:
            Rendered tensor, or execution graph dict if debug=True
        """

        try:
            if debug:
                rng.reset_call_count()
            tensor = multires(self, seed, tensor=tensor, shape=shape, with_supersample=with_supersample,
                              octave_effects=self.octave_effects, post_effects=self.post_effects,
                              with_fxaa=with_fxaa, with_ai=with_ai, final_effects=self.final_effects, with_alpha=with_alpha,
                              with_upscale=with_upscale, stability_model=stability_model, style_filename=style_filename,
                              time=time, speed=speed, **self.generator_kwargs)

            if debug:
                effect_names = [
                    getattr(e, '_effect_name', getattr(e, '__name__', str(e)))
                    for e in self.octave_effects + self.post_effects + self.final_effects
                ]
                return {"effects": effect_names, "rng_calls": rng.get_call_count()}

            save(tensor, filename)

        except Exception as e:
            logger.error(f"Error rendering preset named {self.name}: {e}")

            raise


def Effect(effect_name: str, **kwargs: Any) -> Callable:
    """Create a partial effect function with preset parameters.

    The returned function can be invoked with runtime parameters: tensor, shape, time, and speed.

    Args:
        effect_name: Name of the registered effect to wrap.
        **kwargs: Parameter overrides to bind to the effect function.

    Returns:
        Partial function with preset parameters bound.

    Raises:
        ValueError: If effect_name is not registered or kwargs contains invalid parameters.
    """

    if effect_name not in EFFECTS:
        raise ValueError(f'"{effect_name}" is not a registered effect name.')

    for k in kwargs:
        if k not in EFFECTS[effect_name]:
            raise ValueError(f'Effect "{effect_name}" does not accept a parameter named "{k}"')
    effect = partial(EFFECTS[effect_name]["func"], **kwargs)
    effect._effect_name = effect_name
    return effect


def _flatten_layer_entries(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_layer_entries(item))
        return flattened
    if value is None:
        return []
    return [value]


def _resolve_preset_layers(preset_name: str, presets: dict[str, Any], cache: dict[str, Any]) -> tuple[Any, ...]:
    if preset_name in cache:
        return cache[preset_name]

    prototype = presets.get(preset_name, {})
    raw_layers = prototype.get("layers", [])
    dummy_settings = defaultdict(lambda: None)
    resolved = _resolve_metadata_value(raw_layers, dummy_settings)
    cache[preset_name] = tuple(_flatten_layer_entries(resolved))
    return cache[preset_name]


def _flatten_ancestors(preset_name: str, presets: dict[str, Any], unique: dict[str, bool], ancestors: list[str], layer_cache: dict[str, Any]) -> None:
    for ancestor_name in _resolve_preset_layers(preset_name, presets, layer_cache):
        if ancestor_name not in presets:
            raise ValueError(f"\"{ancestor_name}\" was not found among the available presets.")

        # "unique" layers may only be inherited once
        if ancestor_name in unique:
            continue

        if presets[ancestor_name].get("unique"):
            unique[ancestor_name] = True

        _flatten_ancestors(ancestor_name, presets, unique, ancestors, layer_cache)

    ancestors.append(preset_name)


def _callable_param_count_uncached(value: Callable) -> int | None:
    try:
        return len(inspect.signature(value).parameters)
    except (ValueError, TypeError):
        return None


@lru_cache(maxsize=None)
def _callable_param_count(value: Callable) -> int | None:
    return _callable_param_count_uncached(value)


def _resolve_metadata_value(value: Any, settings: dict[str, Any]) -> Any:
    if callable(value):
        try:
            param_count = _callable_param_count(value)
        except TypeError:
            param_count = _callable_param_count_uncached(value)

        if param_count == 0:
            return _resolve_metadata_value(value(), settings)
        if param_count == 1:
            return _resolve_metadata_value(value(settings), settings)
        return value
    if isinstance(value, list):
        return [_resolve_metadata_value(v, settings) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_metadata_value(v, settings) for k, v in value.items()}
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


_CAMEL_PATTERN_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_PATTERN_2 = re.compile(r"([a-z0-9])([A-Z])")


def _camel_to_snake(name: str | Any) -> str | Any:
    if not isinstance(name, str):
        return name

    s1 = _CAMEL_PATTERN_1.sub(r"\1_\2", name)
    return _CAMEL_PATTERN_2.sub(r"\1_\2", s1).lower()


def _lookup_effect_definition(effect_name: str) -> tuple[str, dict[str, Any] | None]:
    if effect_name in EFFECTS:
        return effect_name, EFFECTS[effect_name]

    snake = _camel_to_snake(effect_name)

    if snake in EFFECTS:
        return snake, EFFECTS[snake]

    return effect_name, None


def _map_effect(effect: Any, settings: dict[str, Any]) -> Any:
    seen = set()
    depth = 0

    while (
        callable(effect)
        and not getattr(effect, "_effect_name", None)
        and not getattr(effect, "post_effects", None)
        and not getattr(effect, "final_effects", None)
    ):
        identity = id(effect)

        if identity in seen or depth > 64:
            raise ValueError("Runaway dynamic preset function")

        seen.add(identity)
        depth += 1
        effect = _resolve_metadata_value(effect, settings)

    if isinstance(effect, dict) and "__effectName" in effect:
        raw_name = effect["__effectName"]
        effect_name, effect_def = _lookup_effect_definition(raw_name)
        params = {}

        raw_params = effect.get("__params") or {}

        if raw_params:
            for key, value in raw_params.items():
                resolved = _resolve_metadata_value(value, settings)
                params[_camel_to_snake(key)] = resolved
        elif effect.get("args"):
            if effect_def is None:
                raise ValueError(f'"{raw_name}" is not a registered effect name.')

            keys = [k for k in effect_def.keys() if k != "func"]
            args = effect["args"]

            if len(args) > len(keys):
                raise ValueError(
                    f'Effect "{raw_name}" received {len(args)} positional arguments '
                    f"but only {len(keys)} parameters are available."
                )

            for index, arg_value in enumerate(args):
                params[keys[index]] = _resolve_metadata_value(arg_value, settings)

        return Effect(effect_name, **params)

    return effect


def _flatten_ancestor_metadata(preset: Preset, settings: dict[str, Any], key: str, default: Any, presets: dict[str, Any]) -> dict[str, Any] | list[Any]:
    """Collect and merge metadata from all ancestor presets.

    Traverses the preset inheritance chain and accumulates values for the specified
    key. Dicts are merged via update(), lists/tuples are concatenated.

    Args:
        preset: The preset whose ancestors to traverse.
        settings: Current settings dictionary for value resolution.
        key: Metadata key to collect (e.g., "octaves", "post", "final", "settings").
        default: Default value type (dict or list) to initialize accumulator.
        presets: Dictionary of all available presets.

    Returns:
        Merged metadata as dict (if default is dict) or list (otherwise).

    Raises:
        ValueError: If ancestor metadata has wrong type or evaluation fails.
    """

    if isinstance(default, dict):
        flattened_metadata = {}
    else:
        flattened_metadata = []

    for ancestor_name in preset.flattened_layers:
        ancestor = presets[ancestor_name].get(key, default)

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

        ancestor = _resolve_metadata_value(ancestor, settings)

        if not isinstance(ancestor, type(default)):
            raise ValueError(
                f"{ancestor_name}: Key \"{key}\" should be {type(default)}, not {type(ancestor)}."
            )

        if isinstance(ancestor, dict):
            flattened_metadata.update(ancestor)
        else:
            if key in ("octaves", "post", "final"):
                ancestor = [_map_effect(e, settings) for e in ancestor]
            flattened_metadata += ancestor

    return flattened_metadata


def random_member(*collections: Any) -> Any:
    """Select a random member from one or more collections, ensuring deterministic ordering.

    Accepts enums, enum lists, or regular collections. Multiple collections are merged
    before selection. Order is deterministic to ensure consistent RNG behavior.

    RNG: Single call to :func:`rng.random_int` to choose the index.

    Args:
        *collections: One or more iterables (enums, lists, sets, etc.) to select from.

    Returns:
        A randomly selected member from the merged collections.

    Raises:
        ValueError: If any argument is not iterable.
    """

    collection = []

    for c in collections:
        if not hasattr(c, "__iter__"):
            raise ValueError(f"random_member(arg) should be iterable (collection, enum list, or enum)")

        if isinstance(c, EnumMeta):
            collection += list(c)

        # maybe it's a list of enum members
        elif isinstance(next(iter(c), None), Enum):
            collection += [s[1] for s in sorted([(m.name if m is not None else "", m) for m in c])]

        else:
            # make sure order is deterministic
            collection += sorted(c)

    return collection[rng.random_int(0, len(collection) - 1)]


def coin_flip() -> bool:
    """Return a random boolean (True/False with equal probability).

    RNG: Single call to :func:`rng.random_int` with range [0, 1].

    Returns:
        Randomly selected True or False.
    """
    return bool(rng.random_int(0, 1))  # RNG[1]


def enum_range(a: Enum, b: Enum) -> list[Enum]:
    """Return a list of enum members within the specified inclusive numeric value range.

    Args:
        a: Starting enum member (inclusive).
        b: Ending enum member (inclusive).

    Returns:
        List of all enum members from a.value to b.value (inclusive).
    """

    enum_class = type(a)

    members = []

    for i in range(a.value, b.value + 1):
        members.append(enum_class(i))

    return members


def reload_presets(presets: Callable[[], dict[str, Any]]) -> None:
    """Re-evaluate all presets and categorize them as generators or effects.

    Clears existing preset caches and reloads from the provided factory function.
    This should be called after changing the random seed to get fresh evaluations.

    Args:
        presets: Factory function that returns a dictionary of preset definitions.

    Raises:
        ValueError: If any preset fails to initialize or validate.
    """

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


def stash(key: str, value: Any | None = None) -> Any:
    """Store and retrieve values for cross-reference within preset lambda functions.

    Useful for holding intermediate values that need to be shared across multiple
    effect definitions within the same preset.

    Args:
        key: Identifier for the stashed value.
        value: Value to store (if provided).

    Returns:
        The currently stored value for the given key.
    """

    global _STASH

    if value is not None:
        _STASH[key] = value

    return _STASH[key]


class UnusedKeys(Exception):
    """Exception raised when a preset has keys that aren't being used."""

    pass


class SettingsDict(UserDict):
    """dict, but it makes sure the caller eats everything on their plate."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.__accessed__ = {}

        super().__init__(*args, **kwargs)

    def __getitem__(self, key: str) -> Any:
        self.__accessed__[key] = True

        return super().__getitem__(key)

    def was_accessed(self, key: str) -> bool:
        return key in self.__accessed__

    def raise_if_unaccessed(self, unused_okay: list[str] | None = None) -> None:
        keys = []

        for key in self:
            if not self.was_accessed(key) and (unused_okay is None or key not in unused_okay):
                keys.append(key)

        if keys:
            if len(keys) == 1:
                raise UnusedKeys(f"Settings key \"{keys[0]}\" (value: {self[keys[0]]}) is unused. This is usually human error.")
            else:
                raise UnusedKeys(f"Settings keys {keys} are unused. This is usually human error.")
