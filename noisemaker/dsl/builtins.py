from noisemaker.constants import *  # noqa: F401,F403
import noisemaker.constants as constants
import types

from enum import Enum

from noisemaker.composer import (
    coin_flip as _coin_flip,
    enum_range as _enum_range,
    random_member as _random_member,
    stash as _stash,
    Preset as _Preset,
)
import noisemaker.rng as _random
import noisemaker.masks as _masks
from noisemaker.palettes import PALETTES as _PALETTES


class _SettingsSurface:
    def __getattr__(self, name):
        return lambda settings: settings[name]


surfaces = {"settings": _SettingsSurface()}

def coin_flip(*args):
    if len(args) != 0:
        raise ValueError(f"coin_flip() takes no arguments, received {len(args)}")
    return _coin_flip()

def enum_range(*args):
    if len(args) != 2:
        raise ValueError(f"enum_range(a, b) requires exactly 2 arguments, received {len(args)}")
    a, b = args
    if isinstance(a, Enum) and isinstance(b, Enum):
        return _enum_range(a, b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return list(range(int(a), int(b) + 1))
    raise ValueError("enum_range(a, b) requires numeric arguments")

def random_member(*collections):
    if len(collections) == 0:
        raise ValueError("random_member() requires at least one iterable argument")
    return _random_member(*collections)

def stash(*args):
    if len(args) == 0 or len(args) > 2:
        raise ValueError(f"stash(key[, value]) expects 1 or 2 arguments, received {len(args)}")
    key = args[0]
    if not isinstance(key, str):
        raise ValueError('stash(key[, value]) key must be a string')
    value = args[1] if len(args) == 2 else None

    def _thunk(settings=None):
        try:
            resolved = value(settings) if callable(value) else value
            return _stash(key, resolved)
        except KeyError:
            return None

    return _thunk

def random(*args):
    if len(args) != 0:
        raise ValueError(f"random() takes no arguments, received {len(args)}")
    return _random.random()

def random_int(*args):
    if len(args) != 2:
        raise ValueError(f"random_int(a, b) requires exactly 2 arguments, received {len(args)}")
    a, b = args
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise ValueError("random_int(a, b) requires numeric arguments")
    return _random.randint(int(a), int(b))

def mask_freq(*args):
    if len(args) != 2:
        raise ValueError(f"mask_freq(mask, repeat) requires exactly 2 arguments, received {len(args)}")
    mask, repeat = args
    shape = _masks.mask_shape(mask)
    return [int(i * 0.5 + i * repeat) for i in shape[0:2]]

def preset(*args):
    if len(args) == 0 or len(args) > 2:
        raise ValueError(f"preset(name[, settings]) expects 1 or 2 arguments, received {len(args)}")
    name = args[0]
    if not isinstance(name, str):
        raise ValueError('preset(name[, settings]) name must be a string')
    settings = args[1] if len(args) == 2 else {}

    def _thunk(parent_settings=None):
        from noisemaker.presets import PRESETS as _PRESETS
        resolved = {}
        for k, v in settings.items():
            resolved[k] = v(parent_settings) if callable(v) else v
        return _Preset(name, _PRESETS(), resolved)

    return _thunk


# Mark non-deterministic operations as thunks so the evaluator defers their
# execution until settings are resolved, mirroring the JavaScript
# implementation.
coin_flip.__thunk = True
random_member.__thunk = True
random.__thunk = True
random_int.__thunk = True

# Unlike the JavaScript implementation, eagerly evaluate these helpers so tests
# can inspect their return values directly.

operations = {
    "coin_flip": coin_flip,
    "random_member": random_member,
    "enum_range": enum_range,
    "stash": stash,
    "random": random,
    "random_int": random_int,
    "mask_freq": mask_freq,
    "preset": preset,
    # expose helper functions used via enum method-style calls
    "distanceMetricAbsoluteMembers": constants.DistanceMetric.absolute_members,
    "distanceMetricAll": constants.DistanceMetric.all,
    "colorSpaceMembers": constants.ColorSpace.color_members,
    "valueMaskProceduralMembers": constants.ValueMask.procedural_members,
    "valueMaskGridMembers": constants.ValueMask.grid_members,
    "valueMaskGlyphMembers": constants.ValueMask.glyph_members,
    "valueMaskNonproceduralMembers": constants.ValueMask.nonprocedural_members,
    "valueMaskRgbMembers": constants.ValueMask.rgb_members,
    "circularMembers": constants.PointDistribution.circular_members,
    "gridMembers": constants.PointDistribution.grid_members,
    "wormBehaviorAll": constants.WormBehavior.all,
    "maskShape": _masks.mask_shape,
    "squareMasks": _masks.square_masks,
}

# Merge constants with PALETTES so the DSL can reference the palette table as
# an enum, matching the behaviour of the JavaScript implementation.
_enum_dict = {name: getattr(constants, name) for name in dir(constants) if not name.startswith("_")}
_enum_dict["PALETTES"] = _PALETTES
enums = types.SimpleNamespace(**_enum_dict)
enumMethods = {
    "DistanceMetric": {
        "absolute_members": operations["distanceMetricAbsoluteMembers"],
        "all": operations["distanceMetricAll"],
    },
    "PointDistribution": {
        "grid_members": lambda: operations["gridMembers"](),
        "circular_members": lambda: operations["circularMembers"](),
    },
    "ColorSpace": {
        "color_members": operations["colorSpaceMembers"],
    },
    "ValueMask": {
        "procedural_members": lambda: operations["valueMaskProceduralMembers"](),
        "grid_members": lambda: operations["valueMaskGridMembers"](),
        "glyph_members": lambda: operations["valueMaskGlyphMembers"](),
        "nonprocedural_members": lambda: operations["valueMaskNonproceduralMembers"](),
        "rgb_members": lambda: operations["valueMaskRgbMembers"](),
    },
    "WormBehavior": {
        "all": lambda: operations["wormBehaviorAll"](),
    },
    "masks": {
        "mask_shape": operations["maskShape"],
        "square_masks": lambda: operations["squareMasks"](),
    },
}

defaultContext = {
    "surfaces": surfaces,
    "operations": operations,
    "enums": enums,
    "enumMethods": enumMethods,
}
