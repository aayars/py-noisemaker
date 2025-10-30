"""Effect decorator for Noisemaker Composer Presets"""

from __future__ import annotations

import inspect
from typing import Any, Callable

EFFECTS: dict[str, dict[str, Any]] = {}


def effect(*args: str) -> Callable:
    """
    Function decorator for declaring composable effects.

    Registers effect functions with their parameter defaults for use in Composer presets.
    Validates that effects accept required "time" and "speed" keyword arguments.

    Args:
        *args: Optional effect name. If not provided, uses function name.

    Returns:
        Decorator function that registers the effect and returns the original function

    Raises:
        ValueError: If function doesn't accept required "time" or "speed" parameters,
                   or if keyword parameter counts don't match defaults
    """

    def decorator_fn(func: Callable) -> Callable:
        argspec = inspect.getfullargspec(func)

        params = argspec.args

        for param in ["time", "speed"]:
            if param not in params:
                raise ValueError(f'{func.__name__}() needs to accept a "{param}" keyword arg. Please add it to the function signature.')

        # All effects respond to "tensor", "shape". Removing these non-keyword args should make params the same length as defaults.
        params.remove("tensor")
        params.remove("shape")

        defaults = argspec.defaults or ()
        if params and len(params) != len(defaults):
            raise ValueError(f'Expected {len(defaults)} keyword params to "{func.__name__}", but got {len(params)}.')

        # Register effect name and params
        name = args[0] if args else func.__name__
        EFFECTS[name] = dict((params[i], defaults[i]) for i in range(len(params)))
        EFFECTS[name]["func"] = func

        return func

    return decorator_fn
