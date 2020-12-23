"""Effect decorator for Noisemaker Composer Presets"""

import inspect

EFFECTS = {}


def effect(*args):
    """Function decorator for declaring composable effects."""

    def decorator_fn(func):
        argspec = inspect.getfullargspec(func)

        params = argspec.args

        for param in ["time", "speed"]:
            if param not in params:
                raise ValueError(f'{func.__name__}() needs to accept a "{param}" keyword arg. Please add it to the function signature.')

        # All effects respond to "tensor", "shape". Removing these non-keyword args should make params the same length as defaults.
        params.remove("tensor")
        params.remove("shape")

        if params and len(params) != len(argspec.defaults):
            raise ValueError(f'Expected {len(argspec.defaults)} keyword params to "{func.__name__}", but got {len(params)}.')

        # Register effect name and params
        name = args[0] if args else func.__name__
        EFFECTS[name] = dict((params[i], argspec.defaults[i]) for i in range(len(params)))
        EFFECTS[name]["func"] = func

        return func

    return decorator_fn
