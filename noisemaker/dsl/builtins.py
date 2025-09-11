from noisemaker.constants import *  # noqa: F401,F403
import noisemaker.constants as constants
from noisemaker.composer import coin_flip as _coin_flip, random_member as _random_member, stash as _stash
import noisemaker.rng as _random
import noisemaker.masks as _masks

surfaces = { }

def coin_flip(*args):
    if len(args) != 0:
        raise ValueError(f"coin_flip() takes no arguments, received {len(args)}")
    return _coin_flip()

def _enum_range(a, b):
    out = []
    i = int(a)
    b = int(b)
    while i <= b:
        out.append(i)
        i += 1
    return out

def enum_range(*args):
    if len(args) != 2:
        raise ValueError(f"enum_range(a, b) requires exactly 2 arguments, received {len(args)}")
    a, b = args
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise ValueError("enum_range(a, b) requires numeric arguments")
    return _enum_range(a, b)

def random_member(*collections):
    if len(collections) == 0:
        raise ValueError("random_member() requires at least one iterable argument")
    return _random_member(*collections)

def stash(*args):
    if len(args) == 0 or len(args) > 2:
        raise ValueError(f"stash(key[, value]) expects 1 or 2 arguments, received {len(args)}")
    key = args[0]
    value = args[1] if len(args) == 2 else None
    if not isinstance(key, str):
        raise ValueError('stash(key[, value]) key must be a string')
    return _stash(key, value)

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

operations = {
    "coin_flip": coin_flip,
    "random_member": random_member,
    "enum_range": enum_range,
    "stash": stash,
    "random": random,
    "random_int": random_int,
    "mask_freq": mask_freq,
}

enums = constants

defaultContext = {
    "surfaces": surfaces,
    "operations": operations,
    "enums": enums,
}
