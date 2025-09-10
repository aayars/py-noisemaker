from noisemaker.constants import *  # noqa: F401,F403
import noisemaker.constants as constants
from noisemaker.composer import coin_flip as _coin_flip, random_member as _random_member, stash as _stash
import noisemaker.rng as _random

surfaces = { }

def coinFlip(*args):
    if len(args) != 0:
        raise ValueError(f"coinFlip() takes no arguments, received {len(args)}")
    return _coin_flip()

def _enum_range(a, b):
    out = []
    i = int(a)
    b = int(b)
    while i <= b:
        out.append(i)
        i += 1
    return out

def enumRange(*args):
    if len(args) != 2:
        raise ValueError(f"enumRange(a, b) requires exactly 2 arguments, received {len(args)}")
    a, b = args
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise ValueError("enumRange(a, b) requires numeric arguments")
    return _enum_range(a, b)

def randomMember(*collections):
    if len(collections) == 0:
        raise ValueError("randomMember() requires at least one iterable argument")
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

def randomInt(*args):
    if len(args) != 2:
        raise ValueError(f"randomInt(a, b) requires exactly 2 arguments, received {len(args)}")
    a, b = args
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise ValueError("randomInt(a, b) requires numeric arguments")
    return _random.randint(int(a), int(b))

operations = {
    "coinFlip": coinFlip,
    "randomMember": randomMember,
    "enumRange": enumRange,
    "stash": stash,
    "random": random,
    "randomInt": randomInt,
}

enums = constants

defaultContext = {
    "surfaces": surfaces,
    "operations": operations,
    "enums": enums,
}
