from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import noisemaker.rng as random

from noisemaker.composer import Preset as ComposerPreset
from noisemaker.dsl import parse_preset_dsl

@lru_cache(maxsize=1)
def _cached_dsl_presets():
    dsl_path = Path(__file__).resolve().parent.parent / "dsl" / "presets.dsl"

    seed_before = random.get_seed()
    random.set_seed(0)
    try:
        with open(dsl_path, "r", encoding="utf-8") as fh:
            return parse_preset_dsl(fh.read())
    finally:
        random.set_seed(seed_before)


def PRESETS():
    presets = deepcopy(_cached_dsl_presets())

    # This is somehow keeping the Python and JS ports in sync, removing it from both 
    # places breaks parity. WTF
    random.random()
    random.random()
    random.random()

    return presets


def Preset(preset_name, *, settings=None):
    return ComposerPreset(
        preset_name, presets=PRESETS(), settings=settings
    )
