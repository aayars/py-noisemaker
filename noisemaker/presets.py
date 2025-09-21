from pathlib import Path
import noisemaker.rng as random

from noisemaker.composer import (
    Effect,
    Preset as ComposerPreset,
    coin_flip,
    enum_range,
    random_member,
    stash,
)
from noisemaker.constants import (
    ColorSpace as color,
    DistanceMetric as distance,
    InterpolationType as interp,
    OctaveBlending as blend,
    PointDistribution as point,
    ValueDistribution as distrib,
    ValueMask as mask,
    VoronoiDiagramType as voronoi,
    WormBehavior as worms,
)
from noisemaker.palettes import PALETTES
from noisemaker.dsl import parse_preset_dsl

import noisemaker.masks as masks


def _dsl_presets():
    dsl_path = Path(__file__).resolve().parent.parent / "dsl" / "presets.dsl"

    seed_before = random.get_seed()
    random.set_seed(0)
    try:
        with open(dsl_path, "r", encoding="utf-8") as fh:
            presets = parse_preset_dsl(fh.read())
    finally:
        random.set_seed(seed_before)

    # XXX Get rid of this
    random.random()
    random.random()
    random.random()

    return presets


def PRESETS():
    return _dsl_presets()


def Preset(preset_name, *, settings=None):
    return ComposerPreset(
        preset_name, presets=PRESETS(), settings=settings
    )
