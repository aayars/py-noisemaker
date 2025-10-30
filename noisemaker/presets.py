from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import noisemaker.rng as random
from noisemaker.composer import Preset as ComposerPreset
from noisemaker.dsl import parse_preset_dsl


@lru_cache(maxsize=1)
def _cached_dsl_presets() -> dict[str, Any]:
    """Load and cache presets from the DSL file with a deterministic seed.

    Uses a fixed seed (0) to ensure preset evaluation is consistent across runs.
    Restores the original RNG seed after loading.

    Returns:
        Dictionary of parsed preset definitions from presets.dsl.
    """
    dsl_path = Path(__file__).resolve().parent.parent / "dsl" / "presets.dsl"

    seed_before = random.get_seed()
    random.set_seed(0)
    try:
        with open(dsl_path, encoding="utf-8") as fh:
            return parse_preset_dsl(fh.read())
    finally:
        random.set_seed(seed_before)


def PRESETS() -> dict[str, Any]:
    """Get a fresh copy of all presets with synchronized RNG state.

    Returns a deep copy to prevent mutations from affecting the cached version.
    Includes mandatory RNG calls to maintain Python/JS parity.

    Returns:
        Dictionary of all available preset definitions.
    """
    presets = deepcopy(_cached_dsl_presets())

    # This is somehow keeping the Python and JS ports in sync, removing it from both
    # places breaks parity. WTF
    random.random()
    random.random()
    random.random()

    return presets


def Preset(preset_name: str, *, settings: dict[str, Any] | None = None) -> ComposerPreset:
    """Create a Preset instance from the DSL preset collection.

    Convenience factory function for creating preset objects with optional settings overrides.

    Args:
        preset_name: Name of the preset to load from presets.dsl.
        settings: Optional dictionary of setting overrides.

    Returns:
        Configured Preset object ready for rendering.

    Raises:
        ValueError: If preset_name is not found or preset definition is invalid.
    """
    return ComposerPreset(preset_name, presets=PRESETS(), settings=settings)
