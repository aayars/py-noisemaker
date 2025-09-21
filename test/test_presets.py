from noisemaker.composer import Preset
from noisemaker.presets import PRESETS
import pytest


def test_presets():
    problems = []

    presets = PRESETS()

    for preset_name in presets:
        try:
            preset = Preset(preset_name, presets)
        except Exception as e:
            problems.append(f"{preset_name} has an error: {e}")

    if problems:
        raise Exception("Some presets have errors:\n    " + "\n    ".join(problems))
