from noisemaker.composer import Preset
from noisemaker.presets import PRESETS
import pytest


@pytest.mark.parametrize(
    "use_dsl",
    [False, pytest.param(True, marks=pytest.mark.xfail(reason="DSL presets under development"))],
)
def test_presets(use_dsl):
    problems = []

    presets = PRESETS(use_dsl=use_dsl)

    for preset_name in presets:
        try:
            preset = Preset(preset_name, presets)
        except Exception as e:
            problems.append(f"{preset_name} has an error: {e}")

    if problems:
        raise Exception("Some presets have errors:\n    " + "\n    ".join(problems))
