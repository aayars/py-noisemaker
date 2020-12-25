from noisemaker.composer import Preset
from noisemaker.presets import PRESETS


def test_presets():
    problems = []

    for preset_name in PRESETS:
        try:
            preset = Preset(preset_name, PRESETS)

        except Exception as e:
            problems.append(f"{preset_name} has an error: {e}")

    if problems:
        raise Exception("Some presets have errors:\n    " + "\n    ".join(problems))
