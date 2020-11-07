from noisemaker.presets import EFFECTS_PRESETS, PRESETS


def test_presets():
    problems = []

    for preset in list(EFFECTS_PRESETS) + list(PRESETS):
        try:
            if preset in PRESETS:
                PRESETS[preset]()

            else:
                EFFECTS_PRESETS[preset]()

        except Exception as e:
            problems.append(f"{preset} had an issue: {e}")

    if problems:
        raise Exception("Problems evaluating presets:\n    " + "\n    ".join(problems))
