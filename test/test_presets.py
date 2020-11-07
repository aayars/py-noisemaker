from noisemaker.presets import PRESETS


def test_presets():
    problems = []

    for preset in PRESETS:
        try:
            PRESETS[preset]()

        except Exception as e:
            problems.append(f"{preset} had an issue: {e}")

    if problems:
        raise Exception("Problems evaluating presets:\n    " + "\n    ".join(problems))
