import json
from pathlib import Path

import pytest

from noisemaker import rng
from noisemaker.composer import Preset
from noisemaker.presets import PRESETS

fixture = Path(__file__).resolve().parent.parent / "fixtures" / "composer.json"
with open(fixture) as f:
    EXPECTED = json.load(f)

CANONICAL = list(EXPECTED.keys())


@pytest.mark.parametrize("name", CANONICAL)
def test_exec_graph(name):
    seed = 1
    rng.set_seed(seed)
    preset = Preset(name, PRESETS())
    graph = preset.render(seed, shape=[8, 8, 3], debug=True)
    assert graph["effects"] == EXPECTED[name]["effects"]
    assert graph["rng_calls"] == EXPECTED[name]["rng_calls"]
