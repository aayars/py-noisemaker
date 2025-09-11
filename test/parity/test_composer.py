import pytest

from noisemaker import rng
from noisemaker.composer import Preset
from noisemaker.presets import PRESETS

from .utils import generate_hashes

DATA = generate_hashes()["composer"]
CANONICAL = list(DATA.keys())


@pytest.mark.parametrize("name", CANONICAL)
def test_exec_graph(name):
    seed = 1
    rng.set_seed(seed)
    preset = Preset(name, PRESETS())
    graph = preset.render(seed, shape=[128, 128, 3], debug=True)
    assert graph["effects"] == DATA[name]["effects"]
    assert graph["rng_calls"] == DATA[name]["rng_calls"]
