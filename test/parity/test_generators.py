import json
from pathlib import Path

import pytest

from noisemaker import generators, value, rng


fixtures = Path(__file__).resolve().parent.parent / "fixtures" / "generators"


def _load():
    for fixture in fixtures.glob("basic_seed_*.json"):
        seed = int(fixture.stem.split("_")[2])
        with open(fixture) as f:
            expected = json.load(f)
        yield seed, expected


def test_basic_fixture():
    for seed, expected in _load():
        rng.set_seed(seed)
        value.set_seed(seed)
        tensor = generators.basic(2, [4, 4, 3], hue_rotation=0)
        result = tensor.numpy().flatten().tolist()
        assert result == pytest.approx(expected, abs=1e-6)
