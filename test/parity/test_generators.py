import json
from pathlib import Path

import pytest

from noisemaker import generators, value, rng


def test_basic_fixture():
    fixture = Path(__file__).resolve().parent.parent / 'fixtures' / 'generators' / 'basic_seed_1.json'
    with open(fixture) as f:
        expected = json.load(f)
    rng.set_seed(1)
    value.set_seed(1)
    tensor = generators.basic(2, [4, 4, 3], hue_rotation=0)
    result = tensor.numpy().flatten().tolist()
    assert result == pytest.approx(expected, abs=1e-6)
