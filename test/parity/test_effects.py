import json
from pathlib import Path

import pytest

from noisemaker import effects, value


def test_worms_params_fixture():
    fixture = Path(__file__).resolve().parent.parent / 'fixtures' / 'effects' / 'worms.json'
    with open(fixture) as f:
        expected = json.load(f)
    value.set_seed(1)
    params = effects.worms_params([4, 4, 1])
    for key in expected:
        assert params[key] == pytest.approx(expected[key], abs=1e-6)
