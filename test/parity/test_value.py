import json
from pathlib import Path

import pytest

from noisemaker import value


def test_value_noise_fixture():
    fixture = Path(__file__).resolve().parent.parent / 'fixtures' / 'value' / 'seed_1.json'
    with open(fixture) as f:
        expected = json.load(f)
    value.set_seed(1)
    result = value.value_noise(64).numpy().tolist()
    assert result == pytest.approx(expected, abs=1e-6)
