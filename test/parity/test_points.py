import json
from pathlib import Path

import pytest

from noisemaker import points, rng


def test_cloud_points_fixture():
    fixture = Path(__file__).resolve().parent.parent / 'fixtures' / 'points' / 'seed_1_freq4.json'
    with open(fixture) as f:
        data = json.load(f)
    rng.set_seed(1)
    x, y = points.cloud_points(4)
    assert x == pytest.approx(data['x'], abs=1e-6)
    assert y == pytest.approx(data['y'], abs=1e-6)
