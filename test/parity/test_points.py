import pytest

from noisemaker import points, rng

from .utils import generate_hashes

DATA = generate_hashes()["points"]
SEEDS = list(DATA.keys())


@pytest.mark.parametrize("seed", SEEDS)
def test_cloud_points_parity(seed):
    rng.set_seed(seed)
    x, y = points.cloud_points(4)
    expected = DATA[seed]
    assert x == pytest.approx(expected["x"], abs=1e-6)
    assert y == pytest.approx(expected["y"], abs=1e-6)
