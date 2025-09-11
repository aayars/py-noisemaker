import pytest

from noisemaker import value

from .utils import generate_hashes

DATA = generate_hashes()["value"]
SEEDS = list(DATA.keys())


@pytest.mark.parametrize("seed", SEEDS)
def test_value_noise_parity(seed):
    value.set_seed(seed)
    result = value.value_noise(64).numpy().tolist()
    assert result == pytest.approx(DATA[seed], abs=1e-6)
