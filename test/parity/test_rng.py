import json
from pathlib import Path

from noisemaker import rng


fixtures = Path(__file__).resolve().parent.parent / "fixtures" / "rng"


def _load():
    for fixture in fixtures.glob("seed_*.json"):
        seed = int(fixture.stem.split("_")[1])
        values = json.loads(fixture.read_text())
        yield seed, values


def test_random():
    for seed, expected in _load():
        rng.set_seed(seed)
        for i, val in enumerate(expected):
            assert abs(rng.random() - val) < 1e-9, f"seed {seed} index {i}"


def test_random_int():
    for seed, expected in _load():
        rng.set_seed(seed)
        for i, val in enumerate(expected):
            assert rng.random_int(0, 99) == int(val * 100), f"seed {seed} index {i}"


def test_choice():
    seq = list(range(10))
    for seed, expected in _load():
        rng.set_seed(seed)
        for i, val in enumerate(expected):
            assert rng.choice(seq) == seq[int(val * len(seq))], f"seed {seed} index {i}"

