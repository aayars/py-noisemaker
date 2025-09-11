from noisemaker import rng

from .utils import generate_hashes

DATA = generate_hashes()["rng"]


def test_random():
    for seed, expected in DATA.items():
        rng.set_seed(seed)
        for i, val in enumerate(expected):
            assert abs(rng.random() - val) < 1e-9, f"seed {seed} index {i}"


def test_random_int():
    for seed, expected in DATA.items():
        rng.set_seed(seed)
        for i, val in enumerate(expected):
            assert rng.random_int(0, 99) == int(val * 100), f"seed {seed} index {i}"


def test_choice():
    seq = list(range(10))
    for seed, expected in DATA.items():
        rng.set_seed(seed)
        for i, val in enumerate(expected):
            assert rng.choice(seq) == seq[int(val * len(seq))], f"seed {seed} index {i}"

