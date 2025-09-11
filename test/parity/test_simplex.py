import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from noisemaker import simplex as simplex_module

SEEDS = [1, 2, 3]


def _hash(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


@pytest.mark.parametrize("seed", SEEDS)
def test_simplex_parity(seed):
    script = Path(__file__).with_name("simplex_integration.js")

    os, data = simplex_module.from_seed(seed)
    r_py = simplex_module.random(0.25, seed, 1)
    dt = 1e-3
    d0_py = (simplex_module.random(dt, seed, 1) - simplex_module.random(0.0, seed, 1)) / dt
    d1_py = (simplex_module.random(1 + dt, seed, 1) - simplex_module.random(1.0, seed, 1)) / dt

    result = subprocess.run(["node", str(script), str(seed)], check=True, capture_output=True, text=True)
    js = json.loads(result.stdout)

    assert js["perm"] == data["perm"][:10]
    assert js["perm_grad"] == data["perm_grad"][:10]
    assert abs(js["random"] - r_py) < 1e-6
    assert abs(js["d0"] - d0_py) < 1e-6
    assert abs(js["d1"] - d1_py) < 1e-6
    assert abs(d0_py - d1_py) < 1e-6

    tensor = simplex_module.simplex((128, 128, 3), seed=seed, time=0, speed=1, as_np=True)
    assert tensor.shape == (128, 128, 3)
    assert _hash(tensor) == js["hash"]

    tile2 = simplex_module.simplex((128, 128, 3), seed=seed, time=0, speed=1, as_np=True)
    assert np.allclose(tensor, tile2, atol=1e-6)
