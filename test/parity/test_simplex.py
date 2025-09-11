import json
import subprocess
from pathlib import Path

import numpy as np

from noisemaker import simplex as simplex_module

seeds = [1, 2, 3]


def test_simplex_parity():
    root = Path(__file__).resolve().parents[2]
    script = root / "test" / "parity" / "simplex_integration.js"

    for seed in seeds:
        os, data = simplex_module.from_seed(seed)
        r_py = simplex_module.random(0.25, seed, 1)
        dt = 1e-3
        d0_py = (simplex_module.random(dt, seed, 1) - simplex_module.random(0.0, seed, 1)) / dt
        d1_py = (simplex_module.random(1 + dt, seed, 1) - simplex_module.random(1.0, seed, 1)) / dt

        out = subprocess.check_output(["node", str(script), str(seed)], cwd=root, text=True)
        js = json.loads(out)

        assert js["perm"] == data["perm"][:10]
        assert js["perm_grad"] == data["perm_grad"][:10]
        assert abs(js["random"] - r_py) < 1e-6
        assert abs(js["d0"] - d0_py) < 1e-6
        assert abs(js["d1"] - d1_py) < 1e-6

        assert abs(d0_py - d1_py) < 1e-6
        tile1 = simplex_module.simplex((4, 4, 1), seed=seed, time=0, speed=1, as_np=True)
        tile2 = simplex_module.simplex((4, 4, 1), seed=seed, time=0, speed=1, as_np=True)
        assert np.allclose(tile1, tile2, atol=1e-6)
