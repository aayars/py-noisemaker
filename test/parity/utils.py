import base64
import json
import subprocess
from pathlib import Path

import numpy as np


def _int_keys(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                key = int(k)
            except (ValueError, TypeError):
                key = k
            out[key] = _int_keys(v)
        return out
    if isinstance(obj, list):
        return [_int_keys(v) for v in obj]
    return obj
_CACHE = None


def generate_hashes():
    global _CACHE
    if _CACHE is None:
        script = Path(__file__).with_name("generate_hashes.js")
        result = subprocess.run(
            ["node", str(script)], check=True, capture_output=True, text=True
        )
        data = json.loads(result.stdout)
        _CACHE = _int_keys(data)
    return _CACHE


def js_generator(name: str, seed: int) -> np.ndarray:
    """Invoke the JavaScript implementation for a generator and return its tensor.

    The returned array has shape (128, 128, 3) and dtype float32.
    """
    script = Path(__file__).with_name("run_generators.js")
    result = subprocess.run(
        ["node", str(script), name, str(seed)],
        check=True,
        capture_output=True,
        text=True,
    )
    data = base64.b64decode(result.stdout)
    return np.frombuffer(data, dtype="<f4").reshape(128, 128, 3)


def js_effect(name: str, seed: int) -> np.ndarray:
    """Invoke the JavaScript implementation for an effect and return its tensor.

    The effect is applied to a basic generator output with hueRotation=0 to
    mirror the Python parity tests. The returned array has shape
    (128, 128, 3) and dtype float32.
    """

    script = Path(__file__).with_name("run_effects.js")
    result = subprocess.run(
        ["node", str(script), name, str(seed)],
        check=True,
        capture_output=True,
        text=True,
    )
    data = base64.b64decode(result.stdout)
    return np.frombuffer(data, dtype="<f4").reshape(128, 128, 3)
