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


def js_generator(name: str, seed: int, **options) -> tuple[np.ndarray, int]:
    """Invoke the JavaScript implementation for a generator.

    Returns a tuple of ``(tensor, call_count)`` where ``tensor`` is reshaped to
    the array returned by the JavaScript process and ``call_count`` is the
    number of RNG calls consumed by the JavaScript run.
    """

    script = Path(__file__).with_name("run_generators.js")
    cmd = ["node", str(script), name, str(seed)]
    if options:
        encoded = base64.b64encode(json.dumps(options).encode()).decode()
        cmd.append(encoded)
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    shape = tuple(payload.get("shape", (128, 128, 3)))
    tensor = np.frombuffer(
        base64.b64decode(payload["tensor"]), dtype="<f4"
    ).reshape(shape)
    return tensor, payload.get("callCount", 0)


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


def js_rng(fn: str, seed: int, *args, scope: str = "global") -> dict:
    """Invoke the JavaScript RNG function and return its JSON output."""

    script = Path(__file__).with_name("run_rng.js")
    cmd = ["node", str(script), scope, fn, str(seed), *map(str, args)]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def js_voronoi(params: dict) -> np.ndarray:
    """Invoke the JavaScript implementation for Voronoi with arbitrary params."""

    script = Path(__file__).with_name("run_voronoi.js")
    encoded = base64.b64encode(json.dumps(params).encode()).decode()
    result = subprocess.run(
        ["node", str(script), encoded],
        check=True,
        capture_output=True,
        text=True,
    )
    data = base64.b64decode(result.stdout)
    return np.frombuffer(data, dtype="<f4").reshape(128, 128, 3)


def js_preset_settings(name: str, seed: int) -> dict:
    """Invoke the JavaScript implementation to evaluate preset settings."""

    script = Path(__file__).with_name("run_preset_settings.js")
    result = subprocess.run(
        ["node", str(script), name, str(seed)],
        check=True,
        capture_output=True,
        text=True,
    )
    return _int_keys(json.loads(result.stdout))


def js_dsl(
    op: str, source: str, seed: int | None = None, count: int | None = None
):
    """Invoke the JavaScript DSL helpers and return their JSON result."""

    script = Path(__file__).with_name("run_dsl.js")
    encoded = base64.b64encode(source.encode()).decode()
    cmd = ["node", str(script), op, encoded]
    if seed is not None:
        cmd.append(str(seed))
        if count is not None:
            cmd.append(str(count))
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)
