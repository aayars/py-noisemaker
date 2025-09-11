import json
import subprocess
from pathlib import Path


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


def generate_hashes():
    script = Path(__file__).with_name("generate_hashes.js")
    result = subprocess.run(
        ["node", str(script)], check=True, capture_output=True, text=True
    )
    data = json.loads(result.stdout)
    return _int_keys(data)
