#!/usr/bin/env python3
"""Generate simplex noise images via Python and JS implementations, compare pixel diffs,
   and store Python baselines to guard against regressions."""

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from opensimplex import OpenSimplex
from PIL import Image


def python_simplex(width: int, height: int, seed: int, time: float = 0.25, speed: float = 1.0) -> np.ndarray:
    """Return a height x width array of simplex noise values in [0,1]."""
    angle = math.tau * time
    z = math.cos(angle) * speed
    simp = OpenSimplex(seed)
    arr = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            val = simp.noise3d(x, y, z)
            arr[y, x] = (val + 1.0) / 2.0
    return arr


def js_simplex(width: int, height: int, seed: int, time: float = 0.25, speed: float = 1.0) -> np.ndarray:
    """Invoke the JS simplex implementation and return a numpy array of values in [0,1]."""
    js_code = f"""
import {{ simplex }} from './js/noisemaker/simplex.js';
const shape = [{height}, {width}, 1];
const tensor = simplex(shape, {{ seed: {seed}, time: {time}, speed: {speed} }});
console.log(JSON.stringify(Array.from(tensor.read())));
"""
    result = subprocess.run(
        ['node', '--input-type=module', '-'],
        input=js_code,
        text=True,
        capture_output=True,
        check=True,
    )
    data = np.array(json.loads(result.stdout), dtype=np.float32)
    return data.reshape(height, width)


def compare(py: np.ndarray, js: np.ndarray) -> float:
    """Return the max absolute difference between two arrays."""
    return float(np.max(np.abs(py - js)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='*', type=int, default=[0, 1], help='Seeds to test')
    parser.add_argument('--width', type=int, default=2, help='Image width')
    parser.add_argument('--height', type=int, default=2, help='Image height')
    parser.add_argument('--time', type=float, default=0.25, help='Time value for 3D noise')
    parser.add_argument('--fixtures', type=Path, default=Path('test/image-fixtures'), help='Directory for baseline images')
    parser.add_argument('--threshold', type=float, default=1e-6, help='Max allowed pixel diff')
    args = parser.parse_args()

    args.fixtures.mkdir(parents=True, exist_ok=True)
    failed = False

    for seed in args.seeds:
        py_arr = python_simplex(args.width, args.height, seed, time=args.time)
        img = Image.fromarray((py_arr * 255).astype(np.uint8))
        baseline = args.fixtures / f'simplex-{seed}.png'
        img.save(baseline)

        js_arr = js_simplex(args.width, args.height, seed, time=args.time)
        diff = compare(py_arr, js_arr)
        print(f'seed {seed}: max diff {diff:.2e}')
        if diff > args.threshold:
            print(f'\u274c divergence detected for seed {seed}')
            failed = True

    if failed:
        return 1
    print('All seeds matched baseline')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
