#!/usr/bin/env python3
import json, math, colorsys
from pathlib import Path
import numpy as np
import tensorflow as tf
import sys

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from noisemaker.effects import (
    posterize,
    adjust_hue,
    ridge,
    sine,
    blur,
    fibers,
    scratches,
    stray_hair,
)
from noisemaker.palettes import PALETTES
from noisemaker import value
from opensimplex import OpenSimplex

# JS RNG equivalent
def js_random(seed):
    seed = seed & 0xFFFFFFFF
    seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
    return seed / 0x100000000, seed

fixtures_dir = root / 'test' / 'fixtures'
fixtures_dir.mkdir(parents=True, exist_ok=True)

# Posterize fixture
poster_vals = [0.1, 0.5, 0.9, 0.3]
tex = tf.constant(poster_vals, shape=[2,2,1], dtype=tf.float32)
poster_out = posterize(tex, [2,2,1], levels=4).numpy().flatten().tolist()
with open(fixtures_dir / 'posterize.json', 'w') as f:
    json.dump(poster_out, f)

# Palette fixture
pal_vals = [0.0, 0.5, 1.0, 0.25]
pal = PALETTES['grayscale']
pal_out = []
for t in pal_vals:
    r = pal['offset'][0] + pal['amp'][0]*math.cos(math.tau*(pal['freq'][0]*t*0.875 + 0.0625 + pal['phase'][0]))
    g = pal['offset'][1] + pal['amp'][1]*math.cos(math.tau*(pal['freq'][1]*t*0.875 + 0.0625 + pal['phase'][1]))
    b = pal['offset'][2] + pal['amp'][2]*math.cos(math.tau*(pal['freq'][2]*t*0.875 + 0.0625 + pal['phase'][2]))
    pal_out.extend([r, g, b])
with open(fixtures_dir / 'palette.json', 'w') as f:
    json.dump(pal_out, f)

# Vignette fixture
vig_vals = [0.1, 0.5, 0.3, 0.8]
arr = np.array(vig_vals).reshape(2,2,1)
minv, maxv = arr.min(), arr.max()
norm = (arr - minv) / (maxv - minv)
cx, cy = (2 - 1)/2, (2 - 1)/2
maxd = math.sqrt(cx*cx + cy*cy)
mask = np.zeros((2,2,1))
for y in range(2):
    for x in range(2):
        dx, dy = x - cx, y - cy
        dist = math.sqrt(dx*dx + dy*dy) / maxd
        mask[y,x,0] = dist ** 2
edges = np.ones((2,2,1)) * 0.25
vig = norm * (1 - mask) + edges * mask
final = norm * (1 - 0.5) + vig * 0.5
with open(fixtures_dir / 'vignette.json', 'w') as f:
    json.dump(final.flatten().tolist(), f)

# Saturation fixture
sat_vals = [0.2,0.4,0.6,0.8,0.1,0.3]
sat_out = []
for i in range(0, len(sat_vals), 3):
    r, g, b = sat_vals[i:i+3]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = min(1, max(0, s * 0.5))
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    sat_out.extend([r2, g2, b2])
with open(fixtures_dir / 'saturation.json', 'w') as f:
    json.dump(sat_out, f)

# Random hue fixture
rh_vals = [0.1,0.2,0.3,0.4,0.5,0.6]
seed = 5
rand, seed = js_random(seed)
shift = rand * 0.1 - 0.05
rh_out = []
for i in range(0, len(rh_vals), 3):
    r, g, b = rh_vals[i:i+3]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + shift) % 1
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    rh_out.extend([r2, g2, b2])
with open(fixtures_dir / 'randomHue.json', 'w') as f:
    json.dump(rh_out, f)

# Adjust hue fixture
ah_vals = [0.2, 0.4, 0.6, 0.8, 0.1, 0.3]
ah_tex = tf.constant(ah_vals, shape=[2,1,3], dtype=tf.float32)
ah_out = adjust_hue(ah_tex, [2,1,3], amount=0.25).numpy().flatten().tolist()
with open(fixtures_dir / 'adjustHue.json', 'w') as f:
    json.dump(ah_out, f)

# Ridge fixture
ridge_vals = [0.2, 0.8, 0.4, 0.6]
ridge_tex = tf.constant(ridge_vals, shape=[2,2,1], dtype=tf.float32)
ridge_out = ridge(ridge_tex, [2,2,1]).numpy().flatten().tolist()
with open(fixtures_dir / 'ridge.json', 'w') as f:
    json.dump(ridge_out, f)

# Sine fixture
sine_vals = [0.1,0.2,0.3,0.4,0.5,0.6]
sine_tex = tf.constant(sine_vals, shape=[2,1,3], dtype=tf.float32)
sine_out = sine(sine_tex, [2,1,3], amount=1.0, rgb=False).numpy().flatten().tolist()
with open(fixtures_dir / 'sine.json', 'w') as f:
    json.dump(sine_out, f)

# Blur fixture
blur_vals = [0.1,0.5,0.3,0.7]
blur_tex = tf.constant(blur_vals, shape=[2,2,1], dtype=tf.float32)
blur_out = blur(blur_tex, [2,2,1]).numpy().flatten().tolist()
with open(fixtures_dir / 'blur.json', 'w') as f:
    json.dump(blur_out, f)

# Simplex fixture
shape = (2,2)
time = 0.25
seed = 12345
speed = 1.0
angle = math.tau * time
z = math.cos(angle) * speed
simp = OpenSimplex(seed)
rows = [[(simp.noise3d(x, y, z) + 1) / 2 for x in range(shape[1])] for y in range(shape[0])]
rand = (OpenSimplex(seed).noise2d(math.cos(angle)*speed, math.sin(angle)*speed) + 1) / 2
with open(fixtures_dir / 'simplex.json', 'w') as f:
    json.dump({'tensor': rows, 'random': rand}, f)

# Fibers fixture
value.set_seed(1)
base = tf.zeros([4,4,1], dtype=tf.float32)
fib_out = fibers(base, [4,4,1]).numpy().flatten().tolist()
with open(fixtures_dir / 'fibers.json', 'w') as f:
    json.dump(fib_out, f)

# Scratches fixture
value.set_seed(1)
base = tf.zeros([4,4,1], dtype=tf.float32)
scr_out = scratches(base, [4,4,1]).numpy().flatten().tolist()
with open(fixtures_dir / 'scratches.json', 'w') as f:
    json.dump(scr_out, f)

# Stray hair fixture
value.set_seed(1)
base = tf.zeros([4,4,1], dtype=tf.float32)
sh_out = stray_hair(base, [4,4,1]).numpy().flatten().tolist()
with open(fixtures_dir / 'strayHair.json', 'w') as f:
    json.dump(sh_out, f)

print(f'Wrote fixtures to {fixtures_dir}')
