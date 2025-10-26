import assert from 'assert';
import { fromSeed, random as simplexRandom, simplex } from '../js/noisemaker/simplex.js';

const coords = [0, 1 / 7, 2 / 7, 3 / 7];
const seeds = [1, 2, 3];

function grid(os) {
  const arr = [];
  for (const x of coords) {
    for (const y of coords) {
      for (const z of coords) {
        arr.push((os.noise3D(x, y, z) + 1) / 2);
      }
    }
  }
  return arr;
}

function arraysClose(a, b, eps = 1e-6) {
  assert.strictEqual(a.length, b.length);
  for (let i = 0; i < a.length; i++) {
    assert.ok(Math.abs(a[i] - b[i]) < eps, `index ${i}`);
  }
}

for (const seed of seeds) {
  const { os } = fromSeed(seed);
  const g1 = grid(os);
  const { os: os2 } = fromSeed(seed);
  const g2 = grid(os2);
  arraysClose(g1, g2);

  const r = simplexRandom(0.25, seed, 1);
  assert.ok(r >= 0 && r <= 1);
  const dt = 1e-3;
  const d0 = (simplexRandom(dt, seed, 1) - simplexRandom(0.0, seed, 1)) / dt;
  const d1 = (simplexRandom(1 + dt, seed, 1) - simplexRandom(1.0, seed, 1)) / dt;
  assert.ok(Math.abs(d0 - d1) < 1e-6);

  const c = coords[1];
  const eps = 1e-5;
  const gx = (os.noise3D(c + eps, c, c) - os.noise3D(c - eps, c, c)) / (2 * eps);
  const gy = (os.noise3D(c, c + eps, c) - os.noise3D(c, c - eps, c)) / (2 * eps);
  const gz = (os.noise3D(c, c, c + eps) - os.noise3D(c, c, c - eps)) / (2 * eps);
  const gx2 = (os2.noise3D(c + eps, c, c) - os2.noise3D(c - eps, c, c)) / (2 * eps);
  const gy2 = (os2.noise3D(c, c + eps, c) - os2.noise3D(c, c - eps, c)) / (2 * eps);
  const gz2 = (os2.noise3D(c, c, c + eps) - os2.noise3D(c, c, c - eps)) / (2 * eps);
  assert.ok(Math.abs(gx - gx2) < 1e-6);
  assert.ok(Math.abs(gy - gy2) < 1e-6);
  assert.ok(Math.abs(gz - gz2) < 1e-6);

  const tile1 = simplex([4, 4, 1], { seed }).read();
  const tile2 = simplex([4, 4, 1], { seed }).read();
  arraysClose(tile1, tile2);
}

console.log('Simplex tests passed');
