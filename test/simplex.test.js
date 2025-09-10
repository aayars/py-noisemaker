import assert from 'assert';
import { fromSeed, random as simplexRandom } from '../src/simplex.js';

const coords = [0, 1 / 7, 2 / 7, 3 / 7];
const EXPECTED = {
  1: {
    perm: [68, 151, 59, 159, 166, 58, 162, 75, 242, 1],
    perm_grad: [60, 21, 33, 45, 66, 30, 54, 9, 6, 3],
    random: 0.5625817649168068,
  },
  2: {
    perm: [176, 1, 234, 36, 27, 83, 21, 156, 138, 147],
    perm_grad: [24, 3, 54, 36, 9, 33, 63, 36, 54, 9],
    random: 0.4374182350831932,
  },
  3: {
    perm: [214, 180, 149, 105, 98, 243, 249, 251, 208, 11],
    perm_grad: [66, 36, 15, 27, 6, 9, 27, 33, 48, 33],
    random: 0.36106808044234445,
  },
};

for (const [seedStr, exp] of Object.entries(EXPECTED)) {
  const seed = Number(seedStr);
  const { os } = fromSeed(seed);
  assert.deepStrictEqual(Array.from(os.perm.slice(0, 10)), exp.perm);
  assert.deepStrictEqual(Array.from(os.permGradIndex3D.slice(0, 10)), exp.perm_grad);
  const grid = [];
  for (const x of coords) {
    for (const y of coords) {
      for (const z of coords) {
        const v = (os.noise3D(x, y, z) + 1) / 2;
        assert.ok(v >= 0 && v <= 1);
        grid.push(v);
      }
    }
  }
  const { os: os2 } = fromSeed(seed);
  const grid2 = [];
  for (const x of coords) {
    for (const y of coords) {
      for (const z of coords) {
        grid2.push((os2.noise3D(x, y, z) + 1) / 2);
      }
    }
  }
  assert.ok(Math.abs(grid.reduce((a, b) => a + b, 0) - grid2.reduce((a, b) => a + b, 0)) < 1e-9);
  const r = simplexRandom(0.25, seed, 1);
  assert.ok(Math.abs(r - exp.random) < 1e-6);
  const dt = 1e-3;
  const d0 = (simplexRandom(dt, seed, 1) - simplexRandom(0.0, seed, 1)) / dt;
  const d1 = (simplexRandom(1 + dt, seed, 1) - simplexRandom(1.0, seed, 1)) / dt;
  assert.ok(Math.abs(d0 - d1) < 1e-6);
}

console.log('Simplex tests passed');
