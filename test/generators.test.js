import assert from 'assert';
import { basic, multires } from '../src/generators.js';

function arraysClose(a, b, eps = 1e-6) {
  assert.strictEqual(a.length, b.length);
  for (let i = 0; i < a.length; i++) {
    assert.ok(Math.abs(a[i] - b[i]) < eps, `index ${i}`);
  }
}

const shape = [4, 4, 3];
const g1 = basic(2, shape, { seed: 1, hueRotation: 0 });
const g2 = basic(2, shape, { seed: 1, hueRotation: 0 });
arraysClose(g1.read(), g2.read());

const m1 = multires(2, shape, { octaves: 2, seed: 1, hueRotation: 0 });
const m2 = multires(2, shape, { octaves: 2, seed: 1, hueRotation: 0 });
arraysClose(m1.read(), m2.read());
assert.deepStrictEqual(m1.shape, shape);
