import assert from 'assert';
import { basic, multires } from '../src/generators.js';
import { ColorSpace } from '../src/constants.js';
import { register, EFFECTS, EFFECT_METADATA } from '../src/effectsRegistry.js';
import { Effect } from '../src/composer.js';

function arraysClose(a, b, eps = 1e-6) {
  assert.strictEqual(a.length, b.length);
  for (let i = 0; i < a.length; i++) {
    assert.ok(Math.abs(a[i] - b[i]) < eps, `index ${i}`);
  }
}

const shape = [4, 4, 3];
const g1 = await basic(2, shape, { seed: 1, hueRotation: 0 });
const g2 = await basic(2, shape, { seed: 1, hueRotation: 0 });
arraysClose(await g1.read(), await g2.read());

const m1 = await multires(2, shape, { octaves: 2, seed: 1, hueRotation: 0 });
const m2 = await multires(2, shape, { octaves: 2, seed: 1, hueRotation: 0 });
arraysClose(await m1.read(), await m2.read());
assert.deepStrictEqual(m1.shape, shape);

let captured = null;
function captureEffect(tensor, shape, time, speed, displacement = 0) {
  captured = displacement;
  return tensor;
}
register('test_octave_displacement', captureEffect, { displacement: 1 });
const eff = Effect('test_octave_displacement', { displacement: 1 });
await basic(1, [2, 2, 1], { octaveEffects: [eff], octave: 3 });
assert.strictEqual(captured, 1 / 8);
delete EFFECTS['test_octave_displacement'];
delete EFFECT_METADATA['test_octave_displacement'];

const base = await basic(2, [4, 4, 1], {
  seed: 2,
  color_space: ColorSpace.grayscale,
});
const withSin = await basic(2, [4, 4, 1], {
  seed: 2,
  color_space: ColorSpace.grayscale,
  sin: 1.2,
});
const baseData = await base.read();
const sinData = await withSin.read();
const expected = new Float32Array(baseData.length);
for (let i = 0; i < baseData.length; i++) {
  expected[i] = Math.sin(1.2 * baseData[i]);
}
arraysClose(sinData, expected);
