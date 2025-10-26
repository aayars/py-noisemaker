import assert from 'assert';
import { basic, multires } from '../js/noisemaker/generators.js';
import { ColorSpace, ValueDistribution, ValueMask } from '../js/noisemaker/constants.js';
import { register, EFFECTS, EFFECT_METADATA } from '../js/noisemaker/effectsRegistry.js';
import { Effect } from '../js/noisemaker/composer.js';

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

const dynamicFreq = await multires(
  [
    (...args) => {
      if (!args.length) {
        throw new Error('settings missing');
      }
      return args[0].freqY;
    },
    (settings) => settings.freqX,
  ],
  shape,
  { freqY: 3, freqX: 5, octaves: 1 },
);
assert.deepStrictEqual(dynamicFreq.shape, shape);

await assert.rejects(
  () =>
    multires(
      [
        () => {
          throw new Error('boom');
        },
        2,
      ],
      shape,
      { octaves: 1 },
    ),
  /Unable to resolve dynamic frequency for axis Y/,
);

const masked = await multires([160, 160], [32, 32, 3], {
  octaves: 1,
  mask: ValueMask.arecibo,
  maskRepeat: 2,
  distrib: ValueDistribution.simplex,
});
const maskedData = Array.from(await masked.read());
assert.ok(maskedData.every(Number.isFinite), 'expected finite masked output');

