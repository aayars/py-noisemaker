import assert from 'assert';
import { values, downsample, upsample, blend, sobel, valueMap, hsvToRgb, rgbToHsv, warp } from '../src/value.js';
import { ValueDistribution } from '../src/constants.js';
import { Tensor } from '../src/tensor.js';

function arraysClose(a, b, eps = 1e-6) {
  assert.strictEqual(a.length, b.length);
  for (let i = 0; i < a.length; i++) {
    assert.ok(Math.abs(a[i] - b[i]) < eps, `index ${i}`);
  }
}

// values determinism
const v1 = values(4, [4, 4, 1], { seed: 42, time: 0 });
const v2 = values(4, [4, 4, 1], { seed: 42, time: 0 });
arraysClose(v1.read(), v2.read());

// distribution check
const col = values(1, [1, 4, 1], { distrib: ValueDistribution.column_index });
arraysClose(col.read(), new Float32Array([0, 1/3, 2/3, 1]));

// resampling
const base = values(4, [4, 4, 1], { seed: 1 });
const down = downsample(base, 2);
assert.deepStrictEqual(down.shape, [2, 2, 1]);
const up = upsample(down, 2);
assert.deepStrictEqual(up.shape, [4, 4, 1]);

// blend
const a = values(4, [2, 2, 1], { seed: 1 });
const b = values(4, [2, 2, 1], { seed: 2 });
const blended = blend(a, b, 0.5);
const expBlend = a.read().map((v, i) => (v + b.read()[i]) / 2);
arraysClose(blended.read(), expBlend);

// sobel edge detection
const edgeData = new Float32Array([
  0, 0, 1, 1,
  0, 0, 1, 1,
  0, 0, 1, 1,
  0, 0, 1, 1,
]);
const edgeTensor = Tensor.fromArray(null, edgeData, [4, 4, 1]);
const sob = sobel(edgeTensor);
assert.ok(sob.read().some(v => v > 0));

// palette mapping
const gray = Tensor.fromArray(null, new Float32Array([0, 0.5, 1, 0.25]), [2, 2, 1]);
const palette = [[0,0,0],[1,0,0],[0,1,0],[0,0,1]];
const mapped = valueMap(gray, palette);
assert.deepStrictEqual(mapped.shape, [2, 2, 3]);
assert.deepStrictEqual(Array.from(mapped.read().slice(0,3)), palette[0]);

// color conversions
const rgb = Tensor.fromArray(null, new Float32Array([1, 0, 0]), [1, 1, 3]);
const hsv = rgbToHsv(rgb);
const rgbBack = hsvToRgb(hsv);
arraysClose(rgb.read(), rgbBack.read());

// warp with zero flow
const flow = Tensor.fromArray(null, new Float32Array(2*2*2).fill(0), [2,2,2]);
const src = values(1, [2,2,1], { distrib: ValueDistribution.row_index });
const warped = warp(src, flow, 1);
arraysClose(src.read(), warped.read());

console.log('Value tests passed');
