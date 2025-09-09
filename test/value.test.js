import assert from 'assert';
import { values, downsample, upsample, blend, sobel, valueMap, hsvToRgb, rgbToHsv, warp, fft, ifft, refract, convolution, ridge, rotate, zoom, fxaa, gaussianBlur } from '../src/value.js';
import { rgbToOklab, oklabToRgb } from '../src/oklab.js';
import { ValueDistribution } from '../src/constants.js';
import { Tensor } from '../src/tensor.js';
import { Context } from '../src/context.js';

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

// frequency as [x, y] should produce valid numbers
const vxy = values([4, 2], [4, 4, 1], { seed: 7 });
assert.ok(vxy.read().every((v) => !Number.isNaN(v)));

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

// blend with channel broadcasting
const a3 = values(4, [2, 2, 3], { seed: 1 });
const b1 = values(4, [2, 2, 1], { seed: 2 });
const mask1 = values(4, [2, 2, 1], { seed: 3 });
const blendedBroadcast = blend(a3, b1, mask1);
const da3 = a3.read();
const db1 = b1.read();
const dm1 = mask1.read();
const expectedBroadcast = new Float32Array(2 * 2 * 3);
for (let i = 0; i < 4; i++) {
  for (let k = 0; k < 3; k++) {
    const baseA = i * 3 + k;
    const bVal = db1[i];
    const tVal = dm1[i];
    expectedBroadcast[baseA] = da3[baseA] * (1 - tVal) + bVal * tVal;
  }
}
arraysClose(blendedBroadcast.read(), expectedBroadcast);

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

// oklab conversions with reference values for primary colours
const samples = [
  { rgb: [1, 0, 0], lab: [0.62791519, 0.22490323, 0.12580287] },
  { rgb: [0, 1, 0], lab: [0.86643975, -0.23392021, 0.17942364] },
  { rgb: [0, 0, 1], lab: [0.45203033, -0.03237854, -0.31161998] },
];
for (const { rgb: vals, lab } of samples) {
  const rgbT = Tensor.fromArray(null, new Float32Array(vals), [1, 1, 3]);
  const labT = rgbToOklab(rgbT);
  arraysClose(labT.read(), new Float32Array(lab));
  const rgbRoundTrip = oklabToRgb(labT);
  arraysClose(rgbT.read(), rgbRoundTrip.read(), 1e-5);
}

// warp with zero flow
const flow = Tensor.fromArray(null, new Float32Array(2*2*2).fill(0), [2,2,2]);
const src = values(1, [2,2,1], { distrib: ValueDistribution.row_index });
const warped = warp(src, flow, 1);
arraysClose(src.read(), warped.read());

// ridge
const ridgeInput = Tensor.fromArray(null, new Float32Array([0, 0.5, 1]), [3, 1, 1]);
arraysClose(ridge(ridgeInput).read(), new Float32Array([0, 1, 0]));

// convolution
const convInput = Tensor.fromArray(null, new Float32Array([
  0, 1, 2,
  3, 4, 5,
  6, 7, 8,
]), [3, 3, 1]);
const convKernel = [[0, 1, 0], [1, 0, 1], [0, 1, 0]];
const convResult = convolution(convInput, convKernel);
arraysClose(convResult.read(), new Float32Array([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]));

// refract
const refrInput = Tensor.fromArray(null, new Float32Array([
  0, 1, 2,
  3, 4, 5,
  6, 7, 8,
]), [3, 3, 1]);
const refX = Tensor.fromArray(null, new Float32Array(9).fill(1), [3, 3, 1]);
const refY = Tensor.fromArray(null, new Float32Array(9).fill(0), [3, 3, 1]);
const refracted = refract(refrInput, refX, refY, 1);
arraysClose(refracted.read(), new Float32Array([1, 2, 2, 1, 2, 2, 4, 5, 5]));

// fft / ifft
const fftInput = Tensor.fromArray(null, new Float32Array([1, 2, 3, 4]), [2, 2, 1]);
const fftOut = fft(fftInput);
arraysClose(fftOut.read(), new Float32Array([10, 0, -2, 0, -4, 0, 0, 0]));
const ifftOut = ifft(fftOut);
arraysClose(ifftOut.read(), fftInput.read());

// rotate
const rotInput = Tensor.fromArray(null, new Float32Array([
  0, 1, 2,
  3, 4, 5,
  6, 7, 8,
]), [3, 3, 1]);
const rotated = rotate(rotInput, Math.PI / 2);
arraysClose(rotated.read(), new Float32Array([6, 3, 0, 7, 4, 1, 8, 5, 2]));

// zoom
const zoomInput = Tensor.fromArray(null, new Float32Array(Array.from({ length: 25 }, (_, i) => i)), [5, 5, 1]);
const zoomed = zoom(zoomInput, 2);
arraysClose(zoomed.read(), new Float32Array([6, 7, 7, 8, 8, 11, 12, 12, 13, 13, 11, 12, 12, 13, 13, 16, 17, 17, 18, 18, 16, 17, 17, 18, 18]));

// fxaa
const fxaaInput = Tensor.fromArray(null, new Float32Array([
  0, 0, 0,
  0, 1, 0,
  0, 0, 0,
]), [3, 3, 1]);
const fxaaOut = fxaa(fxaaInput);
arraysClose(
  fxaaOut.read(),
  new Float32Array([
    0,
    0.19695032,
    0,
    0.19695032,
    0.40460968,
    0.19695032,
    0,
    0.19695032,
    0,
  ])
);

// gaussian blur
const gbData = new Float32Array(25);
gbData[12] = 1;
const gbInput = Tensor.fromArray(null, gbData, [5, 5, 1]);
const gb = gaussianBlur(gbInput, 1);
arraysClose(gb.read(), new Float32Array([
  0, 0, 0, 0, 0,
  0, 0.018315639, 0.13533528, 0.018315639, 0,
  0, 0.13533528, 1, 0.13533528, 0,
  0, 0.018315639, 0.13533528, 0.018315639, 0,
  0, 0, 0, 0, 0,
]));

// GPU vs CPU parity for additional distributions
let canvas = null;
if (typeof document !== 'undefined' && document.createElement) {
  canvas = document.createElement('canvas');
} else if (typeof OffscreenCanvas !== 'undefined') {
  canvas = new OffscreenCanvas(1, 1);
}
const gpuCtx = new Context(canvas);
const cpuCtx = new Context(null);
if (!gpuCtx.isCPU) {
  const distribs = [
    ValueDistribution.exp,
    ValueDistribution.scan_up,
    ValueDistribution.scan_down,
    ValueDistribution.scan_left,
    ValueDistribution.scan_right,
    ValueDistribution.center_circle,
  ];
  for (const d of distribs) {
    const gpu = values(4, [4, 4, 1], { seed: 1, ctx: gpuCtx, distrib: d });
    const cpu = values(4, [4, 4, 1], { seed: 1, ctx: cpuCtx, distrib: d });
    arraysClose(gpu.read(), cpu.read());
  }
} else {
  console.log('Skipping GPU vs CPU distribution comparisons');
}

console.log('Value tests passed');
