import assert from 'assert';
import { values, downsample, upsample, blend, sobel, valueMap, hsvToRgb, rgbToHsv, warp, fft, ifft, refract, convolution, ridge, rotate, zoom, fxaa, gaussianBlur, freqForShape, normalize } from '../js/noisemaker/value.js';
import { rgbToOklab, oklabToRgb } from '../js/noisemaker/oklab.js';
import { ValueDistribution, ValueMask, InterpolationType } from '../js/noisemaker/constants.js';
import { Tensor } from '../js/noisemaker/tensor.js';
import { Context } from '../js/noisemaker/context.js';

const DEBUG = false; // Set true to diagnose shader issues.

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
const row = values(1, [1, 4, 1], { distrib: ValueDistribution.row_index });
arraysClose(row.read(), new Float32Array([0, 1 / 3, 2 / 3, 1]));

// ensure axes are not flipped
const row2d = values(1, [2, 3, 1], { distrib: ValueDistribution.row_index });
arraysClose(row2d.read(), new Float32Array(6));
const col2d = values(1, [3, 2, 1], { distrib: ValueDistribution.column_index });
arraysClose(col2d.read(), new Float32Array(6));

// constant distributions
arraysClose(
  values(1, [2, 2, 1], { distrib: ValueDistribution.ones }).read(),
  new Float32Array(4).fill(1)
);
arraysClose(
  values(1, [2, 2, 1], { distrib: ValueDistribution.mids }).read(),
  new Float32Array(4).fill(0.5)
);
arraysClose(
  values(1, [2, 2, 1], { distrib: ValueDistribution.zeros }).read(),
  new Float32Array(4)
);

// multi-channel noise should produce independent values per channel
const colorVals = values(2, [2, 2, 3], { seed: 1 });
const colorArr = colorVals.read();
let colorDiff = false;
for (let i = 0; i < colorArr.length; i += 3) {
  if (colorArr[i] !== colorArr[i + 1] || colorArr[i] !== colorArr[i + 2]) {
    colorDiff = true;
    break;
  }
}
assert.ok(colorDiff, 'color channels should differ');

// center distributions should peak at the middle and fade towards corners
const centerDistribs = Object.values(ValueDistribution).filter(
  (d) =>
    d >= ValueDistribution.center_circle &&
    d <= ValueDistribution.center_dodecagon
);
for (const d of centerDistribs) {
  const t = values(1, [3, 3, 1], { distrib: d });
  const arr = t.read();
  assert.ok(arr[4] > arr[0], `center greater than corner for distrib ${d}`);
}

// ensure all distributions produce values within [0,1]
for (const d of Object.values(ValueDistribution)) {
  const arr = values(2, [3, 3, 1], { distrib: d }).read();
  assert.ok(
    Array.from(arr).every((v) => v >= 0 && v <= 1),
    `range check for distrib ${d}`
  );
}

// freqForShape helper
assert.deepStrictEqual(freqForShape(4, [64, 64]), [4, 4]);
assert.deepStrictEqual(freqForShape(4, [32, 64]), [4, 8]);
assert.deepStrictEqual(freqForShape(4, [64, 32]), [8, 4]);

// resampling
const base = values(4, [4, 4, 1], { seed: 1 });
const down = downsample(base, 2);
assert.deepStrictEqual(down.shape, [2, 2, 1]);
const up = upsample(down, 2);
assert.deepStrictEqual(up.shape, [4, 4, 1]);

// mask interpolation
const maskConst = values(4, [8, 8, 1], {
  distrib: ValueDistribution.ones,
  mask: ValueMask.square,
  splineOrder: InterpolationType.constant,
});
assert.ok(Array.from(maskConst.read()).every((v) => v === 0 || v === 1));
const maskLinear = values(4, [8, 8, 1], {
  distrib: ValueDistribution.ones,
  mask: ValueMask.square,
  splineOrder: InterpolationType.linear,
});
assert.ok(Array.from(maskLinear.read()).some((v) => v > 0 && v < 1));

// blend
const a = values(4, [2, 2, 1], { seed: 1 });
const b = values(4, [2, 2, 1], { seed: 2 });
const blended = await blend(a, b, 0.5);
const expBlend = a.read().map((v, i) => (v + b.read()[i]) / 2);
arraysClose(blended.read(), expBlend);

// blend with channel broadcasting
const a3 = values(4, [2, 2, 3], { seed: 1 });
const b1 = values(4, [2, 2, 1], { seed: 2 });
const mask1 = values(4, [2, 2, 1], { seed: 3 });
const blendedBroadcast = await blend(a3, b1, mask1);
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

// blend with spatial broadcasting
const baseA = Tensor.fromArray(null, new Float32Array([0, 1, 2, 3]), [2, 2, 1]);
const rowB = Tensor.fromArray(null, new Float32Array([4, 5]), [1, 2, 1]);
const blendedRow = await blend(baseA, rowB, 0.5);
arraysClose(blendedRow.read(), new Float32Array([2, 3, 3, 4]));
const colB = Tensor.fromArray(null, new Float32Array([6, 7]), [2, 1, 1]);
const blendedCol = await blend(baseA, colB, 0.5);
arraysClose(blendedCol.read(), new Float32Array([3, 3.5, 4.5, 5]));

// GPU blend should handle channel mismatch by falling back to CPU
let gpuCanvas = null;
if (typeof document !== 'undefined' && document.createElement) {
  gpuCanvas = document.createElement('canvas');
} else if (typeof OffscreenCanvas !== 'undefined') {
  gpuCanvas = new OffscreenCanvas(1, 1);
}
const gpuBlendCtx = new Context(gpuCanvas, DEBUG);
if (!gpuBlendCtx.isCPU) {
  const a3g = values(4, [2, 2, 3], { seed: 1, ctx: gpuBlendCtx });
  const b1g = values(4, [2, 2, 1], { seed: 2, ctx: gpuBlendCtx });
  const blendGPU = await blend(a3g, b1g, 0.5);
  const da = a3g.read();
  const db = b1g.read();
  const expected = new Float32Array(2 * 2 * 3);
  for (let i = 0; i < 4; i++) {
    const bVal = db[i];
    for (let k = 0; k < 3; k++) {
      const idx = i * 3 + k;
      expected[idx] = da[idx] * 0.5 + bVal * 0.5;
    }
  }
  arraysClose(blendGPU.read(), expected);

  const maskg = values(4, [2, 2, 1], { seed: 3, ctx: gpuBlendCtx });
  const blendMaskGPU = await blend(a3g, b1g, maskg);
  const dm = maskg.read();
  const expectedMask = new Float32Array(2 * 2 * 3);
  for (let i = 0; i < 4; i++) {
    const bVal = db[i];
    const tVal = dm[i];
    for (let k = 0; k < 3; k++) {
      const idx = i * 3 + k;
      expectedMask[idx] = da[idx] * (1 - tVal) + bVal * tVal;
    }
  }
  arraysClose(await blendMaskGPU.read(), expectedMask);
} else {
  console.log('Skipping GPU blend mismatch test');
}

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

// normalize should use global min/max across channels
const normMulti = Tensor.fromArray(null, new Float32Array([0, 1, 2, 3, 4, 5]), [1, 2, 3]);
arraysClose(
  normalize(normMulti).read(),
  new Float32Array([0, 0.2, 0.4, 0.6, 0.8, 1])
);

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

// rgbToOklab and oklabToRgb should accept promised tensors
const promisedRgb = Tensor.fromArray(null, new Float32Array([1, 0, 0]), [1, 1, 3]);
const promisedLab = await rgbToOklab(Promise.resolve(promisedRgb));
arraysClose(promisedLab.read(), rgbToOklab(promisedRgb).read());
const promisedRoundTrip = await oklabToRgb(Promise.resolve(promisedLab));
arraysClose(promisedRgb.read(), promisedRoundTrip.read(), 1e-5);

// warp with zero flow
const flow = Tensor.fromArray(null, new Float32Array(2*2*2).fill(0), [2,2,2]);
const src = values(1, [2,2,1], { distrib: ValueDistribution.row_index });
const warped = warp(src, flow, 1);
arraysClose(src.read(), warped.read());

// warp with fractional flow (bilinear interpolation)
const flowFrac = Tensor.fromArray(
  null,
  new Float32Array([
    0.25, 0,
    0.25, 0,
    0.25, 0,
    0.25, 0,
  ]),
  [2, 2, 2]
);
const srcFrac = Tensor.fromArray(null, new Float32Array([0, 1, 2, 3]), [2, 2, 1]);
const warpedFrac = warp(srcFrac, flowFrac, 1, InterpolationType.linear);
// Warp uses wrapping semantics, so values beyond the right edge wrap to the left.
arraysClose(warpedFrac.read(), new Float32Array([0.5, 0.5, 2.5, 2.5]));

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
arraysClose(
  convResult.read(),
  new Float32Array([1, 0.75, 0.875, 0.25, 0, 0.125, 0.625, 0.375, 0.5]),
);

const convAsync = await convolution(Promise.resolve(convInput), convKernel);
arraysClose(
  convAsync.read(),
  new Float32Array([1, 0.75, 0.875, 0.25, 0, 0.125, 0.625, 0.375, 0.5]),
);

// refract
const refrInput = Tensor.fromArray(null, new Float32Array([
  0, 1, 2,
  3, 4, 5,
  6, 7, 8,
]), [3, 3, 1]);
const refX = Tensor.fromArray(null, new Float32Array(9).fill(1), [3, 3, 1]);
const refY = Tensor.fromArray(null, new Float32Array(9).fill(0), [3, 3, 1]);
const refracted = refract(refrInput, refX, refY, 1 / 3, InterpolationType.linear);
// Refract offsets wrap around the texture boundaries.
arraysClose(refracted.read(), new Float32Array([7, 8, 6, 1, 2, 0, 4, 5, 3]));

const refXNeg = Tensor.fromArray(null, new Float32Array(9).fill(0.25), [3, 3, 1]);
const refYZero = Tensor.fromArray(null, new Float32Array(9).fill(0), [3, 3, 1]);
const refractedNeg = refract(
  refrInput,
  refXNeg,
  refYZero,
  1 / 3,
  InterpolationType.linear,
);
arraysClose(
  refractedNeg.read(),
  new Float32Array([
    7, 6.5, 7.5,
    1, 0.5, 1.5,
    4, 3.5, 4.5,
  ]),
);

const wrapCoord = (coord, limit) => {
  const mod = coord % limit;
  return mod < 0 ? mod + limit : mod;
};

const manualRefract = (src, w, h, c, dxs, dys) => {
  const out = new Float32Array(h * w * c);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const sampleX = wrapCoord(x + dxs[idx], w);
      const sampleY = wrapCoord(y + dys[idx], h);
      const x0 = Math.floor(sampleX);
      const y0 = Math.floor(sampleY);
      const x1 = (x0 + 1) % w;
      const y1 = (y0 + 1) % h;
      const fx = sampleX - x0;
      const fy = sampleY - y0;
      for (let k = 0; k < c; k++) {
        const base = (y0 * w + x0) * c + k;
        const s00 = src[base];
        const s10 = src[(y0 * w + x1) * c + k];
        const s01 = src[(y1 * w + x0) * c + k];
        const s11 = src[(y1 * w + x1) * c + k];
        const x_y0 = s00 * (1 - fx) + s10 * fx;
        const x_y1 = s01 * (1 - fx) + s11 * fx;
        out[(y * w + x) * c + k] = x_y0 * (1 - fy) + x_y1 * fy;
      }
    }
  }
  return out;
};

const complexData = new Float32Array([
  0, 1, 2, 3,
  4, 5, 6, 7,
  8, 9, 10, 11,
]);
const complexTensor = Tensor.fromArray(null, complexData, [3, 4, 1]);
const complexDx = [
  -1.5, -0.5, 0.5, 1.5,
  -0.75, 0.75, -1.25, 1.25,
  0.0, 0.0, 0.0, 0.0,
];
const complexDy = [
  0.5, 0.5, 0.5, 0.5,
  -1.25, -1.25, -0.75, -0.75,
  0.25, -0.25, 1.0, -1.0,
];
const complexDisp = 0.5;
const complexRefX = new Float32Array(complexDx.length);
const complexRefY = new Float32Array(complexDy.length);
const wComplex = 4;
const hComplex = 3;
for (let i = 0; i < complexDx.length; i++) {
  complexRefX[i] = (complexDx[i] / (complexDisp * wComplex) + 1) * 0.5;
  complexRefY[i] = (complexDy[i] / (complexDisp * hComplex) + 1) * 0.5;
}
const complexRefXTensor = Tensor.fromArray(null, complexRefX, [hComplex, wComplex, 1]);
const complexRefYTensor = Tensor.fromArray(null, complexRefY, [hComplex, wComplex, 1]);
const complexRefract = refract(
  complexTensor,
  complexRefXTensor,
  complexRefYTensor,
  complexDisp,
  InterpolationType.linear,
);
const expectedComplex = manualRefract(
  complexData,
  wComplex,
  hComplex,
  1,
  complexDx,
  complexDy,
);
arraysClose(complexRefract.read(), expectedComplex, 1e-5);

const multiWrapData = new Float32Array([
  0, 1, 2, 3,
  4, 5, 6, 7,
]);
const multiWrapTensor = Tensor.fromArray(null, multiWrapData, [2, 4, 1]);
const multiWrapRefX = Tensor.fromArray(null, new Float32Array(8).fill(1), [2, 4, 1]);
const multiWrapRefY = Tensor.fromArray(null, new Float32Array(8).fill(0.5), [2, 4, 1]);
const multiWrapDisp = 2.125;
const multiWrapResult = refract(
  multiWrapTensor,
  multiWrapRefX,
  multiWrapRefY,
  multiWrapDisp,
  InterpolationType.linear,
);
const multiWrapDx = Array(8).fill(multiWrapDisp * 4);
const multiWrapDy = Array(8).fill(0);
const multiWrapExpected = manualRefract(
  multiWrapData,
  4,
  2,
  1,
  multiWrapDx,
  multiWrapDy,
);
arraysClose(multiWrapResult.read(), multiWrapExpected, 1e-5);

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
const fxaaOut = await (await fxaa(fxaaInput)).read();
arraysClose(
  fxaaOut,
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
arraysClose(
  gb.read(),
  new Float32Array([
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.018315639, 0.13533528, 0.018315639,
    0, 0, 0.13533528, 1, 0.13533528,
    0, 0, 0.018315639, 0.13533528, 0.018315639,
  ]),
);

// GPU vs CPU parity for additional distributions
let canvas = null;
if (typeof document !== 'undefined' && document.createElement) {
  canvas = document.createElement('canvas');
} else if (typeof OffscreenCanvas !== 'undefined') {
  canvas = new OffscreenCanvas(1, 1);
}
const gpuCtx = new Context(canvas, DEBUG);
const cpuCtx = new Context(null, DEBUG);
if (!gpuCtx.isCPU) {
  const distribs = Object.values(ValueDistribution);
  for (const d of distribs) {
    const gpu = values(4, [4, 4, 1], { seed: 1, ctx: gpuCtx, distrib: d });
    const cpu = values(4, [4, 4, 1], { seed: 1, ctx: cpuCtx, distrib: d });
    arraysClose(gpu.read(), cpu.read());
  }
} else {
  console.log('Skipping GPU vs CPU distribution comparisons');
}

console.log('Value tests passed');
