import assert from 'assert';
import { Tensor } from '../src/tensor.js';
import {
  posterize,
  palette,
  invert,
  aberration,
  reindex,
  vignette,
  dither,
  grain,
  saturation,
  randomHue,
  adjustHueEffect,
  ridgeEffect,
  sine,
  blur,
  erosionWorms,
  worms,
  wormhole,
} from '../src/effects.js';
import {
  adjustHue as adjustHueValue,
  rgbToHsv,
  hsvToRgb,
  values,
  blend,
} from '../src/value.js';
import { setSeed, random } from '../src/util.js';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';

function arraysClose(a, b, eps = 1e-6) {
  assert.strictEqual(a.length, b.length);
  for (let i = 0; i < a.length; i++) {
    assert.ok(Math.abs(a[i] - b[i]) < eps, `index ${i}`);
  }
}

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const loadFixture = (name) => JSON.parse(readFileSync(path.join(__dirname, 'fixtures', name), 'utf8'));

// posterize regression
const posterData = new Float32Array([0.1, 0.5, 0.9, 0.3]);
const posterTensor = Tensor.fromArray(null, posterData, [2, 2, 1]);
const jsPoster = posterize(posterTensor, [2, 2, 1], 0, 1, 4).read();
const posterExpected = loadFixture('posterize.json');
arraysClose(Array.from(jsPoster), posterExpected);

// palette regression
const palData = new Float32Array([0.0, 0.5, 1.0, 0.25]);
const palTensor = Tensor.fromArray(null, palData, [2, 2, 1]);
const jsPal = palette(palTensor, [2, 2, 1], 0, 1, 'grayscale').read();
const palExpected = loadFixture('palette.json');
arraysClose(Array.from(jsPal), palExpected);

// invert
const invData = new Float32Array([0.2, 0.5, 0.8]);
const invTensor = Tensor.fromArray(null, invData, [1, 3, 1]);
const invResult = invert(invTensor, [1, 3, 1], 0, 1).read();
arraysClose(Array.from(invResult), [0.8, 0.5, 0.2]);

// aberration deterministic check
setSeed(123);
const abShape = [1, 4, 3];
const abData = new Float32Array([
  0.1, 0.2, 0.3,
  0.4, 0.5, 0.6,
  0.7, 0.8, 0.9,
  0.2, 0.4, 0.6,
]);
const abTensor = Tensor.fromArray(null, abData, abShape);
const disp = Math.round(abShape[1] * 0.25 * random());
const hueShift = random() * 0.1 - 0.05;
const shifted = adjustHueValue(abTensor, hueShift).read();
const manual = new Float32Array(abShape[0] * abShape[1] * 3);
for (let x = 0; x < abShape[1]; x++) {
  const base = x * 3;
  const rIdx = Math.min(abShape[1] - 1, x + disp) * 3;
  const bIdx = Math.max(0, x - disp) * 3;
  manual[base] = shifted[rIdx];
  manual[base + 1] = shifted[base + 1];
  manual[base + 2] = shifted[bIdx + 2];
}
const expected = adjustHueValue(Tensor.fromArray(null, manual, abShape), -hueShift).read();
setSeed(123);
const abResult = aberration(abTensor, abShape, 0, 1, 0.25).read();
arraysClose(Array.from(abResult), Array.from(expected));

// reindex deterministic
const reData = new Float32Array([0.1, 0.2, 0.3, 0.4]);
const reTensor = Tensor.fromArray(null, reData, [2, 2, 1]);
const manualRe = new Float32Array(4);
const mod = 2;
for (let y = 0; y < 2; y++) {
  for (let x = 0; x < 2; x++) {
    const idx = y * 2 + x;
    const r = reData[idx];
    const xo = Math.floor((r * 0.5 * mod + r) % 2);
    const yo = Math.floor((r * 0.5 * mod + r) % 2);
    manualRe[idx] = reData[yo * 2 + xo];
  }
}
const jsRe = reindex(reTensor, [2, 2, 1], 0, 1, 0.5).read();
arraysClose(Array.from(jsRe), Array.from(manualRe));

// vignette regression (numpy impl)
const vigData = new Float32Array([0.1, 0.5, 0.3, 0.8]);
const vigTensor = Tensor.fromArray(null, vigData, [2, 2, 1]);
const jsVig = vignette(vigTensor, [2, 2, 1], 0, 1, 0.25, 0.5).read();
const vigExpected = loadFixture('vignette.json');
arraysClose(Array.from(jsVig), vigExpected);

// dither deterministic
setSeed(1);
const ditData = new Float32Array([0.2, 0.4, 0.6, 0.8]);
const ditTensor = Tensor.fromArray(null, ditData, [2, 2, 1]);
setSeed(1);
const noise = values(Math.max(2,2), [2,2,1], { time:0, seed:0, speed:1000 });
const nData = noise.read();
const manualDit = new Float32Array(4);
for (let i=0;i<4;i++){let v=ditData[i]+(nData[i]-0.5)/2; v=Math.floor(Math.min(1,Math.max(0,v))*2)/2; manualDit[i]=v;}
setSeed(1);
const jsDit = dither(ditTensor, [2,2,1], 0,1,2).read();
arraysClose(Array.from(jsDit), Array.from(manualDit));

// grain deterministic
setSeed(2);
const grData = new Float32Array([0.3,0.6,0.9,0.0]);
const grTensor = Tensor.fromArray(null, grData, [2,2,1]);
setSeed(2);
const gn = values(Math.max(2,2), [2,2,1], { time:0, speed:200 });
const gnData = gn.read();
const blended = blend(grTensor, Tensor.fromArray(null, (()=>{const arr=new Float32Array(4); for(let i=0;i<4;i++) arr[i]=gnData[i]; return arr;})(), [2,2,1]), 0.25).read();
setSeed(2);
const jsGrain = grain(grTensor, [2,2,1], 0,1,0.25).read();
arraysClose(Array.from(jsGrain), Array.from(blended));

// saturation regression via python colorsys
const satData = new Float32Array([
  0.2,0.4,0.6,
  0.8,0.1,0.3,
]);
const satTensor = Tensor.fromArray(null, satData, [2,1,3]);
const jsSat = saturation(satTensor, [2,1,3],0,1,0.5).read();
const satExpected = loadFixture('saturation.json');
arraysClose(Array.from(jsSat), satExpected);

// randomHue deterministic
setSeed(5);
const rhData = new Float32Array([
  0.1,0.2,0.3,
  0.4,0.5,0.6,
]);
const rhTensor = Tensor.fromArray(null, rhData, [2,1,3]);
setSeed(5);
const jsHue = randomHue(rhTensor,[2,1,3],0,1,0.05).read();
const hueExpected = loadFixture('randomHue.json');
arraysClose(Array.from(jsHue), hueExpected);

// adjustHue regression
const ahData = new Float32Array([0.2,0.4,0.6,0.8,0.1,0.3]);
const ahTensor = Tensor.fromArray(null, ahData, [2,1,3]);
const ahOut = adjustHueEffect(ahTensor, [2,1,3], 0, 1, 0.25).read();
const ahExpected = loadFixture('adjustHue.json');
arraysClose(Array.from(ahOut), ahExpected);

// ridge regression
const ridgeData = new Float32Array([0.2,0.8,0.4,0.6]);
const ridgeTensor = Tensor.fromArray(null, ridgeData, [2,2,1]);
const ridgeOut = ridgeEffect(ridgeTensor, [2,2,1], 0, 1).read();
const ridgeExpected = loadFixture('ridge.json');
arraysClose(Array.from(ridgeOut), ridgeExpected);

// sine regression
const sineData = new Float32Array([
  0.1,0.2,0.3,
  0.4,0.5,0.6,
]);
const sineTensor = Tensor.fromArray(null, sineData, [2,1,3]);
const sineOut = sine(sineTensor, [2,1,3], 0, 1, 1.0, false).read();
const sineExpected = loadFixture('sine.json');
arraysClose(Array.from(sineOut), sineExpected);

// blur regression
const blurData = new Float32Array([0.1,0.5,0.3,0.7]);
const blurTensor = Tensor.fromArray(null, blurData, [2,2,1]);
const blurOut = blur(blurTensor, [2,2,1], 0, 1, 10.0).read();
const blurExpected = loadFixture('blur.json');
arraysClose(Array.from(blurOut), blurExpected);

// wormhole regression
const whTensor = Tensor.fromArray(null, new Float32Array([0.5]), [1,1,1]);
const whOut = wormhole(whTensor, [1,1,1], 0, 1, 1.0, 1.0, 1.0).read();
const whExpected = loadFixture('wormhole.json');
arraysClose(Array.from(whOut), whExpected);

// worms regression with zero density
const wormsTensor = Tensor.fromArray(null, new Float32Array([0.4]), [1,1,1]);
const wormsOut = worms(wormsTensor, [1,1,1], 0, 1, 1, 0.0, 4.0, 1.0, 0.05, 0.5).read();
const wormsExpected = loadFixture('worms.json');
arraysClose(Array.from(wormsOut), wormsExpected);

// erosionWorms regression with zero density
const ewData = new Float32Array(25); ewData[0] = 0.6;
const ewTensor = Tensor.fromArray(null, ewData, [5,5,1]);
const ewOut = erosionWorms(ewTensor, [5,5,1], 0, 1, 0.0, 5, 1.0, false, 0.25).read();
const ewExpected = loadFixture('erosionWorms.json');
arraysClose(Array.from(ewOut), ewExpected);

console.log('Effects tests passed');
