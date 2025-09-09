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
  bloom,
  lightLeak,
  vaseline,
  erosionWorms,
  worms,
  wormhole,
  ripple,
  colorMap,
  derivative,
  sobelOperator,
  outline,
  vortex,
  normalMap,
  densityMap,
  voronoi,
  singularity,
  jpegDecimate,
  convFeedback,
  wobble,
  reverb,
  dla,
  rotate,
  pixelSort,
  glyphMap,
  sketch,
  lowpoly,
  kaleido,
  texture,
  simpleFrame,
  frame,
  glitch,
  lensWarp,
  lensDistortion,
  degauss,
  vhs,
  scanlineError,
  crt,
  snow,
  spatter,
  nebula,
  clouds,
  fibers,
  scratches,
  strayHair,
  grime,
  watermark,
  onScreenDisplay,
  spookyTicker,
  falseColor,
  tint,
  valueRefract,
} from '../src/effects.js';
import {
  adjustHue as adjustHueValue,
  rgbToHsv,
  hsvToRgb,
  values,
  blend,
  sobel,
  normalize,
  refract,
  distance,
} from '../src/value.js';
import { setSeed, random } from '../src/util.js';
import { random as simplexRandom } from '../src/simplex.js';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';
import { VoronoiDiagramType, DistanceMetric } from '../src/constants.js';

function arraysClose(a, b, eps = 1e-6) {
  assert.strictEqual(a.length, b.length);
  for (let i = 0; i < a.length; i++) {
    assert.ok(Math.abs(a[i] - b[i]) < eps, `index ${i}`);
  }
}

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const loadFixture = (name) => JSON.parse(readFileSync(path.join(__dirname, 'fixtures', name), 'utf8'));

// gradient effects edge accuracy
const edgeData = new Float32Array([
  0, 0, 1, 1,
  0, 0, 1, 1,
  0, 0, 1, 1,
  0, 0, 1, 1,
]);
const edgeTensor = Tensor.fromArray(null, edgeData, [4, 4, 1]);

// derivative
const manualDeriv = (() => {
  const gxKernel = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const gyKernel = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  const mag = new Float32Array(16);
  function get(x, y) {
    x = Math.max(0, Math.min(3, x));
    y = Math.max(0, Math.min(3, y));
    return edgeData[y * 4 + x];
  }
  for (let y = 0; y < 4; y++) {
    for (let x = 0; x < 4; x++) {
      let gx = 0, gy = 0, idx = 0;
      for (let yy = -1; yy <= 1; yy++) {
        for (let xx = -1; xx <= 1; xx++) {
          const v = get(x + xx, y + yy);
          gx += gxKernel[idx] * v;
          gy += gyKernel[idx] * v;
          idx++;
        }
      }
      mag[y * 4 + x] = Math.sqrt(gx * gx + gy * gy);
    }
  }
  return Array.from(normalize(Tensor.fromArray(null, mag, [4, 4, 1])).read());
})();
const derivRes = derivative(edgeTensor, [4, 4, 1], 0, 1).read();
arraysClose(Array.from(derivRes), manualDeriv);

// sobel operator
const blurred = blur(edgeTensor, [4, 4, 1], 0, 1);
let sob = sobel(blurred);
sob = normalize(sob);
const sobData = sob.read();
for (let i = 0; i < sobData.length; i++) sobData[i] = Math.abs(sobData[i] * 2 - 1);
const sobRes = sobelOperator(edgeTensor, [4, 4, 1], 0, 1).read();
arraysClose(Array.from(sobRes), Array.from(sobData));

// outline
const outlineRes = outline(edgeTensor, [4, 4, 1], 0, 1).read();
const manualOutline = new Float32Array(16);
for (let i = 0; i < 16; i++) manualOutline[i] = sobData[i] * edgeData[i];
arraysClose(Array.from(outlineRes), Array.from(manualOutline));
const outlineInv = outline(edgeTensor, [4, 4, 1], 0, 1, undefined, true).read();
const manualOutlineInv = new Float32Array(16);
for (let i = 0; i < 16; i++) manualOutlineInv[i] = (1 - sobData[i]) * edgeData[i];
arraysClose(Array.from(outlineInv), Array.from(manualOutlineInv));

// normal map
const nmRes = normalMap(edgeTensor, [4, 4, 1], 0, 1).read();
const nmExpect = (() => {
  const gxKernel = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const gyKernel = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  const gx = new Float32Array(16);
  const gy = new Float32Array(16);
  function get(x, y) {
    x = Math.max(0, Math.min(3, x));
    y = Math.max(0, Math.min(3, y));
    return edgeData[y * 4 + x];
  }
  for (let y = 0; y < 4; y++) {
    for (let x = 0; x < 4; x++) {
      let sx = 0, sy = 0, idx = 0;
      for (let yy = -1; yy <= 1; yy++) {
        for (let xx = -1; xx <= 1; xx++) {
          const v = get(x + xx, y + yy);
          sx += gxKernel[idx] * v;
          sy += gyKernel[idx] * v;
          idx++;
        }
      }
      const i = y * 4 + x;
      gx[i] = 1 - sx;
      gy[i] = sy;
    }
  }
  const xNorm = normalize(Tensor.fromArray(null, gx, [4, 4, 1])).read();
  const yNorm = normalize(Tensor.fromArray(null, gy, [4, 4, 1])).read();
  const mag = new Float32Array(16);
  for (let i = 0; i < 16; i++) mag[i] = Math.sqrt(xNorm[i] * xNorm[i] + yNorm[i] * yNorm[i]);
  const zNorm = normalize(Tensor.fromArray(null, mag, [4, 4, 1])).read();
  const out = new Float32Array(16 * 3);
  for (let i = 0; i < 16; i++) {
    const z = 1 - Math.abs(zNorm[i] * 2 - 1) * 0.5 + 0.5;
    out[i * 3] = xNorm[i];
    out[i * 3 + 1] = yNorm[i];
    out[i * 3 + 2] = z;
  }
  return Array.from(out);
})();
arraysClose(Array.from(nmRes), nmExpect);

// singularity
const sgTensor = Tensor.fromArray(null, new Float32Array(16), [4, 4, 1]);
const sgRes = singularity(sgTensor, [4, 4, 1], 0, 1).read();
arraysClose(Array.from(sgRes), [
  1,
  0.889139711856842,
  0.8408964276313782,
  0.889139711856842,
  0.889139711856842,
  0.7071067690849304,
  0.5946035385131836,
  0.7071067690849304,
  0.8408964276313782,
  0.5946035385131836,
  0,
  0.5946035385131836,
  0.889139711856842,
  0.7071067690849304,
  0.5946035385131836,
  0.7071067690849304,
]);

// voronoi color regions
const xPts = [1, 3];
const yPts = [1, 3];
const vorRes = voronoi(
  edgeTensor,
  [4, 4, 1],
  0,
  1,
  VoronoiDiagramType.color_regions,
  0,
  DistanceMetric.euclidean,
  1,
  0,
  0,
  0,
  0,
  false,
  [xPts, yPts, 2],
).read();
assert.strictEqual(vorRes[0], 0);
assert.strictEqual(vorRes[3], 1);
assert.strictEqual(vorRes[12], 1);
assert.strictEqual(vorRes[15], 1);

// densityMap regression
const dmData = new Float32Array([0.1, 0.4, 0.4, 0.9]);
const dmTensor = Tensor.fromArray(null, dmData, [2, 2, 1]);
const dmOut = densityMap(dmTensor, [2, 2, 1], 0, 1).read();
const dmExpected = loadFixture('densityMap.json');
arraysClose(Array.from(dmOut), dmExpected);

// jpegDecimate regression
setSeed(7);
const jdData = new Float32Array([
  0.1, 0.2, 0.3,
  0.4, 0.5, 0.6,
  0.7, 0.8, 0.9,
  0.2, 0.4, 0.6,
]);
const jdTensor = Tensor.fromArray(null, jdData, [2, 2, 3]);
setSeed(7);
const jdOut = jpegDecimate(jdTensor, [2, 2, 3], 0, 1, 1).read();
const jdExpected = loadFixture('jpegDecimate.json');
arraysClose(Array.from(jdOut), jdExpected);

// convFeedback regression
const cfData = new Float32Array([0.1, 0.2, 0.3, 0.4]);
const cfTensor = Tensor.fromArray(null, cfData, [2, 2, 1]);
const cfOut = convFeedback(cfTensor, [2, 2, 1], 0, 1, 2, 0.5).read();
const cfExpected = loadFixture('convFeedback.json');
arraysClose(Array.from(cfOut), cfExpected);

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

// ripple deterministic
setSeed(10);
const ripData = new Float32Array([0.1, 0.2, 0.3, 0.4]);
const ripTensor = Tensor.fromArray(null, ripData, [2, 2, 1]);
setSeed(10);
const jsRipple = ripple(ripTensor, [2, 2, 1], 0, 1, 2, 0.5, 1).read();
const rippleExpected = loadFixture('ripple.json');
arraysClose(Array.from(jsRipple), rippleExpected);

// colorMap regression
const cmData = new Float32Array([0.1, 0.2, 0.3, 0.4]);
const cmTensor = Tensor.fromArray(null, cmData, [2, 2, 1]);
const clutData = new Float32Array([
  0.0, 0.1, 0.2,
  0.3, 0.4, 0.5,
  0.6, 0.7, 0.8,
  0.9, 1.0, 0.1,
]);
const clutTensor = Tensor.fromArray(null, clutData, [2, 2, 3]);
const jsColorMap = colorMap(cmTensor, [2, 2, 1], 0, 1, clutTensor, false, 0.5).read();
const colorMapExpected = loadFixture('colorMap.json');
arraysClose(Array.from(jsColorMap), colorMapExpected);

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

// vortex deterministic
const vxData = new Float32Array([0.1, 0.2, 0.3, 0.4]);
const vxTensor = Tensor.fromArray(null, vxData, [2,2,1]);
const randV = simplexRandom(0, undefined, 1);
const centerX = 2 / 2;
const centerY = 2 / 2;
const vxX = new Float32Array(4);
const vxY = new Float32Array(4);
for (let y = 0; y < 2; y++) {
  for (let x = 0; x < 2; x++) {
    const dx = x - centerX;
    const dy = y - centerY;
    const dist = Math.sqrt(dx * dx + dy * dy) + 1e-6;
    const fade = 1 - Math.max(Math.abs(dx) / centerX, Math.abs(dy) / centerY);
    const nx = (-dy / dist) * fade;
    const ny = (dx / dist) * fade;
    const idx = y * 2 + x;
    vxX[idx] = nx * 0.5 + 0.5;
    vxY[idx] = ny * 0.5 + 0.5;
  }
}
const vxManual = refract(
  vxTensor,
  Tensor.fromArray(null, vxX, [2,2,1]),
  Tensor.fromArray(null, vxY, [2,2,1]),
  randV * 100 * 64,
).read();
const vxResult = vortex(vxTensor, [2,2,1], 0, 1, 64).read();
arraysClose(Array.from(vxResult), Array.from(vxManual));

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

// bloom regression
const bloomData = new Float32Array([0.1, 0.2, 0.3, 0.4]);
const bloomTensor = Tensor.fromArray(null, bloomData, [2,2,1]);
const bloomOut = bloom(bloomTensor, [2,2,1], 0, 1, 1.0).read();
const bloomExpected = loadFixture('bloom.json');
arraysClose(Array.from(bloomOut), bloomExpected);

// vaseline regression
const vasOut = vaseline(bloomTensor, [2,2,1], 0, 1, 1.0).read();
const vasExpected = loadFixture('vaseline.json');
arraysClose(Array.from(vasOut), vasExpected);

// lightLeak regression
setSeed(1);
const leakOut = lightLeak(bloomTensor, [2,2,1], 0, 1, 0.25).read();
const leakExpected = loadFixture('lightLeak.json');
arraysClose(Array.from(leakOut), leakExpected);

// wobble regression
setSeed(1);
const wobData = new Float32Array([0.1, 0.2, 0.3, 0.4]);
const wobTensor = Tensor.fromArray(null, wobData, [2,2,1]);
setSeed(1);
const wobOut = wobble(wobTensor, [2,2,1], 0.8, 1).read();
const wobExpected = loadFixture('wobble.json');
arraysClose(Array.from(wobOut), wobExpected);

// reverb regression
const revData = new Float32Array([0.1, 0.4, 0.6, 0.9]);
const revTensor = Tensor.fromArray(null, revData, [2,2,1]);
const revOut = reverb(revTensor, [2,2,1], 0, 1, 2, 1, true).read();
const revExpected = loadFixture('reverb.json');
arraysClose(Array.from(revOut), revExpected);

// dla regression
setSeed(7);
const dlaData = new Float32Array([
  0.1, 0.2, 0.3, 0.4,
  0.5, 0.6, 0.7, 0.8,
  0.9, 0.8, 0.7, 0.6,
  0.5, 0.4, 0.3, 0.2,
]);
const dlaTensor = Tensor.fromArray(null, dlaData, [4,4,1]);
setSeed(7);
const dlaOut = dla(dlaTensor, [4,4,1], 1, 1, 1, 1, 0.5).read();
const dlaExpected = loadFixture('dla.json');
arraysClose(Array.from(dlaOut), dlaExpected);

// rotate effect
const rotData = new Float32Array([
  1, 2,
  3, 4,
]);
const rotTensor = Tensor.fromArray(null, rotData, [2,2,1]);
const rotOut = rotate(rotTensor, [2,2,1], 0, 1, 90).read();
assert.deepStrictEqual(Array.from(rotOut), [1,3,2,4]);

// pixelSort effect
const psData = new Float32Array([0.2, 0.8, 0.5, 0.1]);
const psTensor = Tensor.fromArray(null, psData, [1,4,1]);
const psOut = pixelSort(psTensor, [1,4,1], 0, 1).read();
assert.strictEqual(psOut.length, 4);
assert.notDeepStrictEqual(Array.from(psOut), [0.2,0.8,0.5,0.1]);

// sketch regression
setSeed(1);
const skData = new Float32Array([0.1, 0.5, 0.3, 0.8]);
const skTensor = Tensor.fromArray(null, skData, [2,2,1]);
setSeed(1);
const skOut = sketch(skTensor, [2,2,1], 0, 1).read();
const skExpected = loadFixture('sketch.json');
arraysClose(Array.from(skOut), skExpected);

// simpleFrame regression
const sfTensor = Tensor.fromArray(null, skData, [2,2,1]);
const sfOut = simpleFrame(sfTensor, [2,2,1], 0, 1, 0.5).read();
const sfExpected = loadFixture('simpleFrame.json');
arraysClose(Array.from(sfOut), sfExpected);

// frame regression
setSeed(1);
const frTensor = Tensor.fromArray(null, skData, [2,2,1]);
setSeed(1);
const frOut = frame(frTensor, [2,2,1], 0, 1).read();
const frExpected = loadFixture('frame.json');
arraysClose(Array.from(frOut), frExpected);

// lowpoly regression
setSeed(1);
const lpOut = lowpoly(edgeTensor, [4,4,1], 0, 1, undefined, 2).read();
const lpExpected = loadFixture('lowpoly.json');
arraysClose(Array.from(lpOut), lpExpected);

// kaleido regression
setSeed(1);
const kalOut = kaleido(edgeTensor, [4,4,1], 0, 1, 3).read();
const kalExpected = loadFixture('kaleido.json');
arraysClose(Array.from(kalOut), kalExpected);

// texture regression
const texOut = texture(edgeTensor, [4,4,1], 0, 1).read();
const texExpected = loadFixture('texture.json');
arraysClose(Array.from(texOut), texExpected);

// glyphMap shape test
const gmTensor = Tensor.fromArray(null, edgeData, [4,4,1]);
const gmOut = glyphMap(gmTensor, [4,4,1], 0, 1).read();
assert.strictEqual(gmOut.length, 16);

// glitch regression
const glData = new Float32Array([
  0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
  0.7, 0.8, 0.9, 0.1, 0.2, 0.3,
]);
const glTensor = Tensor.fromArray(null, glData, [2,2,3]);
const glOut = glitch(glTensor, [2,2,3], 0.25, 1).read();
const glExpected = loadFixture('glitch.json');
arraysClose(Array.from(glOut), glExpected);

// vhs regression
const vhsTensor = Tensor.fromArray(null, glData, [2,2,3]);
const vhsOut = vhs(vhsTensor, [2,2,3], 0.25, 1).read();
const vhsExpected = loadFixture('vhs.json');
arraysClose(Array.from(vhsOut), vhsExpected);

// scanlineError regression
const sleData = new Float32Array([0.1, 0.2, 0.3, 0.4]);
const sleTensor = Tensor.fromArray(null, sleData, [2,2,1]);
const sleOut = scanlineError(sleTensor, [2,2,1], 0.25, 1).read();
const sleExpected = loadFixture('scanlineError.json');
arraysClose(Array.from(sleOut), sleExpected);

// crt regression
setSeed(1);
const crtTensor = Tensor.fromArray(null, glData, [2,2,3]);
setSeed(1);
const crtOut = crt(crtTensor, [2,2,3], 0.25, 1).read();
const crtExpected = loadFixture('crt.json');
arraysClose(Array.from(crtOut), crtExpected);

// snow regression
const snowData = new Float32Array([0.1, 0.2, 0.3, 0.4]);
const snowTensor = Tensor.fromArray(null, snowData, [2,2,1]);
const snowOut = snow(snowTensor, [2,2,1], 0, 1).read();
const snowExpected = loadFixture('snow.json');
arraysClose(Array.from(snowOut), snowExpected);

// spatter regression
setSeed(1);
const spTensor = Tensor.fromArray(null, new Float32Array(16), [4,4,1]);
setSeed(1);
const spOut = spatter(spTensor, [4,4,1], 0, 1).read();
const spExpected = loadFixture('spatter.json');
arraysClose(Array.from(spOut), spExpected);

// clouds regression
setSeed(1);
const clTensor = Tensor.fromArray(null, new Float32Array(16), [4,4,1]);
setSeed(1);
const clOut = clouds(clTensor, [4,4,1], 0, 1).read();
const clExpected = loadFixture('clouds.json');
arraysClose(Array.from(clOut), clExpected);

// fibers regression
setSeed(1);
const fbTensor = Tensor.fromArray(null, new Float32Array(16), [4,4,1]);
setSeed(1);
const fbOut = fibers(fbTensor, [4,4,1], 0, 1).read();
const fbExpected = loadFixture('fibers.json');
arraysClose(Array.from(fbOut), fbExpected);
const fbArr = Array.from(fbOut);
assert.ok(fbArr.every(v => v >= 0 && v <= 1));

// scratches regression
setSeed(1);
const scTensor = Tensor.fromArray(null, new Float32Array(16), [4,4,1]);
setSeed(1);
const scOut = scratches(scTensor, [4,4,1], 0, 1).read();
const scArr = Array.from(scOut);
assert.ok(scArr.some(v => v > 0));
assert.ok(scArr.every(v => v >= 0 && v <= 1));

// strayHair regression
setSeed(1);
const shTensor = Tensor.fromArray(null, new Float32Array(16), [4,4,1]);
setSeed(1);
const shOut = strayHair(shTensor, [4,4,1], 0, 1).read();
const shExpected = loadFixture('strayHair.json');
arraysClose(Array.from(shOut), shExpected);
const shArr = Array.from(shOut);
assert.ok(shArr.every(v => v >= 0 && v <= 1));

// grime regression
setSeed(1);
const grimeTensor = Tensor.fromArray(null, new Float32Array(64), [8,8,1]);
setSeed(1);
const grOut = grime(grimeTensor, [8,8,1], 0, 1).read();
const grExpected = loadFixture('grime.json');
arraysClose(Array.from(grOut), grExpected);

// watermark regression
setSeed(1);
const watermarkTensor = Tensor.fromArray(null, new Float32Array(64), [8,8,1]);
setSeed(1);
const wmOut = watermark(watermarkTensor, [8,8,1], 0, 1).read();
const wmExpected = loadFixture('watermark.json');
arraysClose(Array.from(wmOut), wmExpected);

// spookyTicker regression
setSeed(1);
const spookyTensor = Tensor.fromArray(null, new Float32Array(64), [8,8,1]);
setSeed(1);
const stOut = spookyTicker(spookyTensor, [8,8,1], 0, 1).read();
const stExpected = loadFixture('spookyTicker.json');
arraysClose(Array.from(stOut), stExpected);

// onScreenDisplay regression
setSeed(1);
const osdTensor = Tensor.fromArray(null, new Float32Array(64 * 64), [64, 64, 1]);
setSeed(1);
const osdOut = onScreenDisplay(osdTensor, [64, 64, 1], 0, 1).read();
const osdExpected = loadFixture('onScreenDisplay.json');
arraysClose(Array.from(osdOut), osdExpected);

// nebula regression
setSeed(1);
const nebTensor = Tensor.fromArray(null, new Float32Array(32 * 32 * 3), [32, 32, 3]);
setSeed(1);
const nebOut = nebula(nebTensor, [32, 32, 3], 0, 1).read();
const nebExpected = loadFixture('nebula.json');
arraysClose(Array.from(nebOut), nebExpected);

// falseColor regression
const fcData = new Float32Array([
  0.1, 0.1, 0.1,
  0.2, 0.2, 0.2,
  0.3, 0.3, 0.3,
  0.4, 0.4, 0.4,
]);
const fcTensor = Tensor.fromArray(null, fcData, [2, 2, 3]);
const fcOut = falseColor(fcTensor, [2, 2, 3], 0, 1).read();
const fcExpected = loadFixture('falseColor.json');
arraysClose(Array.from(fcOut), fcExpected);

// tint regression
setSeed(1);
const tintData = new Float32Array([
  0.1, 0.2, 0.3,
  0.4, 0.5, 0.6,
  0.7, 0.8, 0.9,
  0.2, 0.3, 0.4,
]);
const tintTensor = Tensor.fromArray(null, tintData, [2, 2, 3]);
setSeed(1);
const tintOut = tint(tintTensor, [2, 2, 3], 0, 1, 0.5).read();
const tintExpected = loadFixture('tint.json');
arraysClose(Array.from(tintOut), tintExpected);
const tintArr = Array.from(tintOut);
assert.ok(tintArr.every((v) => v >= 0 && v <= 1));

// valueRefract regression
const vrData = new Float32Array([
  0.0, 0.1, 0.2, 0.3,
  0.4, 0.5, 0.6, 0.7,
  0.8, 0.9, 1.0, 0.1,
  0.2, 0.3, 0.4, 0.5,
]);
const vrTensor = Tensor.fromArray(null, vrData, [4, 4, 1]);
const vrOut = valueRefract(vrTensor, [4, 4, 1], 0, 1).read();
const vrExpected = loadFixture('valueRefract.json');
arraysClose(Array.from(vrOut), vrExpected);
const vrArr = Array.from(vrOut);
assert.ok(vrArr.every((v) => v >= 0 && v <= 1));

// lensWarp extreme displacement
setSeed(1);
const lwOut = lensWarp(edgeTensor, [4,4,1], 0, 1, 5).read();
for (const v of lwOut) {
  assert.ok(Number.isFinite(v));
  assert.ok(v >= 0 && v <= 1);
}

// lensDistortion negative displacement regression
const ldDisp = -2;
const ldRes = lensDistortion(edgeTensor, [4,4,1], 0, 1, ldDisp).read();
const ldManual = (() => {
  const h = 4, w = 4;
  const out = new Float32Array(h * w);
  const maxDist = Math.sqrt(0.5 * 0.5 + 0.5 * 0.5);
  const zoom = ldDisp < 0 ? ldDisp * -0.25 : 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const xIndex = x / w;
      const yIndex = y / h;
      const xDist = xIndex - 0.5;
      const yDist = yIndex - 0.5;
      const centerDist = 1 - distance(xDist, yDist) / maxDist;
      const xOff = ((xIndex - xDist * zoom) - xDist * centerDist * centerDist * ldDisp) * w;
      const yOff = ((yIndex - yDist * zoom) - yDist * centerDist * centerDist * ldDisp) * h;
      const xi = ((Math.floor(xOff) % w) + w) % w;
      const yi = ((Math.floor(yOff) % h) + h) % h;
      out[y * w + x] = edgeData[yi * w + xi];
    }
  }
  return out;
})();
arraysClose(Array.from(ldRes), Array.from(ldManual));

// degauss per-channel lensWarp consistency
setSeed(2);
const dgTensor = Tensor.fromArray(null, glData, [2,2,3]);
const dgOut = degauss(dgTensor, [2,2,3], 0, 1, 1).read();
setSeed(2);
const channelShape = [2,2,1];
const rChan = Tensor.fromArray(null, new Float32Array([0.1, 0.4, 0.7, 0.1]), channelShape);
const gChan = Tensor.fromArray(null, new Float32Array([0.2, 0.5, 0.8, 0.2]), channelShape);
const bChan = Tensor.fromArray(null, new Float32Array([0.3, 0.6, 0.9, 0.3]), channelShape);
const rWarp = lensWarp(rChan, channelShape, 0, 1, 1).read();
const gWarp = lensWarp(gChan, channelShape, 0, 1, 1).read();
const bWarp = lensWarp(bChan, channelShape, 0, 1, 1).read();
const dgManual = new Float32Array(2 * 2 * 3);
for (let i = 0; i < 4; i++) {
  dgManual[i * 3] = rWarp[i];
  dgManual[i * 3 + 1] = gWarp[i];
  dgManual[i * 3 + 2] = bWarp[i];
}
arraysClose(Array.from(dgOut), Array.from(dgManual));

console.log('Effects tests passed');
