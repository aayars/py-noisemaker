import assert from 'assert';
import { Tensor } from '../src/tensor.js';
import { posterize, palette, invert, aberration } from '../src/effects.js';
import { adjustHue } from '../src/value.js';
import { setSeed, random } from '../src/util.js';
import { spawnSync } from 'child_process';

function arraysClose(a, b, eps = 1e-6) {
  assert.strictEqual(a.length, b.length);
  for (let i = 0; i < a.length; i++) {
    assert.ok(Math.abs(a[i] - b[i]) < eps, `index ${i}`);
  }
}

// posterize regression
const posterData = new Float32Array([0.1, 0.5, 0.9, 0.3]);
const posterTensor = Tensor.fromArray(null, posterData, [2, 2, 1]);
const jsPoster = posterize(posterTensor, [2, 2, 1], 0, 1, 4).read();
const posterPy = spawnSync('python', ['-'], {
  input: `import json, tensorflow as tf\nfrom noisemaker.effects import posterize\nvals=${JSON.stringify(Array.from(posterData))}\ntex=tf.constant(vals, shape=[2,2,1], dtype=tf.float32)\nout=posterize(tex,[2,2,1],levels=4)\nprint(json.dumps(out.numpy().flatten().tolist()))`,
  encoding: 'utf8'
});
const posterExpected = JSON.parse(posterPy.stdout.trim());
arraysClose(Array.from(jsPoster), posterExpected);

// palette regression
const palData = new Float32Array([0.0, 0.5, 1.0, 0.25]);
const palTensor = Tensor.fromArray(null, palData, [2, 2, 1]);
const jsPal = palette(palTensor, [2, 2, 1], 0, 1, 'grayscale').read();
const palPy = spawnSync('python', ['-'], {
  input: `import json, math\nfrom noisemaker.palettes import PALETTES\nvals=[0.0,0.5,1.0,0.25]\np=PALETTES['grayscale']\nout=[]\nfor t in vals:\n    r=p['offset'][0]+p['amp'][0]*math.cos(math.tau*(p['freq'][0]*t*0.875+0.0625+p['phase'][0]))\n    g=p['offset'][1]+p['amp'][1]*math.cos(math.tau*(p['freq'][1]*t*0.875+0.0625+p['phase'][1]))\n    b=p['offset'][2]+p['amp'][2]*math.cos(math.tau*(p['freq'][2]*t*0.875+0.0625+p['phase'][2]))\n    out.extend([r,g,b])\nprint(json.dumps(out))`,
  encoding: 'utf8'
});
const palExpected = JSON.parse(palPy.stdout.trim());
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
const shifted = adjustHue(abTensor, hueShift).read();
const manual = new Float32Array(abShape[0] * abShape[1] * 3);
for (let x = 0; x < abShape[1]; x++) {
  const base = x * 3;
  const rIdx = Math.min(abShape[1] - 1, x + disp) * 3;
  const bIdx = Math.max(0, x - disp) * 3;
  manual[base] = shifted[rIdx];
  manual[base + 1] = shifted[base + 1];
  manual[base + 2] = shifted[bIdx + 2];
}
const expected = adjustHue(Tensor.fromArray(null, manual, abShape), -hueShift).read();
setSeed(123);
const abResult = aberration(abTensor, abShape, 0, 1, 0.25).read();
arraysClose(Array.from(abResult), Array.from(expected));

console.log('Effects tests passed');
