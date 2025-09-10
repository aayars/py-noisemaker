import assert from 'assert';
import { rgbToHsv, hsvToRgb } from '../src/value.js';
import { Tensor } from '../src/tensor.js';
import { spawnSync } from 'child_process';

// rgb -> hsv parity with Python
const rgb = [0.2, 0.4, 0.6];
const rgbTensor = Tensor.fromArray(null, new Float32Array(rgb), [1, 1, 3]);
const jsHsv = Array.from(rgbToHsv(rgbTensor).read());

const pyRgbToHsv = `
import colorsys, json
rgb = ${JSON.stringify(rgb)}
hsv = colorsys.rgb_to_hsv(*rgb)
print(json.dumps(hsv))
`;
const py1 = spawnSync('python', ['-'], { input: pyRgbToHsv, encoding: 'utf8' });
const pyHsv = JSON.parse(py1.stdout.trim());

for (let i = 0; i < 3; i++) {
  assert.ok(Math.abs(jsHsv[i] - pyHsv[i]) < 1e-6, `rgb->hsv channel ${i}`);
}

// hsv -> rgb parity with Python
const hsv = [0.7, 0.5, 0.3];
const hsvTensor = Tensor.fromArray(null, new Float32Array(hsv), [1, 1, 3]);
const jsRgb = Array.from(hsvToRgb(hsvTensor).read());

const pyHsvToRgb = `
import colorsys, json
hsv = ${JSON.stringify(hsv)}
rgb = colorsys.hsv_to_rgb(*hsv)
print(json.dumps(rgb))
`;
const py2 = spawnSync('python', ['-'], { input: pyHsvToRgb, encoding: 'utf8' });
const pyRgb = JSON.parse(py2.stdout.trim());

for (let i = 0; i < 3; i++) {
  assert.ok(Math.abs(jsRgb[i] - pyRgb[i]) < 1e-6, `hsv->rgb channel ${i}`);
}

console.log('Color conversion tests passed');
