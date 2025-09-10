import assert from 'assert';
import { rgbToHsv, hsvToRgb } from '../src/value.js';
import { Tensor } from '../src/tensor.js';
import { srgbToLin, linToSRGB, fromSRGB, toSRGB } from '../src/util.js';
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

// sRGB <-> linear conversions preserve alpha
const rgba = [0.2, 0.4, 0.6, 0.3];
const rgbaTensor = Tensor.fromArray(null, new Float32Array(rgba), [1, 1, 4]);
const linTensor = fromSRGB(rgbaTensor);
const linData = Array.from(linTensor.read());
assert.ok(Math.abs(linData[0] - srgbToLin(rgba[0])) < 1e-6, 'red channel');
assert.ok(Math.abs(linData[1] - srgbToLin(rgba[1])) < 1e-6, 'green channel');
assert.ok(Math.abs(linData[2] - srgbToLin(rgba[2])) < 1e-6, 'blue channel');
assert.ok(Math.abs(linData[3] - rgba[3]) < 1e-6, 'alpha channel');

const srgbTensor = toSRGB(linTensor);
const srgbData = Array.from(srgbTensor.read());
for (let i = 0; i < 3; i++) {
  assert.ok(Math.abs(srgbData[i] - rgba[i]) < 1e-6, `roundtrip channel ${i}`);
}
assert.ok(Math.abs(srgbData[3] - rgba[3]) < 1e-6, 'roundtrip alpha');

console.log('Color conversion tests passed');
