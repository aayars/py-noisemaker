import assert from 'assert';
import { rgbToHsv } from '../src/value.js';
import { Tensor } from '../src/tensor.js';
import { spawnSync } from 'child_process';

const rgb = [0.2, 0.4, 0.6];
const tensor = Tensor.fromArray(null, new Float32Array(rgb), [1, 1, 3]);
const jsHsv = Array.from(rgbToHsv(tensor).read());

const pyCode = `
import colorsys, json
rgb = ${JSON.stringify(rgb)}
hsv = colorsys.rgb_to_hsv(*rgb)
print(json.dumps(hsv))
`;
const py = spawnSync('python', ['-'], { input: pyCode, encoding: 'utf8' });
const pyHsv = JSON.parse(py.stdout.trim());

for (let i = 0; i < 3; i++) {
  assert.ok(Math.abs(jsHsv[i] - pyHsv[i]) < 1e-6, `channel ${i}`);
}

console.log('Color conversion tests passed');
