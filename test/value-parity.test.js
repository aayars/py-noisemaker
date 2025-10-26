import assert from 'assert';
import { values } from '../js/noisemaker/value.js';
import { ValueDistribution } from '../js/noisemaker/constants.js';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..');

function getPythonValues(distribName, freq, shape) {
  const py = `
import json
from noisemaker.value import values
from noisemaker.constants import ValueDistribution
freq = ${JSON.stringify(freq)}
shape = ${JSON.stringify(shape)}
t = values(freq, shape, distrib=ValueDistribution.${distribName})
print(json.dumps(t.numpy().flatten().tolist()))
`;
  const res = spawnSync('python', ['-c', py], { cwd: repoRoot, encoding: 'utf8' });
  if (res.status !== 0) {
    throw new Error(res.stderr);
  }
  return JSON.parse(res.stdout);
}

function arraysClose(a, b, eps = 1e-6) {
  assert.strictEqual(a.length, b.length);
  for (let i = 0; i < a.length; i++) {
    assert.ok(Math.abs(a[i] - b[i]) < eps, `index ${i}`);
  }
}

const rowShape = [2, 3, 1];
const rowFreq = [2, 3];
const pyRow = getPythonValues('row_index', rowFreq, rowShape);
const jsRow = Array.from(values(rowFreq, rowShape, { distrib: ValueDistribution.row_index }).read());
arraysClose(jsRow, pyRow);

const colShape = [3, 2, 1];
const colFreq = [3, 2];
const pyCol = getPythonValues('column_index', colFreq, colShape);
const jsCol = Array.from(values(colFreq, colShape, { distrib: ValueDistribution.column_index }).read());
arraysClose(jsCol, pyCol);

console.log('value parity ok');
