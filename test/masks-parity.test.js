import assert from 'assert';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { ValueMask } from '../js/noisemaker/constants.js';
import { Masks as JsMasks, squareMasks } from '../js/noisemaker/masks.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..');

function getPythonMasks() {
  const py = `
import json
import numpy as np
np.random.seed(0)
from noisemaker.masks import Masks
result = {}
for mask, value in Masks.items():
    if not callable(value):
        result[mask.name] = value
print(json.dumps(result))
`;
  const res = spawnSync('python3', ['-c', py], { cwd: repoRoot, encoding: 'utf8' });
  if (res.status !== 0) {
    throw new Error(res.stderr);
  }
  return JSON.parse(res.stdout);
}

const pyMasks = getPythonMasks();
const mismatches = [];

for (const [name, pyMask] of Object.entries(pyMasks)) {
  const enumVal = ValueMask[name];
  if (enumVal === undefined) {
    mismatches.push(`Missing ValueMask.${name} in JS constants`);
    continue;
  }
  const jsMask = JsMasks[enumVal];
  if (!jsMask || typeof jsMask === 'function') {
    mismatches.push(`Missing JS mask for ${name}`);
    continue;
  }
  try {
    assert.deepStrictEqual(jsMask, pyMask);
  } catch (e) {
    mismatches.push(`Mask ${name} differs`);
  }
}

for (const [enumName, enumVal] of Object.entries(ValueMask)) {
  const jsMask = JsMasks[enumVal];
  if (jsMask && typeof jsMask !== 'function' && !(enumName in pyMasks)) {
    mismatches.push(`Extra JS mask ${enumName}`);
  }
}

function getPythonSquareMasks() {
  const py = `
import json
from noisemaker.masks import square_masks
print(json.dumps([m.name for m in square_masks()]))
`;
  const res = spawnSync('python3', ['-c', py], { cwd: repoRoot, encoding: 'utf8' });
  if (res.status !== 0) {
    throw new Error(res.stderr);
  }
  return JSON.parse(res.stdout);
}

const pySquares = getPythonSquareMasks();
const jsSquares = squareMasks().map((m) => Object.entries(ValueMask).find(([, v]) => v === m)[0]);
if (JSON.stringify(jsSquares.sort()) !== JSON.stringify(pySquares.sort())) {
  mismatches.push('squareMasks mismatch');
}

if (mismatches.length) {
  assert.fail('Mask mismatches:\n' + mismatches.join('\n'));
}

console.log('mask parity ok');
