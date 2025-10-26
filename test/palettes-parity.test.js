import assert from 'assert';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { PALETTES as jsPalettes } from '../js/noisemaker/palettes.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..');

function getPythonPalettes() {
  const py = `
import json
from noisemaker.palettes import PALETTES
print(json.dumps(PALETTES))
`;
  const res = spawnSync('python', ['-c', py], { cwd: repoRoot, encoding: 'utf8' });
  if (res.status !== 0) {
    throw new Error(res.stderr);
  }
  return JSON.parse(res.stdout);
}

const pyPalettes = getPythonPalettes();
assert.deepStrictEqual(jsPalettes, pyPalettes, 'Palette mismatch');

console.log('palettes parity ok');
