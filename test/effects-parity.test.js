import assert from 'assert';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { list as listEffects } from '../src/effectsRegistry.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..');

function getPythonEffects() {
  const py = `
import json
from noisemaker.effects_registry import EFFECTS
print(json.dumps(sorted(EFFECTS.keys())))
`;
  const res = spawnSync('python', ['-c', py], { cwd: repoRoot, encoding: 'utf8' });
  if (res.status !== 0) {
    throw new Error(res.stderr);
  }
  return JSON.parse(res.stdout);
}

function toCamelCase(name) {
  return name.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
}

const pyEffects = getPythonEffects().map(toCamelCase).sort();
const jsEffects = listEffects()
  .filter((name) => name !== 'list')
  .slice()
  .sort();

const missing = pyEffects.filter((e) => !jsEffects.includes(e));
const extra = jsEffects.filter((e) => !pyEffects.includes(e));

if (missing.length || extra.length) {
  let msg = '';
  if (missing.length) msg += `Missing JS effects: ${missing.join(', ')}\n`;
  if (extra.length) msg += `Extra JS effects: ${extra.join(', ')}`;
  assert.fail(msg.trim());
}

console.log('effect parity ok');
