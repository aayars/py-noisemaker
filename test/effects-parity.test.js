import assert from 'assert';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { list as listEffects } from '../js/noisemaker/effectsRegistry.js';
import '../js/noisemaker/effects.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..');

function getPythonEffects() {
  const py = `
import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import noisemaker.effects  # populate EFFECTS
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

function toSnakeCase(name) {
  return name
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replace(/-/g, '_')
    .toLowerCase();
}
const pyEffects = getPythonEffects();
const jsEffects = listEffects().filter((name) => name !== 'list');

const jsNames = new Set();
for (const name of jsEffects) {
  jsNames.add(name);
  jsNames.add(toCamelCase(name));
  jsNames.add(toSnakeCase(name));
}

const missing = [];
for (const pyName of pyEffects) {
  const camel = toCamelCase(pyName);
  if (!jsNames.has(pyName) && !jsNames.has(camel)) {
    missing.push(pyName);
  }
}

if (missing.length) {
  assert.fail(`Missing JS effects: ${missing.join(', ')}`);
}

const pyCanonical = new Set();
for (const name of pyEffects) {
  pyCanonical.add(name);
  pyCanonical.add(toCamelCase(name));
}

const extra = jsEffects
  .map((name) => ({ original: name, snake: toSnakeCase(name) }))
  .filter(({ original, snake }) => !pyCanonical.has(original) && !pyCanonical.has(snake))
  .map(({ original }) => original)
  .sort();

if (extra.length) {
  console.warn(`Extra JS effects (not present in Python registry): ${extra.join(', ')}`);
}

console.log('effect parity ok');
