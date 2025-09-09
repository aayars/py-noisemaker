import assert from 'assert';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { PRESETS as JSPRESETS } from '../src/presets.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..');

const JS_ONLY = [
  'erode-post',
  'ghost-diagram',
  'ghost',
  'maybe-hyperspace',
  'maybe-mask',
  'shake-it',
  'shrink-triangulate',
];

function getPythonPresetSettings() {
  const py = `import json, random\nfrom noisemaker.presets import PRESETS\nrandom.seed(123)\nkeys={name:sorted(p.get('settings')().keys()) for name,p in PRESETS().items() if p.get('settings')}\nprint(json.dumps(keys))`;
  const res = spawnSync('python', ['-c', py], { cwd: repoRoot, encoding: 'utf8' });
  if (res.status !== 0) {
    throw new Error(res.stderr);
  }
  return JSON.parse(res.stdout);
}

const py = getPythonPresetSettings();
const jsPresets = JSPRESETS();

for (const [name, pyKeys] of Object.entries(py)) {
  if (JS_ONLY.includes(name)) continue;
  const preset = jsPresets[name];
  assert(preset, `Missing JS preset: ${name}`);
  assert.strictEqual(typeof preset.settings, 'function', `${name} missing settings()`);
  const jsKeys = Object.keys(preset.settings()).sort();
  assert.deepStrictEqual(jsKeys, pyKeys, `${name} settings mismatch`);
}

for (const [name, preset] of Object.entries(jsPresets)) {
  if (JS_ONLY.includes(name)) continue;
  if (typeof preset.settings === 'function' && !(name in py)) {
    assert.fail(`Extra JS preset with settings: ${name}`);
  }
}

console.log('preset params parity ok');
