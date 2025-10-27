import assert from 'assert';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { PRESETS as JSPRESETS, setSeed } from '../js/noisemaker/presets.js';

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
  'cell-reflect',
  'density-wave',
  'glom',
  'jorts',
  'jovian-clouds',
  'paintball-party',
  'pearlescent',
  'posterize',
  'quadrants',
  'rasteroids',
  'sbup',
  'symmetry',
  'the-data-must-flow',
];

function getPythonPresetData() {
const py = `import json, random\nfrom noisemaker import rng\nfrom noisemaker.presets import PRESETS\nseeds=[0,1,2,3,4]\npreset_names=sorted(PRESETS().keys())\ncombined={}\nfor name in preset_names:\n    layers=None\n    settings_per_seed=[]\n    for seed in seeds:\n        random.seed(seed)\n        rng.set_seed(seed)\n        preset=PRESETS()[name]\n        if layers is None and preset.get('layers'):\n            layers=preset['layers']\n        if preset.get('settings'):\n            s=preset['settings']()\n            s={k:getattr(v,'value',v) for k,v in s.items()}\n        else:\n            s={}\n        settings_per_seed.append(s)\n    entry={}\n    if layers is not None:\n        entry['layers']=layers\n    if settings_per_seed and settings_per_seed[0]:\n        s={}\n        for key in settings_per_seed[0]:\n            vals=[d.get(key) for d in settings_per_seed]\n            first=vals[0]\n            s[key]=first if all(v==first for v in vals) else 'RANDOM'\n        entry['settings']=s\n    combined[name]=entry\nprint(json.dumps(combined))`;
  const res = spawnSync('python3', ['-c', py], {
    cwd: repoRoot,
    encoding: 'utf8',
    maxBuffer: 1024 * 1024 * 50,
  });
  if (res.status !== 0) {
    throw new Error(res.stderr);
  }
  return JSON.parse(res.stdout);
}

function getJSPresetData() {
  const seeds = [0, 1, 2, 3, 4];
  const presetNames = Object.keys(JSPRESETS()).sort();
  const combined = {};

  for (const name of presetNames) {
    let layers = null;
    const settingsPerSeed = [];

    for (const seed of seeds) {
      setSeed(seed);
      const preset = JSPRESETS()[name];
      if (layers === null && preset.layers) {
        layers = preset.layers.slice();
      }
      const s = typeof preset.settings === 'function' ? { ...preset.settings() } : {};
      settingsPerSeed.push(s);
    }

    const entry = {};
    if (layers) entry.layers = layers;
    if (settingsPerSeed.length && Object.keys(settingsPerSeed[0]).length) {
      const s = {};
      for (const key of Object.keys(settingsPerSeed[0])) {
        const vals = settingsPerSeed.map((d) => d[key]);
        const first = vals[0];
        s[key] = vals.every((v) => v === first) ? first : 'RANDOM';
      }
      entry.settings = s;
    }
    combined[name] = entry;
  }

  return combined;
}

const pyData = getPythonPresetData();
const jsData = getJSPresetData();

// Align JS classification with Python for randomised params
for (const [name, pyPreset] of Object.entries(pyData)) {
  const jsPreset = jsData[name];
  if (!jsPreset || !pyPreset.settings || !jsPreset.settings) continue;
  for (const [key, val] of Object.entries(pyPreset.settings)) {
    if (val === 'RANDOM') jsPreset.settings[key] = 'RANDOM';
  }
}

const missing = [];
const extra = [];
const mismatched = [];

for (const [name, pyPreset] of Object.entries(pyData)) {
  if (JS_ONLY.includes(name)) continue;
  const jsPreset = jsData[name];
  if (!jsPreset) {
    missing.push(name);
    continue;
  }
  try {
    assert.deepStrictEqual(jsPreset, pyPreset);
  } catch (e) {
    mismatched.push(name);
  }
}

for (const name of Object.keys(jsData)) {
  if (JS_ONLY.includes(name)) continue;
  if (!(name in pyData)) extra.push(name);
}

if (missing.length || extra.length || mismatched.length) {
  let msg = '';
  if (missing.length) msg += `Missing JS presets: ${missing.join(', ')}\n`;
  if (extra.length) msg += `Extra JS presets: ${extra.join(', ')}\n`;
  if (mismatched.length) msg += `Mismatched presets: ${mismatched.join(', ')}`;
  assert.fail(msg.trim());
}

console.log('preset params parity ok');
