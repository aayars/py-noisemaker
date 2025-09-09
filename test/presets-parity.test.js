import assert from 'assert';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { PRESETS as JSPRESETS } from '../src/presets.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..');

function getPythonPresets() {
  const py = `import json\nfrom noisemaker.presets import PRESETS\nprint(json.dumps(sorted(PRESETS().keys())))`;
  const res = spawnSync('python', ['-c', py], { cwd: repoRoot, encoding: 'utf8' });
  if (res.status !== 0) {
    throw new Error(res.stderr);
  }
  return JSON.parse(res.stdout);
}

const pyPresets = getPythonPresets().sort();
const jsPresets = Object.keys(JSPRESETS()).sort();

const JS_ONLY = [
  'erode-post',
  'ghost-diagram',
  'ghost',
  'maybe-hyperspace',
  'maybe-mask',
  'shake-it',
  'shrink-triangulate',
];

const missing = pyPresets.filter((p) => !jsPresets.includes(p));
const extra = jsPresets.filter(
  (p) => !pyPresets.includes(p) && !JS_ONLY.includes(p),
);

if (missing.length || extra.length) {
  let msg = '';
  if (missing.length) msg += `Missing JS presets: ${missing.join(', ')}\n`;
  if (extra.length) msg += `Extra JS presets: ${extra.join(', ')}`;
  assert.fail(msg.trim());
}

console.log('preset parity ok');
