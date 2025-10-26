import { PRESETS, setSeed } from '../../js/noisemaker/presets.js';
import { Preset } from '../../js/noisemaker/composer.js';

function snakeCase(key) {
  return key.replace(/([A-Z])/g, '_$1').toLowerCase();
}

function snakeKeys(obj) {
  if (Array.isArray(obj)) {
    return obj.map(snakeKeys);
  }
  if (obj && typeof obj === 'object') {
    const out = {};
    for (const [k, v] of Object.entries(obj)) {
      out[snakeCase(k)] = snakeKeys(v);
    }
    return out;
  }
  return obj;
}

const [,, name, seedStr] = process.argv;
const seed = parseInt(seedStr, 10);
// Seed the RNG once before building presets. The Preset constructor should
// not reset the seed so that any random calls made while constructing the
// preset (e.g., during PRESETS()) advance the RNG identically to Python.
setSeed(seed);
// Build only the requested preset (and its ancestors) so that RNG state
// remains aligned with Python, which does not evaluate unrelated presets.
const presets = PRESETS(name);
const preset = new Preset(name, presets);
const raw = JSON.parse(JSON.stringify(preset.settings));
console.log(JSON.stringify(snakeKeys(raw)));
