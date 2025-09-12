import { PRESETS, setSeed } from '../../src/presets.js';
import { Preset } from '../../src/composer.js';

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
setSeed(seed);
const preset = new Preset(name, PRESETS(), {}, seed);
const raw = JSON.parse(JSON.stringify(preset.settings));
console.log(JSON.stringify(snakeKeys(raw)));
