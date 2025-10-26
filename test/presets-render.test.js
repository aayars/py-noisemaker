import { PRESETS } from '../js/noisemaker/presets.js';
import { Preset } from '../js/noisemaker/composer.js';
import { Context } from '../js/noisemaker/context.js';

const DEBUG = false; // Set true to diagnose shader issues.

const ctx = new Context(null, DEBUG);
const presets = PRESETS();
const problems = [];
const SKIP = [];

for (const name of Object.keys(presets)) {
  if (SKIP.includes(name)) continue;
  try {
    const preset = new Preset(name, presets);
    await preset.render(0, { ctx, width: 4, height: 4 });
  } catch (e) {
    problems.push(`${name} failed: ${e.message}`);
  }
}

if (problems.length) {
  throw new Error('Some presets failed to render:\n' + problems.join('\n'));
}

console.log('preset render tests passed');
