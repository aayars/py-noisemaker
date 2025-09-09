import { PRESETS } from '../src/presets.js';
import { Preset } from '../src/composer.js';
import { Context } from '../src/context.js';

const ctx = new Context(null);
const presets = PRESETS();
const problems = [];
const SKIP = ['low-poly-regions'];

for (const name of Object.keys(presets)) {
  if (SKIP.includes(name)) continue;
  try {
    const preset = new Preset(name, presets);
    preset.render(0, { ctx, width: 4, height: 4 });
  } catch (e) {
    problems.push(`${name} failed: ${e.message}`);
  }
}

if (problems.length) {
  throw new Error('Some presets failed to render:\n' + problems.join('\n'));
}

console.log('preset render tests passed');
