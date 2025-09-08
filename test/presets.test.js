import assert from 'assert';
import { PRESETS } from '../src/presets.js';
import { Preset } from '../src/composer.js';
import { Context } from '../src/context.js';

assert.strictEqual(typeof PRESETS, 'function');

const presets = PRESETS();
assert('basic' in presets);
assert('warp-shadow' in presets);

const ctx = new Context(null);
const basic = new Preset('basic', presets);
basic.render(0, { ctx, width: 1, height: 1 });

const warp = new Preset('warp-shadow', presets);
warp.render(0, { ctx, width: 1, height: 1 });

console.log('presets tests passed');
