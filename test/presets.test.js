import assert from 'assert';
import { PRESETS, setSeed } from '../src/presets.js';
import { Preset } from '../src/composer.js';
import { Context } from '../src/context.js';

assert.strictEqual(typeof PRESETS, 'function');

setSeed(123);
const presets = PRESETS();
assert('basic' in presets);
assert('warp-shadow' in presets);

const settings1 = presets.basic.settings();
setSeed(123);
const settings2 = PRESETS().basic.settings();
assert.deepStrictEqual(settings1, settings2);

const ctx = new Context(null);
const basic = new Preset('basic', presets);
basic.render(0, { ctx, width: 1, height: 1 });

const warp = new Preset('warp-shadow', presets);
warp.render(0, { ctx, width: 1, height: 1 });

console.log('presets tests passed');
