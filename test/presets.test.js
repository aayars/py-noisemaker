import assert from 'assert';
import { PRESETS, setSeed } from '../src/presets.js';
import { Preset } from '../src/composer.js';
import { Context } from '../src/context.js';

assert.strictEqual(typeof PRESETS, 'function');

setSeed(123);
const presets = PRESETS();
assert('basic' in presets);
assert('warp-shadow' in presets);
assert('aberration' in presets);
assert('acid' in presets);
assert('grain' in presets);
assert('vignette-bright' in presets);
assert('vignette-dark' in presets);

const settings1 = presets.basic.settings();
setSeed(123);
const settings2 = PRESETS().basic.settings();
assert.deepStrictEqual(settings1, settings2);

setSeed(456);
const acid1 = presets.acid.settings();
setSeed(456);
const acid2 = PRESETS().acid.settings();
assert.deepStrictEqual(acid1, acid2);

setSeed(789);
const aberr1 = presets.aberration.settings();
setSeed(789);
const aberr2 = PRESETS().aberration.settings();
assert.deepStrictEqual(aberr1, aberr2);

setSeed(321);
const grain1 = presets.grain.settings();
setSeed(321);
const grain2 = PRESETS().grain.settings();
assert.deepStrictEqual(grain1, grain2);

setSeed(654);
const vb1 = presets['vignette-bright'].settings();
setSeed(654);
const vb2 = PRESETS()['vignette-bright'].settings();
assert.deepStrictEqual(vb1, vb2);

setSeed(987);
const vd1 = presets['vignette-dark'].settings();
setSeed(987);
const vd2 = PRESETS()['vignette-dark'].settings();
assert.deepStrictEqual(vd1, vd2);

const ctx = new Context(null);
const basic = new Preset('basic', presets);
basic.render(0, { ctx, width: 1, height: 1 });

const warp = new Preset('warp-shadow', presets);
warp.render(0, { ctx, width: 1, height: 1 });

const acidPreset = new Preset('acid', presets);
acidPreset.render(0, { ctx, width: 1, height: 1 });

const aberrPreset = new Preset('aberration', presets);
aberrPreset.render(0, { ctx, width: 1, height: 1 });

const grainPreset = new Preset('grain', presets);
grainPreset.render(0, { ctx, width: 1, height: 1 });

const vbPreset = new Preset('vignette-bright', presets);
vbPreset.render(0, { ctx, width: 1, height: 1 });

const vdPreset = new Preset('vignette-dark', presets);
vdPreset.render(0, { ctx, width: 1, height: 1 });

console.log('presets tests passed');
