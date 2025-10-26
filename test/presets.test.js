import assert from 'assert';
import { PRESETS, setSeed } from '../js/noisemaker/presets.js';
import { Preset } from '../js/noisemaker/composer.js';
import { Context } from '../js/noisemaker/context.js';

const DEBUG = false; // Set true to diagnose shader issues.

function sanitize(obj) {
  return JSON.parse(JSON.stringify(obj, (k, v) => (typeof v === 'function' ? null : v)));
}

assert.strictEqual(typeof PRESETS, 'function');

setSeed(123);
const presets = PRESETS();
assert('basic' in presets);
assert('1976' in presets);
assert('aberration' in presets);
assert('acid' in presets);
assert('grain' in presets);
assert('vignette-bright' in presets);
assert('vignette-dark' in presets);

setSeed(123);
const settings1 = PRESETS().basic.settings();
setSeed(123);
const settings2 = PRESETS().basic.settings();
assert.deepStrictEqual(sanitize(settings1), sanitize(settings2));
assert.ok('color_space' in settings1);

setSeed(456);
const acid1 = PRESETS().acid.settings();
setSeed(456);
const acid2 = PRESETS().acid.settings();
assert.deepStrictEqual(sanitize(acid1), sanitize(acid2));

setSeed(789);
const aberr1 = PRESETS().aberration.settings();
setSeed(789);
const aberr2 = PRESETS().aberration.settings();
assert.deepStrictEqual(sanitize(aberr1), sanitize(aberr2));

setSeed(321);
const grain1 = PRESETS().grain.settings();
setSeed(321);
const grain2 = PRESETS().grain.settings();
assert.deepStrictEqual(sanitize(grain1), sanitize(grain2));

setSeed(654);
const vb1 = PRESETS()['vignette-bright'].settings();
setSeed(654);
const vb2 = PRESETS()['vignette-bright'].settings();
assert.deepStrictEqual(sanitize(vb1), sanitize(vb2));

setSeed(987);
const vd1 = PRESETS()['vignette-dark'].settings();
setSeed(987);
const vd2 = PRESETS()['vignette-dark'].settings();
assert.deepStrictEqual(sanitize(vd1), sanitize(vd2));

const ctx = new Context(null, DEBUG);
const basic = new Preset('basic', presets);
await basic.render(0, { ctx, width: 1, height: 1 });

const preset1976 = new Preset('1976', presets);
await preset1976.render(0, { ctx, width: 1, height: 1 });

const acidPreset = new Preset('acid', presets);
await acidPreset.render(0, { ctx, width: 1, height: 1 });

const aberrPreset = new Preset('aberration', presets);
await aberrPreset.render(0, { ctx, width: 1, height: 1 });

const grainPreset = new Preset('grain', presets);
await grainPreset.render(0, { ctx, width: 1, height: 1 });

const vbPreset = new Preset('vignette-bright', presets);
await vbPreset.render(0, { ctx, width: 1, height: 1 });

const vdPreset = new Preset('vignette-dark', presets);
await vdPreset.render(0, { ctx, width: 1, height: 1 });

console.log('presets tests passed');
