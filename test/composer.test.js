import assert from 'assert';
import fs from 'fs';
import { register } from '../js/noisemaker/effectsRegistry.js';
import { Preset, Effect } from '../js/noisemaker/composer.js';
import { Context } from '../js/noisemaker/context.js';
import { multires } from '../js/noisemaker/generators.js';
import { ColorSpace } from '../js/noisemaker/constants.js';
import { PRESETS as JS_PRESETS } from '../js/noisemaker/presets.js';
import { setSeed } from '../js/noisemaker/rng.js';

const DEBUG = false; // Set true to diagnose shader issues.

const log = [];
function record(tensor, shape, time, speed, label = '') {
  log.push(label);
  return tensor;
}
register('record', record, { label: '' });

// define presets graph
const PRESETS = {
  unique: {
    unique: true,
    post: () => [Effect('record', { label: 'unique' })]
  },
  base: {
    settings: () => ({ freq: 1 }),
    generator: (settings) => ({ freq: settings.freq }),
    octaves: () => [Effect('record', { label: 'base-oct' })],
    post: () => [Effect('record', { label: 'base-post' })]
  },
  grand: {
    post: () => [Effect('record', { label: 'grand-post' })],
    final: () => [Effect('record', { label: 'grand-final' })]
  },
  child: {
    layers: ['base', 'unique', 'unique'], // unique should only apply once
    settings: () => ({ freq: 2 }),
    generator: (settings) => ({ freq: settings.freq }),
    octaves: () => [Effect('record', { label: 'child-oct' })],
    post: () => [Effect('record', { label: 'child-post' }), new Preset('grand', PRESETS)],
    final: () => [Effect('record', { label: 'child-final' })]
  }
};

const preset = new Preset('child', PRESETS, { freq: 3 });
assert.strictEqual(preset.settings.freq, 3);

const ctx = new Context(null, DEBUG);
await preset.render(0, { ctx, width: 8, height: 8 });

assert.deepStrictEqual(log, [
  'base-oct',
  'child-oct',
  'base-post',
  'unique',
  'child-post',
  'grand-post',
  'grand-final',
  'child-final'
]);

assert.throws(() => Effect('missing'));
assert.throws(() => Effect('record', { bad: 1 }));

// colour-space test
let captured = null;
async function capture(tensor, shape, time, speed) {
  captured = Array.from(await tensor.read());
  return tensor;
}
register('capture', capture, {});

const CS_PRESETS = {
  cs: {
    settings: () => ({ color_space: ColorSpace.hsv }),
    generator: (settings) => ({ color_space: settings.colorSpace }),
    post: () => [Effect('capture')],
  },
};

const csPreset = new Preset('cs', CS_PRESETS);
const seed = 42;
const shape = [1, 1, 3];
const expectedTensor = await multires(1, shape, { seed, color_space: ColorSpace.hsv });
const expected = Array.from(await expectedTensor.read());
await csPreset.render(seed, { width: 1, height: 1 });
assert.deepStrictEqual(
  captured.map((v) => +v.toFixed(6)),
  expected.map((v) => +v.toFixed(6))
);

// snake_case colour-space
captured = null;
const CS_PRESETS_SNAKE = {
  cs: {
    settings: () => ({ color_space: ColorSpace.hsv }),
    generator: (settings) => ({ color_space: settings.colorSpace }),
    post: () => [Effect('capture')],
  },
};
const csPresetSnake = new Preset('cs', CS_PRESETS_SNAKE);
await csPresetSnake.render(seed, { width: 1, height: 1 });
assert.deepStrictEqual(
  captured.map((v) => +v.toFixed(6)),
  expected.map((v) => +v.toFixed(6))
);

// settings-only generator options should pass through
captured = null;
const SETTINGS_ONLY = {
  only: {
    settings: () => ({
      freq: 2,
      color_space: ColorSpace.hsv,
      hue_rotation: 0.5,
      hue_range: 0,
    }),
    post: (settings) => {
      // access settings keys to satisfy SettingsDict
      settings.freq;
      settings.color_space;
      settings.hue_rotation;
      settings.hue_range;
      return [Effect('capture')];
    },
  },
};
const presetOnly = new Preset('only', SETTINGS_ONLY);
const expectedOnlyTensor = await multires(2, shape, {
  seed,
  color_space: ColorSpace.hsv,
  hueRotation: 0.5,
  hueRange: 0,
});
const expectedOnly = Array.from(await expectedOnlyTensor.read());
await presetOnly.render(seed, { width: 1, height: 1 });
assert.deepStrictEqual(
  captured.map((v) => +v.toFixed(6)),
  expectedOnly.map((v) => +v.toFixed(6))
);

// unused settings should throw
assert.throws(() => new Preset('bad', { bad: { settings: () => ({ unused: 1 }) } }));

// keys in UNUSED_OKAY should be ignored
assert.doesNotThrow(
  () => new Preset('ok', { ok: { settings: () => ({ palette_alpha: 0.5 }) } })
);

// parity execution graph
const fixture = JSON.parse(
  fs.readFileSync(new URL('./fixtures/composer.json', import.meta.url))
);
for (const name of Object.keys(fixture)) {
  setSeed(1);
  const preset = new Preset(name, JS_PRESETS());
  await preset.render(1, { width: 8, height: 8, debug: true });
}

console.log('composer tests passed');
