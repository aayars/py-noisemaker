import assert from 'assert';
import { register } from '../src/effectsRegistry.js';
import { Preset, Effect } from '../src/composer.js';
import { Context } from '../src/context.js';
import { multires } from '../src/generators.js';
import { ColorSpace } from '../src/constants.js';

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
    settings: () => ({ gain: 1 }),
    octaves: () => [Effect('record', { label: 'base-oct' })],
    post: () => [Effect('record', { label: 'base-post' })]
  },
  grand: {
    post: () => [Effect('record', { label: 'grand-post' })],
    final: () => [Effect('record', { label: 'grand-final' })]
  },
  child: {
    layers: ['base', 'unique', 'unique'], // unique should only apply once
    settings: () => ({ gain: 2 }),
    octaves: () => [Effect('record', { label: 'child-oct' })],
    post: () => [Effect('record', { label: 'child-post' }), new Preset('grand', PRESETS)],
    final: () => [Effect('record', { label: 'child-final' })]
  }
};

const preset = new Preset('child', PRESETS, { gain: 3 });
assert.strictEqual(preset.settings.gain, 3);

const ctx = new Context(null);
preset.render(0, { ctx, width: 1, height: 1 });

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
function capture(tensor, shape, time, speed) {
  captured = Array.from(tensor.read());
  return tensor;
}
register('capture', capture, {});

const CS_PRESETS = {
  cs: {
    settings: () => ({ color_space: ColorSpace.hsv }),
    post: () => [Effect('capture')],
  },
};

const csPreset = new Preset('cs', CS_PRESETS);
const seed = 42;
const shape = [1, 1, 3];
const expected = Array.from(
  multires(1, shape, { seed, color_space: ColorSpace.hsv }).read()
);
csPreset.render(seed, { width: 1, height: 1 });
assert.deepStrictEqual(
  captured.map((v) => +v.toFixed(6)),
  expected.map((v) => +v.toFixed(6))
);

// snake_case colour-space
captured = null;
const CS_PRESETS_SNAKE = {
  cs: {
    settings: () => ({ color_space: ColorSpace.hsv }),
    post: () => [Effect('capture')],
  },
};
const csPresetSnake = new Preset('cs', CS_PRESETS_SNAKE);
csPresetSnake.render(seed, { width: 1, height: 1 });
assert.deepStrictEqual(
  captured.map((v) => +v.toFixed(6)),
  expected.map((v) => +v.toFixed(6))
);

console.log('composer tests passed');
