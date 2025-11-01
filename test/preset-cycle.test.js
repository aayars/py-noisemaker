import assert from 'assert';
import { Preset } from '../js/noisemaker/composer.js';
import PRESETS from '../js/noisemaker/presets.js';

// Verify that simple direct cycles are detected
const presets = {
  a: { layers: ['b'] },
  b: { layers: ['a'] },
};
assert.throws(() => new Preset('a', presets), /Cycle detected/);

// Ensure that the real-world preset "basic-low-poly" does not introduce
// a hidden cycle and can be constructed and rendered.
const works = PRESETS(['basic-low-poly']);
const p = new Preset('basic-low-poly', works, {}, 0);
const tensor = await p.render(0, { width: 1, height: 1 });
assert.ok(tensor && tensor.shape, 'render should return a tensor with shape');

console.log('preset cycle test passed');
