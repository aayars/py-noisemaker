import assert from 'assert';
import { Preset } from '../src/composer.js';

const presets = {
  a: { layers: ['b'] },
  b: { layers: ['a'] },
};

assert.throws(() => new Preset('a', presets), /Cycle detected/);

console.log('preset cycle test passed');
