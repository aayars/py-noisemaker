import assert from 'assert';
import { PALETTES, samplePalette } from '../js/noisemaker/palettes.js';

function arraysClose(a, b, eps = 1e-6) {
  assert.strictEqual(a.length, b.length);
  for (let i = 0; i < a.length; i++) {
    assert.ok(Math.abs(a[i] - b[i]) < eps, `index ${i}`);
  }
}

// ensure palettes are exported
assert.ok(PALETTES.grayscale);
assert.ok(PALETTES.rainbow);

// known sample values
arraysClose(samplePalette('grayscale', 0), [0.9619397662556434, 0.9619397662556434, 0.9619397662556434]);
arraysClose(samplePalette('rainbow', 0), [0.9619397662556434, 0.10978479633083504, 0.4451328444544776]);
arraysClose(samplePalette('rainbow', 0.5), [0, 0.7408768370508578, 0.740876837050858]);

assert.throws(() => samplePalette('bogus', 0));

console.log('Palette tests passed');
