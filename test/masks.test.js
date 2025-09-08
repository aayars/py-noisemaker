import assert from 'assert';
import { ValueMask } from '../src/constants.js';
import { maskValues, maskShape } from '../src/masks.js';
import { loadGlyphs } from '../src/glyphs.js';

// Static bitmap mask
let shape = maskShape(ValueMask.chess);
let [tensor] = maskValues(ValueMask.chess, shape);
let data = Array.from(tensor.read());
assert.deepStrictEqual(shape, [2, 2, 1]);
assert.deepStrictEqual(data, [0, 1, 1, 0]);

// Procedural mask
const procShape = [4, 4, 1];
[tensor] = maskValues(ValueMask.truchet_lines, procShape);
const vals = Array.from(tensor.read());
assert.strictEqual(vals.length, 16);
assert(vals.some(v => v === 1));
assert(vals.some(v => v === 0));

// Glyph atlas caching
const atlas1 = loadGlyphs([2, 2, 1]);
const atlas2 = loadGlyphs([2, 2, 1]);
assert.strictEqual(atlas1, atlas2);

console.log('masks ok');
