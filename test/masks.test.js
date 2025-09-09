import assert from 'assert';
import { ValueMask } from '../src/constants.js';
import { maskValues, maskShape } from '../src/masks.js';
import { loadGlyphs, loadFonts } from '../src/glyphs.js';

// Static bitmap mask
let shape = maskShape(ValueMask.chess);
let [tensor] = maskValues(ValueMask.chess, shape);
let data = Array.from(tensor.read());
assert.deepStrictEqual(shape, [2, 2, 1]);
assert.deepStrictEqual(data, [0, 1, 1, 0]);

// Alphanum masks
shape = maskShape(ValueMask.alphanum_0);
[tensor] = maskValues(ValueMask.alphanum_0, shape);
data = Array.from(tensor.read());
assert.deepStrictEqual(shape, [6, 6, 1]);
assert.deepStrictEqual(data, [
  0, 0, 0, 0, 0, 0,
  0, 1, 1, 1, 0, 0,
  1, 0, 0, 1, 1, 0,
  1, 0, 1, 0, 1, 0,
  1, 1, 0, 0, 1, 0,
  0, 1, 1, 1, 0, 0,
]);

shape = maskShape(ValueMask.alphanum_1);
[tensor] = maskValues(ValueMask.alphanum_1, shape);
data = Array.from(tensor.read());
assert.deepStrictEqual(shape, [6, 6, 1]);
assert.deepStrictEqual(data, [
  0, 0, 0, 0, 0, 0,
  0, 0, 1, 0, 0, 0,
  0, 1, 1, 0, 0, 0,
  0, 0, 1, 0, 0, 0,
  0, 0, 1, 0, 0, 0,
  0, 1, 1, 1, 0, 0,
]);

shape = maskShape(ValueMask.alphanum_2);
[tensor] = maskValues(ValueMask.alphanum_2, shape);
data = Array.from(tensor.read());
  assert.deepStrictEqual(shape, [6, 6, 1]);
  assert.deepStrictEqual(data, [
    0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 0, 0,
    0, 0, 0, 0, 1, 0,
    0, 1, 1, 1, 0, 0,
    1, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 0,
  ]);

// Bank OCR masks
const bankOcrMasks = [
  [ValueMask.bank_ocr_0, [
    [0, 1, 1, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
  ]],
  [ValueMask.bank_ocr_1, [
    [0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
  ]],
  [ValueMask.bank_ocr_2, [
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
  ]],
];

for (const [mask, expected] of bankOcrMasks) {
  shape = maskShape(mask);
  [tensor] = maskValues(mask, shape);
  data = Array.from(tensor.read());
  assert.deepStrictEqual(shape, [8, 7, 1]);
  assert.deepStrictEqual(data, expected.flat());
}

// Truchet curve and tile masks
shape = maskShape(ValueMask.truchet_curves_00);
[tensor] = maskValues(ValueMask.truchet_curves_00, shape);
data = Array.from(tensor.read());
assert.deepStrictEqual(shape, [6, 6, 1]);
assert.deepStrictEqual(data, [
  0, 0, 0, 1, 0, 0,
  0, 0, 0, 1, 0, 0,
  0, 0, 0, 0, 1, 1,
  1, 1, 0, 0, 0, 0,
  0, 0, 1, 0, 0, 0,
  0, 0, 1, 0, 0, 0,
]);

shape = maskShape(ValueMask.truchet_tile_00);
[tensor] = maskValues(ValueMask.truchet_tile_00, shape);
data = Array.from(tensor.read());
assert.deepStrictEqual(shape, [6, 6, 1]);
assert.deepStrictEqual(data, [
  0, 0, 0, 0, 0, 1,
  0, 0, 0, 0, 1, 1,
  0, 0, 0, 1, 1, 1,
  0, 0, 1, 1, 1, 1,
  0, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1,
]);

// Halftone masks
const halftoneMasks = [
  [ValueMask.halftone_0, [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ]],
  [ValueMask.halftone_1, [
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ]],
  [ValueMask.halftone_2, [
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
  ]],
  [ValueMask.halftone_3, [
    [0, 0, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 0, 0],
    [0, 1, 0, 1],
  ]],
  [ValueMask.halftone_4, [
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
  ]],
  [ValueMask.halftone_5, [
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
  ]],
  [ValueMask.halftone_6, [
    [1, 1, 1, 1],
    [1, 0, 1, 0],
    [1, 1, 1, 1],
    [1, 0, 1, 0],
  ]],
  [ValueMask.halftone_7, [
    [1, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 0],
  ]],
  [ValueMask.halftone_8, [
    [1, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
  ]],
  [ValueMask.halftone_9, [
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
  ]],
];
for (const [mask, expected] of halftoneMasks) {
  shape = maskShape(mask);
  [tensor] = maskValues(mask, shape);
  data = Array.from(tensor.read());
  assert.deepStrictEqual(shape, [4, 4, 1]);
  assert.deepStrictEqual(data, expected.flat());
}

// LCD masks
const lcdMasks = [
  [ValueMask.lcd_0, [
    [0, 1, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
  ]],
  [ValueMask.lcd_1, [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
  ]],
  [ValueMask.lcd_2, [
    [0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
  ]],
  [ValueMask.lcd_3, [
    [0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
  ]],
  [ValueMask.lcd_4, [
    [0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
  ]],
  [ValueMask.lcd_5, [
    [0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
  ]],
  [ValueMask.lcd_6, [
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
  ]],
  [ValueMask.lcd_7, [
    [0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
  ]],
  [ValueMask.lcd_8, [
    [0, 1, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
  ]],
  [ValueMask.lcd_9, [
    [0, 1, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
  ]],
];
for (const [mask, expected] of lcdMasks) {
  shape = maskShape(mask);
  [tensor] = maskValues(mask, shape);
  data = Array.from(tensor.read());
  assert.deepStrictEqual(shape, [8, 5, 1]);
  assert.deepStrictEqual(data, expected.flat());
}

// Fat LCD masks
const fatLcdMasks = [
  [ValueMask.fat_lcd_0, [
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ]],
  [ValueMask.fat_lcd_1, [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ]],
  [ValueMask.fat_lcd_2, [
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ]],
];
for (const [mask, expected] of fatLcdMasks) {
  shape = maskShape(mask);
  [tensor] = maskValues(mask, shape);
  data = Array.from(tensor.read());
  assert.deepStrictEqual(shape, [10, 10, 1]);
  assert.deepStrictEqual(data, expected.flat());
}

// McPaint masks
shape = maskShape(ValueMask.mcpaint_00);
[tensor] = maskValues(ValueMask.mcpaint_00, shape);
data = Array.from(tensor.read());
assert.deepStrictEqual(shape, [8, 8, 1]);
assert.deepStrictEqual(data.slice(0, 8), Array(8).fill(0));
assert.deepStrictEqual(data.slice(-8), Array(8).fill(0));

// Emoji masks
for (const mask of [ValueMask.emoji_00, ValueMask.emoji_01, ValueMask.emoji_02]) {
  shape = maskShape(mask);
  [tensor] = maskValues(mask, shape);
  assert.deepStrictEqual(shape, [13, 13, 1]);
  assert.strictEqual(tensor.read().length, 13 * 13);
}

// Procedural mask
const procShape = [4, 4, 1];
[tensor] = maskValues(ValueMask.truchet_lines, procShape);
const vals = Array.from(tensor.read());
assert.strictEqual(vals.length, 16);
assert(vals.some(v => v === 1));
assert(vals.some(v => v === 0));

// Font loading cache
const fonts1 = loadFonts();
const fonts2 = loadFonts();
assert.strictEqual(fonts1, fonts2);

// Glyph atlas caching
const atlas1 = loadGlyphs([2, 2, 1]);
const atlas2 = loadGlyphs([2, 2, 1]);
assert.strictEqual(atlas1, atlas2);

console.log('masks ok');
