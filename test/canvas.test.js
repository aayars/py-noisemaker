import assert from 'assert';
import { render } from '../src/composer.js';
import { Context } from '../src/context.js';
import { ColorSpace, ValueDistribution } from '../src/constants.js';

const fakeCtx2d = {
  img: null,
  createImageData(w, h) {
    this.img = { data: new Uint8ClampedArray(w * h * 4) };
    return this.img;
  },
  putImageData(img, x, y) {
    this.img = img;
  }
};

const fakeCanvas = {
  width: 0,
  height: 0,
  getContext(type) {
    if (type === 'webgl2') return null;
    if (type === '2d') return fakeCtx2d;
    return null;
  }
};

const ctx = new Context(fakeCanvas);

// grayscale sanity check
const presets = {
  gray: {
    settings: () => ({ color_space: ColorSpace.grayscale })
  }
};
render('gray', 0, { ctx, width: 2, height: 2, presets });
let data = fakeCtx2d.img.data;
for (let i = 0; i < data.length; i += 4) {
  assert.strictEqual(data[i], data[i + 1]);
  assert.strictEqual(data[i], data[i + 2]);
}

// column gradient: ensure X direction is left-to-right
render('column', 0, {
  ctx,
  width: 3,
  height: 2,
  presets: {
    column: {
      generator: () => ({
        color_space: ColorSpace.grayscale,
        distrib: ValueDistribution.column_index,
      }),
    },
  },
});
data = fakeCtx2d.img.data;
const colPx = (x, y) => data[(y * 3 + x) * 4];
assert.strictEqual(colPx(0, 0), 0);
assert.strictEqual(colPx(1, 0), 128);
assert.strictEqual(colPx(2, 0), 255);
assert.strictEqual(colPx(0, 1), 0);
assert.strictEqual(colPx(1, 1), 128);
assert.strictEqual(colPx(2, 1), 255);

// row gradient: ensure Y direction is top-to-bottom
render('row', 0, {
  ctx,
  width: 2,
  height: 3,
  presets: {
    row: {
      generator: () => ({
        color_space: ColorSpace.grayscale,
        distrib: ValueDistribution.row_index,
      }),
    },
  },
});
data = fakeCtx2d.img.data;
const rowPx = (x, y) => data[(y * 2 + x) * 4];
assert.strictEqual(rowPx(0, 0), 0);
assert.strictEqual(rowPx(1, 0), 0);
assert.strictEqual(rowPx(0, 1), 128);
assert.strictEqual(rowPx(1, 1), 128);
assert.strictEqual(rowPx(0, 2), 255);
assert.strictEqual(rowPx(1, 2), 255);

console.log('canvas tests passed');
