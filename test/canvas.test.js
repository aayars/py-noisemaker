import assert from 'assert';
import { render } from '../src/composer.js';
import { Context } from '../src/context.js';
import { ColorSpace } from '../src/constants.js';

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
const presets = {
  gray: {
    settings: () => ({ colorSpace: ColorSpace.grayscale })
  }
};

render('gray', 0, { ctx, width: 2, height: 2, presets });

const data = fakeCtx2d.img.data;
for (let i = 0; i < data.length; i += 4) {
  assert.strictEqual(data[i], data[i + 1]);
  assert.strictEqual(data[i], data[i + 2]);
}

console.log('canvas tests passed');
