import { ValueMask } from './constants.js';
import { Tensor } from './tensor.js';
import { loadGlyphs } from './glyphs.js';

// Bitmap masks encoded as nested arrays or procedural functions
export const Masks = {
  // Static bitmaps
  [ValueMask.chess]: [
    [0, 1],
    [1, 0],
  ],
  [ValueMask.waffle]: [
    [0, 1],
    [1, 1],
  ],
  [ValueMask.square]: [
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
  ],
  [ValueMask.grid]: [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
  ],
  [ValueMask.h_bar]: [
    [1, 1],
    [0, 0],
  ],
  [ValueMask.v_bar]: [
    [1, 0],
    [1, 0],
  ],
  [ValueMask.h_hex]: [
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
  ],
  [ValueMask.v_hex]: [
    [0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
  ],

  // Procedural masks computed on demand
  [ValueMask.truchet_lines]: ({ x, y, shape }) => {
    const tile = 2;
    const ox = Math.floor(x / tile);
    const oy = Math.floor(y / tile);
    const orient = (ox + oy) % 2;
    const lx = x % tile;
    const ly = y % tile;
    if (orient === 0) {
      return lx === ly ? 1 : 0;
    }
    return lx + ly === tile - 1 ? 1 : 0;
  },

  [ValueMask.invaders_square]: (() => {
    const size = 8;
    const pattern = (() => {
      const half = Math.ceil(size / 2);
      const rows = Array.from({ length: size }, () => Array(size).fill(0));
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < half; x++) {
          const v = Math.random() > 0.5 ? 1 : 0;
          rows[y][x] = v;
          rows[y][size - 1 - x] = v;
        }
      }
      return rows;
    })();

    return ({ x, y }) => pattern[y % size][x % size];
  })(),
};

// Shapes for procedural masks
const ProceduralShapes = {
  [ValueMask.truchet_lines]: [2, 2, 1],
  [ValueMask.invaders_square]: [8, 8, 1],
};

export function maskShape(mask) {
  if (ProceduralShapes[mask]) return [...ProceduralShapes[mask]];
  const m = Masks[mask];
  const height = m.length;
  const width = m[0].length;
  const channels = Array.isArray(m[0][0]) ? m[0][0].length : 1;
  return [height, width, channels];
}

export function getAtlas(mask) {
  if (mask === ValueMask.truetype) {
    return loadGlyphs([15, 15, 1]);
  }
  return null;
}

export function maskValues(mask, glyphShape = null, opts = {}) {
  const { atlas = null, inverse = false } = opts;
  const shape = maskShape(mask);
  if (!glyphShape) glyphShape = [...shape];
  if (shape.length === 3) glyphShape[2] = shape[2];

  const [h, w, c] = glyphShape;
  const data = new Float32Array(h * w * c);
  const fn = Masks[mask];

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let pixel;
      if (typeof fn === 'function') {
        pixel = fn({ x, y, shape });
      } else {
        pixel = fn[y % shape[0]][x % shape[1]];
      }
      if (!Array.isArray(pixel)) pixel = [pixel];
      if (inverse) pixel = pixel.map((v) => 1 - v);
      for (let k = 0; k < c; k++) {
        data[(y * w + x) * c + k] = pixel[k % pixel.length];
      }
    }
  }

  return [Tensor.fromArray(null, data, glyphShape), atlas];
}
