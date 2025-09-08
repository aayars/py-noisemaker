import { createRequire } from 'module';
const require = createRequire(import.meta.url);

let createCanvas = null;
try {
  ({ createCanvas } = require('canvas'));
} catch (e) {
  createCanvas = null;
}

const cache = new Map();

/**
 * Rasterise printable ASCII glyphs into monochrome textures sorted by brightness.
 * Glyph atlases are cached by shape for reuse.
 *
 * @param {number[]} shape [height, width, channels]
 * @returns {Array} atlas of glyph textures
 */
export function loadGlyphs(shape) {
  const key = shape.join('x');
  if (cache.has(key)) return cache.get(key);

  if (!createCanvas) {
    const empty = [];
    cache.set(key, empty);
    return empty;
  }

  const [h, w] = shape;
  const glyphs = [];

  for (let i = 32; i < 127; i++) {
    const canvas = createCanvas(w, h);
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#fff';
    ctx.textBaseline = 'top';
    ctx.font = `${h}px sans-serif`;
    ctx.fillText(String.fromCharCode(i), 0, 0);
    const data = ctx.getImageData(0, 0, w, h).data;
    const pixels = [];
    let total = 0;
    for (let y = 0; y < h; y++) {
      const row = [];
      for (let x = 0; x < w; x++) {
        const v = data[(y * w + x) * 4] / 255;
        row.push([v]);
        total += v;
      }
      pixels.push(row);
    }
    glyphs.push({ pixels, total });
  }

  glyphs.sort((a, b) => a.total - b.total);
  const atlas = glyphs.map((g) => g.pixels);
  cache.set(key, atlas);
  return atlas;
}
