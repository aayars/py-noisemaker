import { choice } from './util.js';

let loadedFonts = null;

/**
 * Discover available fonts in the current environment. In browsers we use the
 * `document.fonts` API and fall back to a default font when unavailable.
 *
 * @returns {Array<string>} Array of loaded font family names
 */
export function loadFonts() {
  if (loadedFonts) return loadedFonts;

  loadedFonts = [];

  if (typeof document !== 'undefined' && document.fonts) {
    document.fonts.forEach((f) => loadedFonts.push(f.family));
  }

  if (!loadedFonts.length) {
    loadedFonts.push('sans-serif');
  }

  return loadedFonts;
}

const cache = new Map();

/**
 * Rasterise printable ASCII glyphs into monochrome textures sorted by
 * brightness. Glyph atlases are cached by shape for reuse.
 *
 * @param {number[]} shape [height, width, channels]
 * @returns {Array} atlas of glyph textures
 */
export function loadGlyphs(shape) {
  const key = shape.join('x');
  if (cache.has(key)) return cache.get(key);

  if (typeof document === 'undefined') {
    const empty = [];
    cache.set(key, empty);
    return empty;
  }

  const fonts = loadFonts();
  if (!fonts.length) {
    const empty = [];
    cache.set(key, empty);
    return empty;
  }

  const [h, w] = shape;
  const glyphs = [];
  const fontFamily = choice(fonts);

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');

  for (let i = 32; i < 127; i++) {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#fff';
    ctx.textBaseline = 'top';
    ctx.font = `${h}px '${fontFamily}'`;
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

