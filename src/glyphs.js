import { createRequire } from 'module';
import { choice } from './util.js';

const require = createRequire(import.meta.url);

let createCanvas = null;
let registerFont = null;
try {
  ({ createCanvas, registerFont } = require('canvas'));
} catch (e) {
  createCanvas = null;
}

let loadedFonts = null;

/**
 * Read TrueType fonts from the user's Noisemaker directory and register them
 * with node-canvas. The directory can be overridden by setting the
 * `NOISEMAKER_DIR` environment variable.
 *
 * @returns {Array<string>} Array of loaded font family names
 */
export function loadFonts() {
  if (loadedFonts) return loadedFonts;

  if (!registerFont) {
    loadedFonts = [];
    return loadedFonts;
  }

  const fs = require('fs');
  const path = require('path');
  const os = require('os');

  let dir = process.env.NOISEMAKER_DIR
    ? path.join(process.env.NOISEMAKER_DIR, 'fonts')
    : path.join(os.homedir(), '.noisemaker', 'fonts');

  loadedFonts = [];

  try {
    const files = fs.readdirSync(dir).filter((f) => f.toLowerCase().endsWith('.ttf'));
    for (const file of files) {
      const family = path.basename(file, path.extname(file));
      try {
        registerFont(path.join(dir, file), { family });
        loadedFonts.push(family);
      } catch (e) {
        // Ignore invalid fonts
      }
    }
  } catch (e) {
    // Directory doesn't exist or can't be read
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

  if (!createCanvas) {
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

  for (let i = 32; i < 127; i++) {
    const canvas = createCanvas(w, h);
    const ctx = canvas.getContext('2d');
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

