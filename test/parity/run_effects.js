import { setSeed } from '../../src/rng.js';
import { setSeed as setValueSeed } from '../../src/value.js';
import { basic } from '../../src/generators.js';
import * as effects from '../../src/effects.js';

const [,, name, seedStr] = process.argv;
const seed = parseInt(seedStr, 10);

setSeed(seed);
setValueSeed(seed);
const base = await basic(2, [128, 128, 3], { hueRotation: 0 });

const EFFECTS = {
  adjust_hue: effects.adjustHueEffect,
  adjust_saturation: effects.saturation,
  adjust_brightness: effects.adjustBrightness,
  adjust_contrast: effects.adjustContrast,
  posterize: effects.posterize,
  blur: effects.blur,
  bloom: effects.bloom,
  vignette: effects.vignette,
  vaseline: effects.vaseline,
  shadow: effects.shadow,
  outline: effects.outline,
  glowing_edges: effects.glowingEdges,
  derivative: effects.derivative,
  normalize: effects.normalizeEffect,
  palette: effects.palette,
  warp: effects.warp,
  ripple: effects.ripple,
  wobble: effects.wobble,
  glitch: effects.glitch,
  reverb: effects.reverb,
  tint: effects.tint,
  aberration: effects.aberration,
  scanline_error: effects.scanlineError,
  light_leak: effects.lightLeak,
  crt: effects.crt,
  grain: effects.grain,
  lens_distortion: effects.lensDistortion,
  vhs: effects.vhs,
  snow: effects.snow,
  reindex: effects.reindex,
  voronoi: effects.voronoi,
  rotate: (tensor, shape, time, speed) => effects.rotate(tensor, shape, time, speed),
};

const fn = EFFECTS[name];
if (!fn) {
  throw new Error(`Unknown effect ${name}`);
}
const tensor = await fn(base, [128, 128, 3], 0, 1);
const arr = await tensor.read();
const buf = Buffer.from(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
console.log(buf.toString('base64'));
