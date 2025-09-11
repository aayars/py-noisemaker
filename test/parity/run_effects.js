import { setSeed } from '../../src/rng.js';
import { setSeed as setValueSeed } from '../../src/value.js';
import { basic } from '../../src/generators.js';
import * as effects from '../../src/effects.js';

const [,, name, seedStr] = process.argv;
const seed = parseInt(seedStr, 10);

setSeed(seed);
setValueSeed(seed);
const base = basic(2, [128, 128, 3], { hueRotation: 0 });

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
  warp: effects.warp,
  ripple: effects.ripple,
  wobble: effects.wobble,
  reverb: effects.reverb,
  light_leak: effects.lightLeak,
  crt: effects.crt,
  reindex: effects.reindex,
};

const fn = EFFECTS[name];
if (!fn) {
  throw new Error(`Unknown effect ${name}`);
}
const tensor = fn(base, [128, 128, 3], 0, 1);
const arr = tensor.read();
const buf = Buffer.from(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
console.log(buf.toString('base64'));
