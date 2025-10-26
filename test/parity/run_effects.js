import { setSeed } from '../../js/noisemaker/rng.js';
import { setSeed as setValueSeed, values as valueValues } from '../../js/noisemaker/value.js';
import { basic } from '../../js/noisemaker/generators.js';
import * as effects from '../../js/noisemaker/effects.js';

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
  color_map: async (tensor, shape, time, speed) => {
    const clut = await valueValues([4, 4], shape, { ctx: tensor.ctx, time, speed });
    return effects.colorMap(tensor, shape, time, speed, clut);
  },
  false_color: effects.falseColor,
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
  ridge: effects.ridgeEffect,
  sine: effects.sine,
  fxaa: effects.fxaaEffect,
  smoothstep: effects.smoothstep,
  sobel: effects.sobelOperator,
  lens_warp: effects.lensWarp,
  lens_distortion: effects.lensDistortion,
  vhs: effects.vhs,
  snow: effects.snow,
  reindex: effects.reindex,
  voronoi: effects.voronoi,
  rotate: (tensor, shape, time, speed) => effects.rotate(tensor, shape, time, speed),
  clouds: effects.clouds,
  conv_feedback: effects.convFeedback,
  convolve: effects.convolve,
  degauss: effects.degauss,
  density_map: effects.densityMap,
  dla: effects.dla,
  erosion_worms: effects.erosionWorms,
  fibers: effects.fibers,
  frame: effects.frame,
  glyph_map: effects.glyphMap,
  grime: effects.grime,
  jpeg_decimate: effects.jpegDecimate,
  kaleido: effects.kaleido,
  lowpoly: effects.lowpoly,
  nebula: effects.nebula,
  normal_map: effects.normalMap,
  on_screen_display: effects.onScreenDisplay,
  pixel_sort: effects.pixelSort,
  refract: effects.refractEffect,
  scratches: effects.scratches,
  simple_frame: effects.simpleFrame,
  sketch: effects.sketch,
  spatter: effects.spatter,
  spooky_ticker: effects.spookyTicker,
  stray_hair: effects.strayHair,
  texture: effects.texture,
  value_refract: effects.valueRefract,
  vortex: effects.vortex,
  wormhole: effects.wormhole,
  worms: effects.worms,
};

const fn = EFFECTS[name];
if (!fn) {
  throw new Error(`Unknown effect ${name}`);
}
const tensor = await fn(base, [128, 128, 3], 0, 1);
const arr = await tensor.read();
const buf = Buffer.from(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
console.log(buf.toString('base64'));
