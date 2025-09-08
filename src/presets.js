import { Effect } from './composer.js';
import {
  ColorSpace as color,
  ValueDistribution as distrib,
  InterpolationType as interp,
  OctaveBlending as blend,
} from './constants.js';
import { PALETTES } from './palettes.js';
import './effects.js';
import { random } from './util.js';
export { setSeed } from './util.js';

// Random helper utilities adapted from the Python implementation
function randomInt(min, max) {
  return Math.floor(random() * (max - min + 1)) + min;
}

export function coinFlip() {
  return random() < 0.5;
}

export function enumRange(a, b) {
  const out = [];
  for (let i = a; i <= b; i++) {
    out.push(i);
  }
  return out;
}

export function randomMember(...collections) {
  const out = [];
  for (const c of collections) {
    if (Array.isArray(c)) {
      out.push(...c.slice().sort());
    } else if (c && typeof c === 'object' && !(c instanceof Map)) {
      out.push(
        ...Object.keys(c)
          .sort()
          .map((k) => c[k])
      );
    } else if (c && typeof c[Symbol.iterator] === 'function') {
      const arr = Array.from(c);
      arr.sort();
      out.push(...arr);
    } else {
      throw new Error('randomMember(arg) should be iterable');
    }
  }
  const idx = Math.floor(random() * out.length);
  return out[idx];
}

const _STASH = new Map();
export function stash(key, value) {
  if (value !== undefined) {
    _STASH.set(key, value);
  }
  return _STASH.get(key);
}

export function PRESETS() {
  return {
    'maybe-palette': {
      settings: () => ({
        palette_alpha: 0.5 + random() * 0.5,
        palette_name: randomMember(Object.keys(PALETTES)),
        palette_on: coinFlip(),
      }),
      post: (settings) =>
        !settings.palette_on
          ? []
          : [Effect('palette', { name: settings.palette_name })],
    },

    basic: {
      unique: true,
      layers: ['maybe-palette'],
      settings: () => ({
        brightness_distrib: null,
        color_space: randomMember(color),
        corners: false,
        distrib: distrib.uniform,
        freq: [randomInt(2, 4), randomInt(2, 4)],
        hue_distrib: null,
        hue_range: random() * 0.25,
        hue_rotation: random(),
        lattice_drift: 0.0,
        mask: null,
        mask_inverse: false,
        mask_static: false,
        octave_blending: blend.falloff,
        octaves: 1,
        ridges: false,
        saturation: 1.0,
        saturation_distrib: null,
        sin: 0.0,
        spline_order: interp.bicubic,
      }),
      generator: (settings) => ({
        brightness_distrib: settings.brightness_distrib,
        color_space: settings.color_space,
        corners: settings.corners,
        distrib: settings.distrib,
        freq: settings.freq,
        hue_distrib: settings.hue_distrib,
        hue_range: settings.hue_range,
        hue_rotation: settings.hue_rotation,
        lattice_drift: settings.lattice_drift,
        mask: settings.mask,
        mask_inverse: settings.mask_inverse,
        mask_static: settings.mask_static,
        octave_blending: settings.octave_blending,
        octaves: settings.octaves,
        ridges: settings.ridges,
        saturation: settings.saturation,
        saturation_distrib: settings.saturation_distrib,
        sin: settings.sin,
        spline_order: settings.spline_order,
      }),
    },

    posterize: {
      layers: ['basic'],
      settings: () => ({
        posterize_levels: randomInt(3, 7),
      }),
      post: (settings) => [
        Effect('posterize', { levels: settings.posterize_levels }),
      ],
    },

    warp: {
      layers: ['basic'],
      settings: () => ({
        warp_freq: randomInt(2, 4),
        warp_octaves: randomInt(1, 3),
        warp_displacement: 0.5 + random() * 0.5,
      }),
      post: (settings) => [
        Effect('warp', {
          freq: settings.warp_freq,
          octaves: settings.warp_octaves,
          displacement: settings.warp_displacement,
        }),
      ],
    },

    'warp-shadow': {
      layers: ['basic'],
      settings: () => ({
        warp_freq: randomInt(2, 4),
        warp_octaves: randomInt(1, 3),
        warp_displacement: 0.5 + random() * 0.5,
        shadow_alpha: 0.5 + random() * 0.5,
      }),
      post: (settings) => [
        Effect('warp', {
          freq: settings.warp_freq,
          octaves: settings.warp_octaves,
          displacement: settings.warp_displacement,
        }),
      ],
      final: (settings) => [
        Effect('shadow', { alpha: settings.shadow_alpha }),
      ],
    },

    'reindex-post': {
      settings: () => ({
        reindex_range: 0.125 + random() * 2.5,
      }),
      post: (settings) => [
        Effect('reindex', { displacement: settings.reindex_range }),
      ],
    },

    normalize: {
      post: () => [Effect('normalize')],
    },

    aberration: {
      settings: () => ({
        aberration_displacement: 0.0125 + random() * 0.000625,
      }),
      final: (settings) => [
        Effect('aberration', { displacement: settings.aberration_displacement }),
      ],
    },

    acid: {
      layers: ['basic', 'reindex-post', 'normalize'],
      settings: () => ({
        color_space: color.rgb,
        freq: randomInt(10, 15),
        octaves: 8,
        reindex_range: 1.25 + random() * 1.25,
      }),
    },

    grain: {
      unique: true,
      settings: () => ({
        grain_alpha: 0.0333 + random() * 0.01666,
        grain_brightness: 0.0125 + random() * 0.00625,
        grain_contrast: 1.025 + random() * 0.0125,
      }),
      final: (settings) => [
        Effect('grain', { alpha: settings.grain_alpha }),
        Effect('adjustBrightness', { amount: settings.grain_brightness }),
        Effect('adjustContrast', { amount: settings.grain_contrast }),
      ],
    },

    'vignette-bright': {
      settings: () => ({
        vignette_bright_alpha: 0.333 + random() * 0.333,
        vignette_bright_brightness: 1.0,
      }),
      final: (settings) => [
        Effect('vignette', {
          alpha: settings.vignette_bright_alpha,
          brightness: settings.vignette_bright_brightness,
        }),
      ],
    },

    'vignette-dark': {
      settings: () => ({
        vignette_dark_alpha: 0.5 + random() * 0.25,
        vignette_dark_brightness: 0.0,
      }),
      final: (settings) => [
        Effect('vignette', {
          alpha: settings.vignette_dark_alpha,
          brightness: settings.vignette_dark_brightness,
        }),
      ],
    },
  };
}

export default PRESETS;
