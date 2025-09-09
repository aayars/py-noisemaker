import { Effect, Preset } from './composer.js';
import {
  ColorSpace as color,
  DistanceMetric as distance,
  InterpolationType as interp,
  OctaveBlending as blend,
  PointDistribution as point,
  ValueDistribution as distrib,
  ValueMask as mask,
  VoronoiDiagramType as voronoi,
  WormBehavior as worms,
  colorSpaceMembers,
  distanceMetricAbsoluteMembers,
  valueMaskProceduralMembers,
  valueMaskGridMembers,
} from './constants.js';
import { PALETTES } from './palettes.js';
import { maskShape } from './masks.js';
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
        colorSpace: randomMember(color),
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
        colorSpace: settings.colorSpace,
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
        colorSpace: color.rgb,
        freq: randomInt(10, 15),
        octaves: 8,
        reindex_range: 1.25 + random() * 1.25,
      }),
    },

    'acid-droplets': {
      layers: [
        'multires',
        'reflect-octaves',
        'density-map',
        'random-hue',
        'bloom',
        'shadow',
        'saturation',
      ],
      settings: () => ({
        freq: randomInt(8, 12),
        hue_range: 0,
        lattice_drift: 1.0,
        mask: mask.sparse,
        mask_static: true,
        palette_on: false,
        reflect_range: 7.5 + random() * 3.5,
      }),
    },

    'acid-grid': {
      layers: ['voronoi-refract', 'sobel', 'funhouse', 'bloom'],
      settings: () => ({
        dist_metric: distance.euclidean,
        lattice_drift: coinFlip(),
        voronoi_alpha: 0.333 + random() * 0.333,
        voronoi_diagram_type: voronoi.color_range,
        voronoi_point_distrib: randomMember([point.square, point.waffle, point.chess]),
        voronoi_point_freq: 4,
        voronoi_point_generations: 2,
        warp_range: 0.125 + random() * 0.0625,
      }),
    },

    'acid-wash': {
      layers: ['basic', 'funhouse', 'ridge', 'shadow', 'saturation'],
      settings: () => ({
        freq: randomInt(4, 6),
        hue_range: 1.0,
        ridges: true,
        warp_octaves: 8,
      }),
    },

    'activation-signal': {
      layers: ['value-mask', 'glitchin-out'],
      settings: () => ({
        colorSpace: randomMember(colorSpaceMembers()),
        freq: 4,
        mask: mask.white_bear,
        spline_order: interp.constant,
      }),
    },

    aesthetic: {
      layers: [
        'basic',
        'maybe-derivative-post',
        'spatter-post',
        'maybe-invert',
        'be-kind-rewind',
        'spatter-final',
      ],
      settings: () => ({
        corners: true,
        distrib: randomMember([distrib.column_index, distrib.ones, distrib.row_index]),
        freq: randomInt(3, 5) * 2,
        mask: mask.chess,
        spline_order: interp.constant,
      }),
    },

    'alien-terrain': {
      layers: [
        'multires-ridged',
        'invert',
        'voronoi',
        'derivative-octaves',
        'invert',
        'erosion-worms',
        'bloom',
        'shadow',
        'grain',
        'saturation',
      ],
      settings: () => ({
        grain_contrast: 1.5,
        deriv_alpha: 0.25 + random() * 0.125,
        dist_metric: distance.euclidean,
        erosion_worms_alpha: 0.05 + random() * 0.025,
        erosion_worms_density: randomInt(150, 200),
        erosion_worms_inverse: true,
        erosion_worms_xy_blend: 0.333 + random() * 0.16667,
        freq: randomInt(3, 5),
        hue_rotation: 0.875,
        hue_range: 0.25 + random() * 0.25,
        palette_on: false,
        voronoi_alpha: 0.5 + random() * 0.25,
        voronoi_diagram_type: voronoi.flow,
        voronoi_point_freq: 10,
        voronoi_point_distrib: point.random,
        voronoi_refract: 0.25 + random() * 0.125,
      }),
    },

    'alien-glyphs': {
      layers: ['entities', 'maybe-rotate', 'smoothstep-narrow', 'posterize', 'grain', 'saturation'],
      settings: () => ({
        corners: true,
        mask: randomMember([mask.arecibo_num, mask.arecibo_bignum, mask.arecibo_nucleotide]),
        mask_repeat: randomInt(6, 12),
        refract_range: 0.025 + random() * 0.0125,
        refract_signed_range: false,
        refract_y_from_offset: true,
        spline_order: randomMember([interp.linear, interp.cosine]),
      }),
    },

    'alien-transmission': {
      layers: ['analog-glitch', 'sobel', 'glitchin-out'],
      settings: () => ({
        mask: randomMember(valueMaskProceduralMembers),
      }),
    },

    'analog-glitch': {
      layers: ['value-mask'],
      settings: () => ({
        mask: randomMember([mask.alphanum_hex, mask.lcd, mask.fat_lcd]),
        mask_repeat: randomInt(20, 30),
      }),
      generator: (settings) => {
        const [h, w] = maskShape(settings.mask);
        return {
          freq: [
            Math.floor(h * 0.5 + h * settings.mask_repeat),
            Math.floor(w * 0.5 + w * settings.mask_repeat),
          ],
        };
      },
    },

    'arcade-carpet': {
      layers: ['multires-alpha', 'funhouse', 'posterize', 'nudge-hue', 'carpet', 'bloom', 'contrast-final'],
      settings: () => ({
        colorSpace: color.rgb,
        distrib: distrib.exp,
        hue_range: 1,
        mask: mask.sparser,
        mask_static: true,
        octaves: 2,
        palette_on: false,
        posterize_levels: 3,
        warp_freq: randomInt(25, 25),
        warp_range: 0.03 + random() * 0.015,
        warp_octaves: 1,
      }),
      generator: (settings) => ({
        freq: settings.warp_freq,
      }),
    },

    'are-you-human': {
      layers: [
        'multires',
        'value-mask',
        'funhouse',
        'density-map',
        'saturation',
        'maybe-invert',
        'aberration',
        'snow',
      ],
      settings: () => ({
        freq: 15,
        hue_range: random() * 0.25,
        hue_rotation: random(),
        mask: mask.truetype,
      }),
    },

    'band-together': {
      layers: ['basic', 'reindex-post', 'funhouse', 'shadow', 'normalize', 'grain'],
      settings: () => ({
        freq: randomInt(6, 12),
        reindex_range: randomInt(8, 12),
        warp_range: 0.333 + random() * 0.16667,
        warp_octaves: 8,
        warp_freq: randomInt(2, 3),
      }),
    },

    'basic-low-poly': {
      layers: ['basic', 'low-poly', 'grain', 'saturation'],
    },

    'basic-voronoi': {
      layers: ['basic', 'voronoi'],
      settings: () => ({
        voronoi_diagram_type: randomMember([
          voronoi.color_range,
          voronoi.color_regions,
          voronoi.range_regions,
          voronoi.color_flow,
        ]),
      }),
    },

    'basic-voronoi-refract': {
      layers: ['basic', 'voronoi'],
      settings: () => ({
        dist_metric: randomMember(distanceMetricAbsoluteMembers()),
        hue_range: 0.25 + random() * 0.5,
        voronoi_diagram_type: voronoi.range,
        voronoi_nth: 0,
        voronoi_refract: 1.0 + random() * 0.5,
      }),
    },

    'basic-water': {
      layers: ['multires', 'refract-octaves', 'reflect-octaves', 'ripple'],
      settings: () => ({
        colorSpace: color.hsv,
        distrib: distrib.uniform,
        freq: randomInt(7, 10),
        hue_range: 0.05 + random() * 0.05,
        hue_rotation: 0.5125 + random() * 0.025,
        lattice_drift: 1.0,
        octaves: 4,
        palette_on: false,
        reflect_range: 0.16667 + random() * 0.16667,
        refract_range: 0.25 + random() * 0.125,
        refract_y_from_offset: true,
        ripple_range: 0.005 + random() * 0.0025,
        ripple_kink: randomInt(2, 4),
        ripple_freq: randomInt(2, 4),
      }),
    },

    'be-kind-rewind': {
      final: () => [Effect('vhs'), Preset('crt')],
    },

    'benny-lava': {
      layers: ['basic', 'posterize', 'funhouse', 'distressed'],
      settings: () => ({
        distrib: distrib.column_index,
        posterize_levels: 1,
        warp_range: 1 + random() * 0.5,
      }),
    },

    berkeley: {
      layers: ['multires-ridged', 'reindex-octaves', 'sine-octaves', 'ridge', 'shadow', 'grain', 'saturation'],
      settings: () => ({
        freq: randomInt(12, 16),
        palette_on: false,
        reindex_range: 0.75 + random() * 0.25,
        sine_range: 2.0 + random() * 2.0,
      }),
    },

    'big-data-startup': {
      layers: ['glyphic'],
      settings: () => ({
        mask: mask.script,
        hue_rotation: random(),
        hue_range: 0.0625 + random() * 0.5,
        posterize_levels: randomInt(2, 4),
      }),
    },

    'bit-by-bit': {
      layers: ['value-mask', 'bloom', 'crt'],
      settings: () => ({
        mask: randomMember([mask.alphanum_binary, mask.alphanum_hex, mask.alphanum_numeric]),
        mask_repeat: randomInt(20, 40),
      }),
    },

    bitmask: {
      layers: ['multires-low', 'value-mask', 'bloom'],
      settings: () => ({
        mask: randomMember(valueMaskProceduralMembers),
        mask_repeat: randomInt(7, 15),
        ridges: true,
      }),
    },

    'blacklight-fantasy': {
      layers: ['voronoi', 'funhouse', 'posterize', 'sobel', 'invert', 'bloom', 'grain', 'nudge-hue', 'contrast-final'],
      settings: () => ({
        colorSpace: color.rgb,
        dist_metric: randomMember(distanceMetricAbsoluteMembers()),
        posterize_levels: 3,
        voronoi_refract: 0.5 + random() * 1.25,
        warp_octaves: randomInt(1, 4),
        warp_range: randomInt(0, 1) * random(),
      }),
    },

    bloom: {
      settings: () => ({
        bloom_alpha: 0.025 + random() * 0.0125,
      }),
      final: (settings) => [Effect('bloom', { alpha: settings.bloom_alpha })],
    },

    blotto: {
      layers: ['basic', 'random-hue', 'spatter-post', 'maybe-palette', 'maybe-invert'],
      settings: () => ({
        colorSpace: randomMember(color),
        distrib: distrib.ones,
        spatter_post_color: false,
      }),
    },

    branemelt: {
      layers: ['multires', 'sobel', 'erode-post', 'refract-post', 'glitchin-out'],
      settings: () => ({
        lattice_drift: 1,
        freq: randomInt(3, 5),
        erode_range: 1 + random() * 2,
        refract_range: 0.125 + random() * 0.125,
        speed: 0.025,
      }),
    },

    branewaves: {
      layers: ['value-mask', 'ripple', 'bloom'],
      settings: () => ({
        mask: randomMember(valueMaskGridMembers),
        mask_repeat: randomInt(5, 10),
        ridges: true,
        ripple_freq: 2,
        ripple_kink: 1.5 + random() * 2,
        ripple_range: 0.15 + random() * 0.15,
        spline_order: randomMember(enumRange(interp.linear, interp.bicubic)),
      }),
    },

    'brightness-post': {
      settings: () => ({
        brightness_post: 0.125 + random() * 0.0625,
      }),
      post: (settings) => [
        Effect('adjustBrightness', { amount: settings.brightness_post }),
      ],
    },

    'brightness-final': {
      settings: () => ({
        brightness_final: 0.125 + random() * 0.0625,
      }),
      final: (settings) => [
        Effect('adjustBrightness', { amount: settings.brightness_final }),
      ],
    },

    'bringing-hexy-back': {
      layers: ['voronoi', 'funhouse', 'maybe-invert', 'bloom'],
      settings: () => ({
        colorSpace: randomMember(color),
        dist_metric: distance.euclidean,
        hue_range: 0.25 + random() * 0.75,
        voronoi_alpha: 0.333 + random() * 0.333,
        voronoi_diagram_type: voronoi.range_regions,
        voronoi_nth: 0,
        voronoi_point_distrib: coinFlip() ? point.v_hex : point.h_hex,
        voronoi_point_freq: randomInt(4, 7) * 2,
        warp_range: 0.05 + random() * 0.25,
        warp_octaves: randomInt(1, 4),
      }),
      generator: (settings) => ({
        freq: settings.voronoi_point_freq,
      }),
    },

    broken: {
      layers: ['multires-low', 'reindex-octaves', 'posterize', 'glowing-edges', 'grain', 'saturation'],
      settings: () => ({
        colorSpace: color.rgb,
        freq: randomInt(3, 4),
        lattice_drift: 2,
        posterize_levels: 3,
        reindex_range: randomInt(3, 4),
        speed: 0.025,
      }),
    },

    'bubble-machine': {
      layers: ['basic', 'posterize', 'wormhole', 'reverb', 'outline', 'maybe-invert'],
      settings: () => ({
        corners: true,
        distrib: distrib.uniform,
        freq: randomInt(3, 6) * 2,
        mask: randomMember([mask.h_hex, mask.v_hex]),
        posterize_levels: randomInt(8, 16),
        reverb_iterations: randomInt(1, 3),
        reverb_octaves: randomInt(3, 5),
        spline_order: randomMember(enumRange(interp.linear, interp.bicubic)),
        wormhole_stride: 0.1 + random() * 0.05,
        wormhole_kink: 0.5 + random() * 4,
      }),
    },

    'bubble-multiverse': {
      layers: ['voronoi', 'refract-post', 'density-map', 'random-hue', 'bloom', 'shadow'],
      settings: () => ({
        dist_metric: distance.euclidean,
        refract_range: 0.125 + random() * 0.05,
        speed: 0.05,
        voronoi_alpha: 1.0,
        voronoi_diagram_type: voronoi.flow,
        voronoi_point_freq: 10,
        voronoi_refract: 0.625 + random() * 0.25,
      }),
    },

    carpet: {
      layers: ['basic', 'sobel', 'posterize', 'contrast-final'],
      settings: () => ({
        freq: randomInt(3, 5) * 2,
        posterize_levels: randomInt(2, 3),
        saturation: 0,
        spline_order: interp.constant,
      }),
    },

    celebrate: {
      layers: ['multires', 'sine-post', 'posterize', 'sobel', 'bloom'],
      settings: () => ({
        posterize_levels: randomInt(3, 6),
        sine_range: randomInt(8, 12),
      }),
    },

    'cell-reflect': {
      layers: ['voronoi', 'sobel', 'reflect-post', 'bloom'],
      settings: () => ({
        voronoi_alpha: 0.5 + random() * 0.5,
        voronoi_diagram_type: voronoi.color_range,
        voronoi_nth: randomInt(1, 3),
        voronoi_point_distrib: point.random,
        voronoi_point_freq: randomInt(4, 6),
        reflect_range: 3 + random() * 3,
      }),
    },

    'cell-refract': {
      layers: ['voronoi', 'sobel', 'refract-post', 'bloom'],
      settings: () => ({
        voronoi_alpha: 0.5 + random() * 0.5,
        voronoi_diagram_type: voronoi.color_range,
        voronoi_nth: randomInt(1, 3),
        voronoi_point_distrib: point.random,
        voronoi_point_freq: randomInt(4, 6),
        refract_range: 0.333 + random() * 0.333,
      }),
    },

    'cell-refract-2': {
      layers: ['voronoi', 'sobel', 'refract-post', 'bloom', 'density-map'],
      settings: () => ({
        voronoi_alpha: 0.5 + random() * 0.5,
        voronoi_diagram_type: voronoi.color_regions,
        voronoi_nth: 0,
        voronoi_point_distrib: point.random,
        voronoi_point_freq: randomInt(4, 6),
        refract_range: 0.333 + random() * 0.333,
      }),
    },

    'cell-worms': {
      layers: ['voronoi', 'worms-post', 'bloom'],
      settings: () => ({
        voronoi_alpha: 0.5 + random() * 0.5,
        voronoi_diagram_type: voronoi.range,
        voronoi_nth: randomInt(1, 2),
        voronoi_point_distrib: point.random,
        voronoi_point_freq: randomInt(4, 6),
        worm_behavior: worms.cubic,
        worm_density: 0.25 + random() * 0.25,
      }),
    },

    chalky: {
      layers: ['multires', 'posterize', 'sobel', 'contrast-final'],
      settings: () => ({
        posterize_levels: randomInt(1, 2),
        saturation: 0,
      }),
    },

    'chunky-knit': {
      layers: ['multires', 'sobel', 'contrast-final'],
      settings: () => ({
        freq: randomInt(3, 5),
        lattice_drift: 1,
        octaves: randomInt(4, 5),
      }),
    },

    'classic-desktop': {
      layers: ['multires', 'posterize', 'sobel', 'bloom'],
      settings: () => ({
        hue_range: 0.5 + random() * 0.5,
        posterize_levels: randomInt(3, 5),
      }),
    },

    cloudburst: {
      layers: ['multires', 'sobel', 'refract-post', 'bloom', 'density-map'],
      settings: () => ({
        dist_metric: distance.euclidean,
        hue_range: 0.5 + random() * 0.5,
        refract_range: 0.333 + random() * 0.333,
      }),
    },

    clouds: {
      layers: ['multires', 'sobel', 'refract-post', 'bloom'],
      settings: () => ({
        dist_metric: distance.euclidean,
        hue_range: 0.5 + random() * 0.5,
        refract_range: 0.333 + random() * 0.333,
      }),
    },

    concentric: {
      layers: ['multires', 'sobel', 'contrast-final'],
      settings: () => ({
        dist_metric: distance.euclidean,
        freq: randomInt(3, 5),
        octaves: randomInt(2, 3),
        spline_order: interp.linear,
        voronoi_diagram_type: voronoi.concentric,
        voronoi_point_distrib: point.concentric,
        voronoi_point_freq: randomInt(3, 5),
      }),
    },

    conference: {
      layers: ['multires', 'sobel', 'bloom', 'posterize'],
      settings: () => ({
        posterize_levels: randomInt(2, 4),
        saturation: 0,
        speed: 0.1,
      }),
    },

    'contrast-post': {
      settings: () => ({
        contrast_amount: 1.25 + random() * 0.25,
      }),
      post: (settings) => [
        Effect('adjustContrast', { amount: settings.contrast_amount }),
      ],
    },

    'contrast-final': {
      settings: () => ({
        contrast_amount: 1.25 + random() * 0.25,
      }),
      final: (settings) => [
        Effect('adjustContrast', { amount: settings.contrast_amount }),
      ],
    },

    'cool-water': {
      layers: ['multires', 'sobel', 'refract-post', 'bloom'],
      settings: () => ({
        hue_range: 0.333 + random() * 0.333,
        refract_range: 0.333 + random() * 0.333,
      }),
    },

    'corner-case': {
      layers: ['multires', 'sobel', 'contrast-final'],
      settings: () => ({
        corners: true,
        dist_metric: distance.euclidean,
        mask: mask.chess,
        mask_static: true,
      }),
    },

    corduroy: {
      layers: ['multires', 'sobel', 'contrast-final'],
      settings: () => ({
        dist_metric: distance.euclidean,
        freq: [randomInt(2, 4), randomInt(8, 12)],
        octaves: randomInt(3, 4),
      }),
    },

    'cosmic-thread': {
      layers: ['multires', 'sobel', 'refract-post', 'gloaming'],
      settings: () => ({
        lattice_drift: 1,
        freq: randomInt(3, 5),
        refract_range: 0.125 + random() * 0.125,
        speed: 0.05,
      }),
    },

    cobblestone: {
      layers: ['voronoi', 'sobel', 'posterize', 'bloom'],
      settings: () => ({
        voronoi_alpha: 0.5 + random() * 0.5,
        voronoi_diagram_type: voronoi.color_regions,
        voronoi_point_distrib: point.hexagon,
        voronoi_point_freq: randomInt(3, 5),
        posterize_levels: randomInt(2, 4),
      }),
    },

    'convolution-feedback': {
      layers: ['basic', 'conv-feedback', 'grain'],
      settings: () => ({
        conv_feedback_iterations: randomInt(25, 50),
        conv_feedback_alpha: 0.25 + random() * 0.25,
      }),
      post: (settings) => [
        Effect('convFeedback', {
          iterations: settings.conv_feedback_iterations,
          alpha: settings.conv_feedback_alpha,
        }),
      ],
    },

    corrupt: {
      layers: ['basic', 'corrupt-post', 'grain'],
    },

    'crime-scene': {
      layers: ['value-mask', 'color-mask', 'posterize', 'sobel', 'shadow', 'vignette-dark'],
      settings: () => ({
        mask: mask.chalk_outline,
        mask_static: true,
        posterize_levels: randomInt(1, 2),
      }),
    },

    crooked: {
      layers: ['multires', 'sobel', 'contrast-final'],
      settings: () => ({
        dist_metric: distance.manhattan,
        freq: randomInt(3, 5),
        octaves: randomInt(2, 3),
      }),
    },

    crt: {
      final: () => [Effect('crt')],
    },

    crystallize: {
      layers: ['basic', 'voronoi', 'posterize', 'sobel'],
      settings: () => ({
        voronoi_diagram_type: voronoi.triangle_color,
        voronoi_point_freq: randomInt(2, 4),
        posterize_levels: randomInt(3, 5),
      }),
    },

    cubert: {
      layers: ['voronoi', 'sobel', 'maybe-rotate'],
      settings: () => ({
        voronoi_diagram_type: voronoi.cube,
        voronoi_point_freq: randomInt(3, 5),
      }),
    },

    cubic: {
      layers: ['basic', 'octave-warp-octaves', 'bloom'],
      settings: () => ({
        octaves: randomInt(3, 5),
        warp_octaves: randomInt(1, 3),
        warp_range: 0.25 + random() * 0.25,
      }),
    },

    'cyclic-dilation': {
      layers: ['voronoi', 'reindex-post', 'saturation', 'grain'],
      settings: () => ({
        freq: randomInt(24, 48),
        hue_range: 0.25 + random() * 1.25,
        reindex_range: randomInt(4, 6),
        voronoi_diagram_type: voronoi.color_range,
        voronoi_point_corners: true,
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
