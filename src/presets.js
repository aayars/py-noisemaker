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
      layers: ['multires', 'refract-octaves', 'sobel', 'density-map', 'lens'],
      settings: () => ({
        colorSpace: color.hsv,
        dist_metric: distance.euclidean,
        freq: randomInt(2, 4),
        hue_range: 0.35 + random() * 0.35,
        lsystem_bias: random(),
        octaves: randomInt(3, 5),
        palette_on: false,
        reflect_range: 3,
        refract_range: random() * 0.375,
        saturate: 0.75,
        spline_order: interp.linear,
        voronoi_alpha: 0.25 + random() * 0.125,
        voronoi_diagram_type: voronoi.flow,
        voronoi_freq: randomInt(2, 3),
      }),
    },

    'alien-glyphs': {
      layers: ['value-mask', 'gloaming', 'lens'],
      settings: () => ({
        colorSpace: color.hsv,
        freq: randomInt(3, 5) * 2,
        mask: randomMember([mask.letters, mask.script, mask.ideogram, mask.iching]),
        palette_on: false,
        spline_order: interp.constant,
      }),
    },

    'alien-transmission': {
      layers: ['basic', 'gloaming', 'lens'],
      settings: () => ({
        freq: randomInt(3, 5) * 2,
        mask: mask.matrix,
        palette_on: false,
        spline_order: interp.constant,
      }),
    },

    'analog-glitch': {
      layers: ['glitch', 'random-hue', 'bloom'],
      settings: () => ({
        glitch_alpha: 0.25 + random() * 0.25,
      }),
      post: (settings) => [Effect('glitch', { alpha: settings.glitch_alpha })],
    },

    'arcade-carpet': {
      layers: ['multires', 'posterize', 'sobel', 'bloom', 'vignette-dark'],
      settings: () => ({
        dist_metric: distance.euclidean,
        octaves: 4,
        posterize_levels: randomInt(2, 4),
        spline_order: interp.linear,
        saturation: 0,
        palette_name: randomMember(Object.keys(PALETTES)),
      }),
      post: (settings) => [Effect('palette', { name: settings.palette_name })],
    },

    'are-you-human': {
      layers: ['value-mask', 'sine-octaves', 'contrast-final'],
      settings: () => ({
        freq: randomInt(6, 8),
        mask: randomMember([
          mask.alphanum_a,
          mask.alphanum_b,
          mask.alphanum_c,
          mask.alphanum_d,
          mask.alphanum_e,
          mask.alphanum_f,
          mask.alphanum_0,
          mask.alphanum_1,
          mask.alphanum_2,
          mask.alphanum_3,
          mask.alphanum_4,
          mask.alphanum_5,
          mask.alphanum_6,
          mask.alphanum_7,
          mask.alphanum_8,
          mask.alphanum_9,
        ]),
        mask_inverse: true,
        mask_static: true,
        palette_on: false,
      }),
    },

    'band-together': {
      layers: ['multires', 'sobel', 'reindex-post', 'warp', 'grain'],
      settings: () => ({
        freq: randomInt(7, 9),
        lattice_drift: coinFlip(),
        mask: mask.chess,
        mask_static: true,
        reindex_range: 0.75 + random() * 0.75,
        warp_freq: 3,
        warp_displacement: 0.5 + random() * 0.5,
      }),
    },

    'basic-low-poly': {
      layers: ['basic'],
      post: () => [Effect('lowpoly')],
    },

    'basic-voronoi': {
      layers: ['basic'],
      post: () => [Effect('voronoi')],
    },

    'basic-voronoi-refract': {
      layers: ['basic'],
      post: () => [Effect('voronoi'), Effect('refract')],
    },

    'basic-water': {
      layers: ['basic', 'refract-post', 'invert'],
      settings: () => ({
        refract_range: 0.1 + random() * 0.1,
        speed: 0.05,
      }),
    },

    'be-kind-rewind': {
      final: () => [Effect('vhs')],
    },

    'benny-lava': {
      layers: ['voronoi', 'reindex-post', 'bloom', 'normalize'],
      settings: () => ({
        dist_metric: randomMember(distanceMetricAbsoluteMembers()),
        palette_on: false,
        reindex_range: 0.125 + random() * 0.125,
        voronoi_alpha: 0.5 + random() * 0.5,
        voronoi_diagram_type: voronoi.color_range,
        voronoi_point_freq: randomInt(3, 5),
      }),
    },

    berkeley: {
      layers: ['basic', 'reindex-post', 'pixel-sort', 'sine-post'],
      settings: () => ({
        freq: randomInt(4, 7),
        lattice_drift: coinFlip(),
        palette_on: false,
        pixel_sort_darkest: coinFlip(),
        pixel_sort_angled: coinFlip(),
        reindex_range: 0.333 + random() * 0.333,
        sine_range: randomInt(3, 5),
      }),
    },

    'big-data-startup': {
      layers: ['basic', 'sobel', 'posterize', 'bloom'],
      settings: () => ({
        dist_metric: distance.triangular,
        octaves: randomInt(4, 5),
        posterize_levels: randomInt(3, 4),
        saturation: 0,
      }),
    },

    'bit-by-bit': {
      layers: ['multires', 'density-map', 'posterize', 'sobel', 'contrast-final'],
      settings: () => ({
        freq: randomInt(5, 7),
        octaves: randomInt(3, 5),
        posterize_levels: randomInt(1, 2),
        saturation: 0,
      }),
    },

    bitmask: {
      layers: ['basic', 'value-mask', 'pixel-sort', 'contrast-final'],
      settings: () => ({
        freq: randomInt(4, 7),
        mask: randomMember([mask.alphanum_a, mask.alphanum_b, mask.alphanum_c]),
        mask_static: true,
        pixel_sort_darkest: coinFlip(),
        pixel_sort_angled: coinFlip(),
      }),
    },

    'blacklight-fantasy': {
      layers: ['multires', 'refract-octaves', 'random-hue', 'posterize', 'bloom'],
      settings: () => ({
        freq: randomInt(2, 3),
        hue_range: 1,
        octaves: randomInt(2, 3),
        posterize_levels: randomInt(2, 3),
        refract_range: 0.5 + random() * 0.5,
      }),
    },

    bloom: {
      layers: ['basic'],
      final: () => [Effect('bloom')],
    },

    blotto: {
      layers: ['multires', 'gloaming', 'light-leak'],
      settings: () => ({
        octaves: randomInt(3, 4),
        palette_on: false,
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
      layers: ['multires', 'sobel', 'erode-post', 'refract-post', 'gloaming'],
      settings: () => ({
        lattice_drift: 1,
        freq: randomInt(3, 5),
        erode_range: 1 + random() * 2,
        refract_range: 0.125 + random() * 0.125,
        speed: 0.05,
      }),
    },

    'brightness-post': {
      settings: () => ({
        brightness_amount: 0.25 + random() * 0.25,
      }),
      post: (settings) => [
        Effect('adjustBrightness', { amount: settings.brightness_amount }),
      ],
    },

    'brightness-final': {
      settings: () => ({
        brightness_amount: 0.25 + random() * 0.25,
      }),
      final: (settings) => [
        Effect('adjustBrightness', { amount: settings.brightness_amount }),
      ],
    },

    'bringing-hexy-back': {
      layers: ['multires', 'sobel', 'contrast-final', 'maybe-rotate'],
      settings: () => ({
        freq: randomInt(2, 3),
        octaves: randomInt(3, 5),
        voronoi_point_distrib: point.h_hex,
      }),
    },

    broken: {
      layers: ['multires', 'sobel', 'refract-post', 'invert', 'brightness-post', 'contrast-post', 'lens'],
      settings: () => ({
        dist_metric: distance.euclidean,
        octaves: randomInt(1, 3),
        refract_range: 0.5 + random() * 0.5,
      }),
    },

    'bubble-machine': {
      layers: ['multires', 'refract-octaves', 'posterize', 'sobel', 'bloom'],
      settings: () => ({
        dist_metric: distance.euclidean,
        hue_range: 0.75,
        posterize_levels: randomInt(2, 4),
        refract_range: 0.5 + random() * 0.5,
      }),
    },

    'bubble-multiverse': {
      layers: ['multires', 'refract-octaves', 'posterize', 'sobel', 'bloom', 'density-map'],
      settings: () => ({
        dist_metric: distance.euclidean,
        hue_range: 0.75,
        posterize_levels: randomInt(2, 4),
        refract_range: 0.5 + random() * 0.5,
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
