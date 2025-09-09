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
  distanceMetricAll,
  valueMaskProceduralMembers,
  valueMaskGridMembers,
  valueMaskGlyphMembers,
  valueMaskNonproceduralMembers,
  circularMembers as pointCircularMembers,
  gridMembers as pointGridMembers,
  isCenterDistribution,
  isScan,
  isValueMaskRgb,
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

    'maybe-derivative-post': {
      post: () => (coinFlip() ? [] : [Preset('derivative-post')]),
    },

    'maybe-invert': {
      post: () => (coinFlip() ? [] : [Preset('invert')]),
    },

    'maybe-rotate': {
      settings: () => ({
        angle: random() * 360.0,
      }),
      post: (settings) =>
        coinFlip() ? [] : [Effect('rotate', { angle: settings.angle })],
    },

    'maybe-skew': {
      final: () => (coinFlip() ? [] : [Preset('skew')]),
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
      layers: ['worms', 'grime'],
      settings: () => ({
        worms_alpha: 0.25 + random() * 0.25,
        worms_behavior: worms.chaotic,
        worms_stride: 0.333 + random() * 0.333,
        worms_stride_deviation: 0.25,
      }),
    },

    celebrate: {
      layers: ['basic', 'posterize', 'distressed'],
      settings: () => ({
        brightness_distrib: distrib.ones,
        hue_range: 1,
        posterize_levels: randomInt(3, 5),
        speed: 0.025,
      }),
    },

    'cell-reflect': {
      layers: [
        'voronoi',
        'reflect-post',
        'derivative-post',
        'density-map',
        'maybe-invert',
        'bloom',
        'grain',
        'saturation',
      ],
      settings: () => ({
        dist_metric: randomMember(distanceMetricAbsoluteMembers()),
        palette_name: null,
        palette_on: false,
        reflect_range: randomInt(2, 4) * 5,
        saturation_final: 0.5 + random() * 0.25,
        voronoi_alpha: 0.333 + random() * 0.333,
        voronoi_diagram_type: voronoi.color_range,
        voronoi_nth: coinFlip(),
        voronoi_point_distrib: randomMember(
          Object.values(point).filter(
            (m) => ![point.square, point.waffle, point.chess].includes(m)
          )
        ),
        voronoi_point_freq: randomInt(2, 3),
      }),
    },

    'cell-refract': {
      layers: ['voronoi', 'ridge'],
      settings: () => ({
        colorSpace: randomMember(colorSpaceMembers()),
        dist_metric: randomMember(distanceMetricAbsoluteMembers()),
        ridges: true,
        voronoi_diagram_type: voronoi.range,
        voronoi_point_freq: randomInt(3, 4),
        voronoi_refract: randomInt(8, 12) * 0.5,
      }),
    },

    'cell-refract-2': {
      layers: [
        'voronoi',
        'refract-post',
        'derivative-post',
        'density-map',
        'saturation',
      ],
      settings: () => ({
        dist_metric: randomMember(distanceMetricAbsoluteMembers()),
        refract_range: randomInt(1, 3) * 0.25,
        voronoi_alpha: 0.333 + random() * 0.333,
        voronoi_diagram_type: voronoi.color_range,
        voronoi_point_distrib: randomMember(
          Object.values(point).filter(
            (m) => ![point.square, point.waffle, point.chess].includes(m)
          )
        ),
        voronoi_point_freq: randomInt(2, 3),
      }),
    },

    'cell-worms': {
      layers: [
        'multires-low',
        'voronoi',
        'worms',
        'density-map',
        'random-hue',
        'saturation',
      ],
      settings: () => ({
        freq: randomInt(3, 7),
        hue_range: 0.125 + random() * 0.875,
        voronoi_alpha: 0.75,
        voronoi_point_distrib: randomMember(
          point,
          Object.values(mask).filter(
            (m) => !valueMaskProceduralMembers.includes(m)
          )
        ),
        voronoi_point_freq: randomInt(2, 4),
        worms_density: 1500,
        worms_kink: randomInt(16, 32),
        worms_stride_deviation: 0,
      }),
    },

    chalky: {
      layers: ['basic', 'refract-post', 'octave-warp-post', 'outline', 'grain', 'lens'],
      settings: () => ({
        colorSpace: color.oklab,
        freq: randomInt(2, 3),
        octaves: randomInt(2, 3),
        outline_invert: true,
        refract_range: 0.1 + random() * 0.05,
        ridges: true,
        warp_octaves: 8,
        warp_range: 0.0333 + random() * 0.016667,
      }),
    },

    'chunky-knit': {
      layers: ['jorts', 'random-hue', 'contrast-final'],
      settings: () => ({
        angle: random() * 360.0,
        glyph_map_alpha: 0.333 + random() * 0.16667,
        glyph_map_mask: mask.waffle,
        glyph_map_zoom: 16.0,
      }),
    },

    'classic-desktop': {
      layers: ['basic', 'lens-warp'],
      settings: () => ({
        hue_range: 0.333 + random() * 0.333,
        lattice_drift: random(),
      }),
    },

    cloudburst: {
      layers: [
        'multires',
        'reflect-octaves',
        'octave-warp-octaves',
        'refract-post',
        'invert',
        'grain',
      ],
      settings: () => ({
        colorSpace: color.hsv,
        distrib: distrib.exp,
        freq: 2,
        hue_range: 0.05 - random() * 0.025,
        hue_rotation: 0.1 - random() * 0.025,
        lattice_drift: 0.75,
        palette_on: false,
        reflect_range: 0.125 + random() * 0.0625,
        refract_range: 0.1 + random() * 0.05,
        saturation_distrib: distrib.ones,
        speed: 0.075,
      }),
    },

    clouds: {
      layers: ['bloom', 'grain'],
      post: () => [Effect('clouds')],
    },

    concentric: {
      layers: ['wobble', 'voronoi', 'contrast-post', 'maybe-palette'],
      settings: () => ({
        colorSpace: color.rgb,
        dist_metric: randomMember(distanceMetricAbsoluteMembers()),
        distrib: distrib.ones,
        freq: 2,
        mask: mask.h_bar,
        speed: 0.75,
        spline_order: interp.constant,
        voronoi_diagram_type: voronoi.range,
        voronoi_refract: randomInt(8, 16),
        voronoi_point_drift: 0,
        voronoi_point_freq: randomInt(1, 2),
      }),
    },

    conference: {
      layers: ['value-mask', 'sobel', 'maybe-rotate', 'maybe-invert', 'grain'],
      settings: () => ({
        mask: mask.halftone,
        mask_repeat: randomInt(4, 12),
        spline_order: interp.cosine,
      }),
    },

    'contrast-post': {
      post: (settings) => [
        Effect('adjustContrast', { amount: settings.contrast_post }),
      ],
      settings: () => ({
        contrast_post: 1.25 + random() * 0.25,
      }),
    },

    'contrast-final': {
      settings: () => ({
        contrast_final: 1.25 + random() * 0.25,
      }),
      final: (settings) => [
        Effect('adjustContrast', { amount: settings.contrast_final }),
      ],
    },

    'cool-water': {
      layers: ['basic-water', 'funhouse', 'bloom', 'lens'],
      settings: () => ({
        warp_range: 0.0625 + random() * 0.0625,
        warp_freq: randomInt(2, 3),
      }),
    },

    'corner-case': {
      layers: ['multires-ridged', 'maybe-rotate', 'grain', 'saturation', 'vignette-dark'],
      settings: () => ({
        corners: true,
        lattice_drift: coinFlip(),
        spline_order: interp.constant,
      }),
    },

    corduroy: {
      layers: ['jorts', 'random-hue', 'contrast-final'],
      settings: () => ({
        saturation: 0.625 + random() * 0.125,
        glyph_map_zoom: 8.0,
      }),
    },

    'cosmic-thread': {
      layers: ['basic', 'worms', 'brightness-final', 'contrast-final', 'bloom'],
      settings: () => ({
        brightness_final: 0.1,
        colorSpace: color.rgb,
        contrast_final: 2.5,
        worms_alpha: 0.875,
        worms_behavior: randomMember(worms),
        worms_density: 0.125,
        worms_drunkenness: 0.125 + random() * 0.25,
        worms_duration: 125,
        worms_kink: 1.0,
        worms_stride: 0.75,
        worms_stride_deviation: 0.0,
      }),
    },

    cobblestone: {
      layers: [
        'bringing-hexy-back',
        'saturation',
        'texture',
        'erosion-worms',
        'shadow',
        'contrast-post',
        'contrast-final',
      ],
      settings: () => ({
        erosion_worms_alpha: random() * 0.05,
        erosion_worms_inverse: coinFlip(),
        erosion_worms_xy_blend: random() * 0.625,
        hue_range: 0.1 + random() * 0.05,
        saturation_final: random() * 0.05,
        shadow_alpha: 0.5,
        voronoi_point_freq: randomInt(3, 4) * 2,
        warp_freq: [randomInt(3, 4), randomInt(3, 4)],
        warp_range: 0.125,
        warp_octaves: 8,
      }),
    },

    'convolution-feedback': {
      post: () => [
        Effect('conv_feedback', {
          alpha: 0.5 * random() * 0.25,
          iterations: randomInt(250, 500),
        }),
      ],
    },

    corrupt: {
      post: () => [
        Effect('warp', {
          displacement: 0.025 + random() * 0.1,
          freq: [randomInt(2, 4), randomInt(1, 3)],
          octaves: randomInt(2, 4),
          spline_order: interp.constant,
        }),
      ],
    },

    'crime-scene': {
      layers: [
        'value-mask',
        'maybe-rotate',
        'grain',
        'dexter',
        'dexter',
        'grime',
        'lens',
      ],
      settings: () => ({
        mask: mask.chess,
        mask_repeat: randomInt(2, 3),
        saturation: coinFlip() ? 0 : 0.125,
        spline_order: interp.constant,
      }),
    },

    crooked: {
      layers: ['starfield', 'pixel-sort', 'glitchin-out'],
      settings: () => ({
        pixel_sort_angled: true,
        pixel_sort_darkest: false,
      }),
    },

    crt: {
      layers: ['scanline-error', 'snow'],
      settings: () => ({
        crt_brightness: 0.05,
        crt_contrast: 1.05,
      }),
      final: (settings) => [
        Effect('crt'),
        Preset('brightness-final', { brightness_final: settings.crt_brightness }),
        Preset('contrast-final', { contrast_final: settings.crt_contrast }),
      ],
    },

    crystallize: {
      layers: ['voronoi', 'vignette-bright', 'bloom', 'contrast-post', 'saturation'],
      settings: () => ({
        dist_metric: distance.triangular,
        voronoi_point_freq: 4,
        voronoi_alpha: 0.875,
        voronoi_diagram_type: voronoi.color_range,
        voronoi_nth: 4,
      }),
    },

    cubert: {
      layers: ['voronoi', 'crt', 'bloom'],
      settings: () => ({
        dist_metric: distance.triangular,
        freq: randomInt(4, 6),
        hue_range: 0.5 + random(),
        voronoi_diagram_type: voronoi.color_range,
        voronoi_inverse: true,
        voronoi_point_distrib: point.h_hex,
        voronoi_point_freq: randomInt(4, 6),
      }),
    },

    cubic: {
      layers: ['basic-voronoi', 'outline'],
      settings: () => ({
        voronoi_nth: randomInt(2, 8),
        voronoi_point_distrib: point.concentric,
        voronoi_point_freq: randomInt(3, 6),
        voronoi_diagram_type: randomMember([voronoi.range, voronoi.color_range]),
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

    deadbeef: {
      layers: ['value-mask', 'corrupt', 'bloom', 'crt', 'vignette-dark'],
      settings: () => ({
        freq: 6 * randomInt(9, 24),
        mask: mask.alphanum_hex,
      }),
    },

    'death-star-plans': {
      layers: [
        'voronoi',
        'refract-post',
        'maybe-rotate',
        'posterize',
        'sobel',
        'invert',
        'crt',
        'vignette-dark',
      ],
      settings: () => ({
        dist_metric: randomMember([distance.chebyshev, distance.manhattan]),
        posterize_levels: randomInt(3, 4),
        refract_range: 0.5 + random() * 0.25,
        refract_y_from_offset: true,
        voronoi_alpha: 1,
        voronoi_diagram_type: voronoi.range,
        voronoi_nth: randomInt(1, 3),
        voronoi_point_distrib: point.random,
        voronoi_point_freq: randomInt(2, 3),
      }),
    },

    'deep-field': {
      layers: ['multires', 'refract-octaves', 'octave-warp-octaves', 'bloom', 'lens'],
      settings: () => ({
        distrib: distrib.uniform,
        freq: randomInt(8, 10),
        hue_range: 1,
        mask: mask.sparser,
        mask_static: true,
        lattice_drift: 1,
        octave_blending: blend.alpha,
        octaves: 5,
        palette_on: false,
        speed: 0.05,
        refract_range: 0.2 + random() * 0.1,
        warp_freq: 2,
        warp_signed_range: true,
      }),
    },

    deeper: {
      layers: ['multires-alpha', 'funhouse', 'lens', 'contrast-final'],
      settings: () => ({
        hue_range: 0.75,
        octaves: 6,
        ridges: true,
      }),
    },

    degauss: {
      final: () => [
        Effect('degauss', { displacement: 0.06 + random() * 0.03 }),
        Preset('crt'),
      ],
    },

    'density-map': {
      layers: ['grain'],
      post: () => [
        Effect('density_map'),
        Effect('convolve', { kernel: mask.conv2d_invert }),
      ],
    },

    'density-wave': {
      layers: ['basic', 'reflect-post', 'density-map', 'invert', 'bloom'],
      settings: () => ({
        reflect_range: randomInt(3, 8),
        saturation: randomInt(0, 1),
      }),
    },

    'derivative-octaves': {
      settings: () => ({
        deriv_alpha: 1.0,
        dist_metric: randomMember(distanceMetricAbsoluteMembers()),
      }),
      octaves: (settings) => [
        Effect('derivative', {
          dist_metric: settings.dist_metric,
          alpha: settings.deriv_alpha,
        }),
      ],
      post: () => [Effect('fxaa')],
    },

    'derivative-post': {
      settings: () => ({
        deriv_alpha: 1.0,
        dist_metric: randomMember(distanceMetricAbsoluteMembers()),
      }),
      post: (settings) => [
        Effect('derivative', {
          dist_metric: settings.dist_metric,
          alpha: settings.deriv_alpha,
        }),
        Effect('fxaa'),
      ],
    },

    dexter: {
      layers: ['spatter-final'],
      settings: () => ({
        spatter_final_color: [
          0.35 + random() * 0.15,
          0.025 + random() * 0.0125,
          0.075 + random() * 0.0375,
        ],
      }),
    },

    different: {
      layers: [
        'multires',
        'sine-octaves',
        'reflect-octaves',
        'reindex-octaves',
        'funhouse',
        'lens',
      ],
      settings: () => ({
        freq: [randomInt(4, 6), randomInt(4, 6)],
        reflect_range: 7.5 + random() * 5.0,
        reindex_range: 0.25 + random() * 0.25,
        sine_range: randomInt(7, 12),
        speed: 0.025,
        warp_range: 0.0375 * random() * 0.0375,
      }),
    },

    distressed: {
      layers: ['grain', 'filthy', 'saturation'],
    },

    distance: {
      layers: [
        'multires',
        'derivative-octaves',
        'bloom',
        'shadow',
        'contrast-final',
        'maybe-rotate',
        'lens',
      ],
      settings: () => ({
        dist_metric: randomMember(distanceMetricAbsoluteMembers()),
        distrib: distrib.exp,
        freq: [randomInt(4, 5), randomInt(2, 3)],
        lattice_drift: 1,
        saturation: 0.0625 + random() * 0.125,
      }),
    },

    dla: {
      layers: ['basic', 'contrast-final'],
      settings: () => ({
        dla_alpha: 0.875 + random() * 0.125,
        dla_padding: randomInt(1, 8),
        dla_seed_density: 0.1 + random() * 0.05,
        dla_density: 0.2 + random() * 0.1,
      }),
      post: (settings) => [
        Effect('dla', {
          alpha: settings.dla_alpha,
          padding: settings.dla_padding,
          seed_density: settings.dla_seed_density,
          density: settings.dla_density,
        }),
      ],
    },

    'dla-forest': {
      layers: ['dla', 'reverb', 'contrast-final', 'bloom'],
      settings: () => ({
        dla_padding: randomInt(2, 8),
        reverb_iterations: randomInt(2, 4),
      }),
    },

    'domain-warp': {
      layers: [
        'multires-ridged',
        'refract-post',
        'vaseline',
        'grain',
        'vignette-dark',
        'saturation',
      ],
      settings: () => ({
        refract_range: 0.5 + random() * 0.5,
      }),
    },

    dropout: {
      layers: ['basic', 'maybe-rotate', 'derivative-post', 'maybe-invert', 'grain'],
      settings: () => ({
        colorSpace: randomMember(colorSpaceMembers()),
        distrib: distrib.ones,
        freq: [randomInt(4, 6), randomInt(2, 4)],
        mask: mask.dropout,
        octave_blending: blend.reduce_max,
        octaves: randomInt(4, 6),
        spline_order: interp.constant,
      }),
    },

    'eat-static': {
      layers: ['basic', 'be-kind-rewind', 'scanline-error', 'crt'],
      settings: () => ({
        freq: 512,
        saturation: 0,
        speed: 2.0,
      }),
    },

    'educational-video-film': {
      layers: ['basic', 'be-kind-rewind'],
      settings: () => ({
        colorSpace: color.oklab,
        ridges: true,
      }),
    },

    'electric-worms': {
      layers: ['voronoi', 'worms', 'density-map', 'glowing-edges', 'lens'],
      settings: () => ({
        dist_metric: randomMember(distanceMetricAll()),
        freq: randomInt(3, 6),
        lattice_drift: 1,
        voronoi_alpha: 0.25 + random() * 0.25,
        voronoi_diagram_type: voronoi.color_range,
        voronoi_nth: randomInt(0, 3),
        voronoi_point_freq: randomInt(3, 6),
        voronoi_point_distrib: point.random,
        worms_alpha: 0.666 + random() * 0.333,
        worms_behavior: worms.random,
        worms_density: 1000,
        worms_duration: 1,
        worms_kink: randomInt(7, 9),
        worms_stride: 1.0,
        worms_stride_deviation: 0,
        worms_quantize: coinFlip(),
      }),
    },

    emboss: {
      post: () => [Effect('convolve', { kernel: mask.conv2d_emboss })],
    },

    emo: {
      layers: [
        'value-mask',
        'voronoi',
        'contrast-final',
        'maybe-rotate',
        'saturation',
        'tint',
        'lens',
      ],
      settings: () => ({
        contrast_final: 4.0,
        dist_metric: randomMember([distance.manhattan, distance.chebyshev]),
        mask: mask.emoji,
        spline_order: interp.cosine,
        voronoi_diagram_type: voronoi.range,
        voronoi_refract: 0.125 + random() * 0.25,
      }),
    },

    emu: {
      layers: ['value-mask', 'voronoi', 'saturation', 'distressed'],
      settings: () => ({
        dist_metric: randomMember(distanceMetricAll()),
        distrib: distrib.ones,
        mask: stash('mask', randomMember(enumRange(mask.emoji_00, mask.emoji_26))),
        mask_repeat: 1,
        spline_order: interp.constant,
        voronoi_alpha: 1.0,
        voronoi_diagram_type: voronoi.range,
        voronoi_point_distrib: stash('mask'),
        voronoi_refract: 0.125 + random() * 0.125,
        voronoi_refract_y_from_offset: false,
      }),
    },

    entities: {
      layers: ['value-mask', 'refract-octaves', 'normalize'],
      settings: () => ({
        hue_range: 2.0 + random() * 2.0,
        mask: mask.invaders_square,
        mask_repeat: randomInt(3, 4) * 2,
        refract_range: 0.1 + random() * 0.05,
        refract_signed_range: false,
        refract_y_from_offset: true,
        spline_order: interp.cosine,
      }),
    },

    entity: {
      layers: ['entities', 'sobel', 'invert', 'bloom', 'random-hue', 'lens'],
      settings: () => ({
        corners: true,
        distrib: distrib.ones,
        hue_range: 1.0 + random() * 0.5,
        mask_repeat: 1,
        refract_range: 0.025 + random() * 0.0125,
        refract_signed_range: true,
        refract_y_from_offset: false,
        speed: 0.05,
      }),
    },

    'erosion-worms': {
      settings: () => ({
        erosion_worms_alpha: 0.5 + random() * 0.5,
        erosion_worms_contraction: 0.5 + random() * 0.5,
        erosion_worms_density: randomInt(25, 100),
        erosion_worms_inverse: false,
        erosion_worms_iterations: randomInt(25, 100),
        erosion_worms_quantize: false,
        erosion_worms_xy_blend: 0.75 + random() * 0.25,
      }),
      post: (settings) => [
        Effect('erosion_worms', {
          alpha: settings.erosion_worms_alpha,
          contraction: settings.erosion_worms_contraction,
          density: settings.erosion_worms_density,
          inverse: settings.erosion_worms_inverse,
          iterations: settings.erosion_worms_iterations,
          quantize: settings.erosion_worms_quantize,
          xy_blend: settings.erosion_worms_xy_blend,
        }),
        Effect('normalize'),
      ],
    },

    'escape-velocity': {
      layers: ['multires-low', 'erosion-worms', 'lens'],
      settings: () => ({
        colorSpace: randomMember(colorSpaceMembers()),
        distrib: randomMember([distrib.exp, distrib.uniform]),
        erosion_worms_contraction: 0.2 + random() * 0.1,
        erosion_worms_iterations: randomInt(625, 1125),
      }),
    },

    'falsetto': {
      final: () => [Effect('falseColor')],
    },

    fargate: {
      layers: ['serene', 'contrast-post', 'crt', 'saturation'],
      settings: () => ({
        brightness_distrib: distrib.uniform,
        freq: 3,
        octaves: 3,
        refract_range: 0.015 + random() * 0.0075,
        saturation_distrib: distrib.uniform,
        speed: -0.25,
        value_distrib: distrib.center_circle,
        value_freq: 3,
        value_refract_range: 0.015 + random() * 0.0075,
      }),
    },

    'fast-eddies': {
      layers: ['basic', 'voronoi', 'worms', 'contrast-final', 'saturation'],
      settings: () => ({
        dist_metric: distance.euclidean,
        hue_range: 0.25 + random() * 0.75,
        hue_rotation: random(),
        octaves: randomInt(1, 3),
        palette_on: false,
        ridges: coinFlip(),
        voronoi_alpha: 0.5 + random() * 0.5,
        voronoi_diagram_type: voronoi.flow,
        voronoi_point_freq: randomInt(2, 6),
        voronoi_refract: 1.0,
        worms_alpha: 0.5 + random() * 0.5,
        worms_behavior: worms.chaotic,
        worms_density: 1000,
        worms_duration: 6,
        worms_kink: randomInt(125, 375),
        worms_stride: 1.0,
        worms_stride_deviation: 0.0,
      }),
    },

    fibers: {
      final: () => [Effect('fibers')],
    },

    figments: {
      layers: [
        'multires-low',
        'voronoi',
        'funhouse',
        'wormhole',
        'bloom',
        'contrast-final',
        'lens',
      ],
      settings: () => ({
        freq: 2,
        hue_range: 2,
        lattice_drift: 1,
        speed: 0.025,
        voronoi_diagram_type: voronoi.flow,
        voronoi_refract: 0.333 + random() * 0.333,
        wormhole_stride: 0.02 + random() * 0.01,
        wormhole_kink: 4,
      }),
    },

    filthy: {
      layers: ['grime', 'scratches', 'stray-hair'],
    },

    fireball: {
      layers: [
        'basic',
        'periodic-refract',
        'refract-post',
        'refract-post',
        'bloom',
        'lens',
        'contrast-final',
      ],
      settings: () => ({
        contrast_final: 2.5,
        distrib: distrib.center_circle,
        hue_rotation: 0.925,
        freq: 1,
        refract_range: 0.025 + random() * 0.0125,
        refract_y_from_offset: false,
        value_distrib: distrib.center_circle,
        value_freq: 1,
        value_refract_range: 0.05 + random() * 0.025,
        speed: 0.05,
      }),
    },

    'financial-district': {
      layers: ['voronoi', 'bloom', 'contrast-final', 'saturation'],
      settings: () => ({
        dist_metric: distance.manhattan,
        voronoi_diagram_type: voronoi.range_regions,
        voronoi_point_distrib: point.random,
        voronoi_nth: randomInt(1, 3),
        voronoi_point_freq: 2,
      }),
    },

    'fossil-hunt': {
      layers: ['voronoi', 'refract-octaves', 'posterize-outline', 'grain', 'saturation'],
      settings: () => ({
        freq: randomInt(3, 5),
        lattice_drift: 1.0,
        posterize_levels: randomInt(3, 5),
        refract_range: randomInt(2, 4) * 0.5,
        refract_y_from_offset: true,
        voronoi_alpha: 0.5,
        voronoi_diagram_type: voronoi.color_range,
        voronoi_point_freq: 10,
      }),
    },

    'fractal-forms': {
      layers: ['fractal-seed'],
      settings: () => ({
        worms_kink: randomInt(256, 512),
      }),
    },

    'fractal-seed': {
      layers: [
        'multires-low',
        'worms',
        'density-map',
        'random-hue',
        'bloom',
        'shadow',
        'contrast-final',
        'saturation',
        'aberration',
      ],
      settings: () => ({
        freq: randomInt(2, 3),
        hue_range: 1.0 + random() * 3.0,
        ridges: coinFlip(),
        speed: 0.05,
        palette_on: false,
        worms_behavior: randomMember([worms.chaotic, worms.random]),
        worms_alpha: 0.9 + random() * 0.1,
        worms_density: randomInt(750, 1250),
        worms_duration: randomInt(2, 3),
        worms_kink: 1.0,
        worms_stride: 1.0,
        worms_stride_deviation: 0.0,
      }),
    },

    'fractal-smoke': {
      layers: ['fractal-seed'],
      settings: () => ({
        worms_behavior: worms.random,
        worms_stride: randomInt(96, 192),
      }),
    },

    fractile: {
      layers: ['symmetry', 'voronoi', 'reverb', 'contrast-post', 'palette', 'random-hue', 'maybe-rotate', 'lens'],
      settings: () => ({
        dist_metric: randomMember(distanceMetricAbsoluteMembers),
        reverb_iterations: randomInt(2, 4),
        reverb_octaves: randomInt(2, 4),
        voronoi_alpha: 0.5 + random() * 0.5,
        voronoi_diagram_type: voronoi.color_range,
        voronoi_nth: randomInt(0, 2),
        voronoi_point_distrib: randomMember([point.square, point.waffle, point.chess]),
        voronoi_point_freq: randomInt(2, 3),
      }),
    },

    fundamentals: {
      layers: ['voronoi', 'derivative-post', 'density-map', 'grain', 'saturation'],
      settings: () => ({
        dist_metric: randomMember([distance.manhattan, distance.chebyshev]),
        freq: randomInt(3, 5),
        voronoi_diagram_type: voronoi.color_range,
        voronoi_nth: randomInt(3, 5),
        voronoi_point_freq: randomInt(3, 5),
        voronoi_refract: 0.125 + random() * 0.0625,
      }),
    },

    funhouse: {
      settings: () => ({
        warp_freq: [randomInt(2, 4), randomInt(2, 4)],
        warp_octaves: randomInt(1, 4),
        warp_range: 0.25 + random() * 0.125,
        warp_signed_range: false,
        warp_spline_order: interp.bicubic,
      }),
      post: (settings) => [
        Effect('warp', {
          displacement: settings.warp_range,
          freq: settings.warp_freq,
          octaves: settings.warp_octaves,
          signed_range: settings.warp_signed_range,
          spline_order: settings.warp_spline_order,
        }),
      ],
    },

    'funky-glyphs': {
      layers: ['value-mask', 'refract-post', 'contrast-final', 'maybe-rotate', 'saturation', 'lens', 'grain'],
      settings: () => ({
        distrib: randomMember([distrib.ones, distrib.uniform]),
        mask: randomMember(valueMaskGlyphMembers),
        mask_repeat: randomInt(1, 6),
        octaves: randomInt(1, 2),
        refract_range: 0.125 + random() * 0.125,
        refract_signed_range: false,
        refract_y_from_offset: true,
        spline_order: randomMember(enumRange(interp.linear, interp.bicubic)),
      }),
    },

    galalaga: {
      layers: ['value-mask', 'contrast-final', 'glitchin-out'],
      settings: () => ({
        distrib: distrib.uniform,
        hue_range: random() * 2.5,
        mask: mask.invaders_square,
        mask_repeat: 4,
        spline_order: interp.constant,
      }),
      post: () => [
        Effect('glyphMap', { colorize: true, mask: mask.invaders_square, zoom: 32.0 }),
        Effect('glyphMap', {
          colorize: true,
          mask: randomMember([mask.invaders_square, mask.rgb]),
          zoom: 4.0,
        }),
        Effect('normalize'),
      ],
    },

    'game-show': {
      layers: ['basic', 'maybe-rotate', 'posterize', 'be-kind-rewind'],
      settings: () => ({
        freq: randomInt(8, 16) * 2,
        mask: randomMember([mask.h_tri, mask.v_tri]),
        posterize_levels: randomInt(2, 5),
        spline_order: interp.cosine,
      }),
    },

    glacial: {
      layers: ['fractal-smoke'],
      settings: () => ({
        worms_quantize: true,
      }),
    },

    'glitchin-out': {
      layers: ['corrupt'],
      final: () => [Effect('glitch'), Preset('crt'), Preset('bloom')],
    },

    globules: {
      layers: ['multires-low', 'reflect-octaves', 'density-map', 'shadow', 'lens'],
      settings: () => ({
        distrib: distrib.ones,
        freq: randomInt(3, 6),
        hue_range: 0.25 + random() * 0.5,
        lattice_drift: 1,
        mask: mask.sparse,
        mask_static: true,
        octaves: randomInt(3, 6),
        palette_on: false,
        reflect_range: 2.5,
        saturation: 0.175 + random() * 0.175,
        speed: 0.125,
      }),
    },

    glom: {
      layers: ['basic', 'refract-octaves', 'reflect-octaves', 'refract-post', 'reflect-post', 'funhouse',
               'bloom', 'shadow', 'contrast-post', 'lens'],
      settings: () => ({
        distrib: distrib.uniform,
        freq: [2, 2],
        hue_range: 0.25 + random() * 0.125,
        lattice_drift: 1,
        octaves: 2,
        reflect_range: 0.625 + random() * 0.375,
        refract_range: 0.333 + random() * 0.16667,
        refract_signed_range: false,
        refract_y_from_offset: true,
        speed: 0.025,
        warp_range: 0.0625 + random() * 0.030625,
        warp_octaves: 1,
      }),
    },

    'glowing-edges': {
      final: () => [Effect('glowingEdges')],
    },

    'glyph-map': {
      layers: ['basic'],
      settings: () => ({
        glyph_map_alpha: 1.0,
        glyph_map_colorize: coinFlip(),
        glyph_map_spline_order: interp.constant,
        glyph_map_mask: randomMember(
          valueMaskProceduralMembers.filter((m) => {
            const [h, w] = maskShape(m);
            return h === w;
          })
        ),
        glyph_map_zoom: randomInt(6, 10),
      }),
      post: (settings) => [
        Effect('glyphMap', {
          alpha: settings.glyph_map_alpha,
          colorize: settings.glyph_map_colorize,
          mask: settings.glyph_map_mask,
          spline_order: settings.glyph_map_spline_order,
          zoom: settings.glyph_map_zoom,
        }),
      ],
    },

    glyphic: {
      layers: ['value-mask', 'posterize', 'palette', 'saturation', 'maybe-rotate', 'maybe-invert', 'distressed'],
      settings: () => ({
        corners: true,
        mask: randomMember(valueMaskProceduralMembers),
        octave_blending: blend.reduce_max,
        octaves: randomInt(3, 5),
        posterize_levels: 1,
        saturation: 0,
        spline_order: interp.cosine,
      }),
      generator: (settings) => ({
        freq: maskShape(settings.mask).slice(0, 2),
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

    'graph-paper': {
      layers: ['wobble', 'voronoi', 'derivative-post', 'maybe-rotate', 'lens', 'crt', 'bloom', 'contrast-final'],
      settings: () => ({
        color_space: color.rgb,
        corners: true,
        distrib: distrib.ones,
        dist_metric: distance.euclidean,
        freq: randomInt(3, 4) * 2,
        mask: mask.chess,
        spline_order: interp.constant,
        voronoi_alpha: 0.5 + random() * 0.25,
        voronoi_refract: 0.75 + random() * 0.375,
        voronoi_refract_y_from_offset: true,
        voronoi_diagram_type: voronoi.flow,
      }),
    },

    grass: {
      layers: ['multires', 'worms', 'grain'],
      settings: () => ({
        color_space: color.hsv,
        freq: randomInt(6, 12),
        hue_rotation: 0.25 + random() * 0.05,
        lattice_drift: 1,
        palette_on: false,
        saturation: 0.625 + random() * 0.25,
        worms_behavior: randomMember([worms.chaotic, worms.meandering]),
        worms_alpha: 0.9,
        worms_density: 50 + random() * 25,
        worms_drunkenness: 0.125,
        worms_duration: 1.125,
        worms_stride: 0.875,
        worms_stride_deviation: 0.125,
        worms_kink: 0.125 + random() * 0.5,
      }),
    },

    grayscale: {
      final: () => [Effect('adjust_saturation', { amount: 0 })],
    },

    griddy: {
      layers: ['basic', 'sobel', 'invert', 'bloom'],
      settings: () => ({
        freq: randomInt(3, 9),
        mask: mask.chess,
        octaves: randomInt(3, 8),
        spline_order: interp.constant,
      }),
    },

    grime: {
      final: () => [Effect('grime')],
    },

    'groove-is-stored-in-the-heart': {
      layers: ['basic', 'posterize', 'ripple', 'distressed'],
      settings: () => ({
        distrib: distrib.column_index,
        posterize_levels: randomInt(1, 2),
        ripple_range: 0.75 + random() * 0.375,
      }),
    },

    'i-made-an-art': {
      layers: ['basic', 'outline', 'distressed', 'contrast-final', 'saturation'],
      settings: () => ({
        spline_order: interp.constant,
        lattice_drift: randomInt(5, 10),
        hue_range: random() * 4,
        hue_rotation: random(),
      }),
    },

    inkling: {
      layers: [
        'voronoi',
        'refract-post',
        'funhouse',
        'grayscale',
        'density-map',
        'contrast-post',
        'maybe-invert',
        'fibers',
        'grime',
        'scratches',
      ],
      settings: () => ({
        distrib: distrib.ones,
        dist_metric: distance.euclidean,
        contrast_post: 2.5,
        freq: randomInt(2, 4),
        lattice_drift: 1.0,
        mask: mask.dropout,
        mask_static: true,
        refract_range: 0.25 + random() * 0.125,
        voronoi_diagram_type: voronoi.flow,
        voronoi_point_freq: randomInt(3, 5),
        voronoi_refract: 0.25 + random() * 0.125,
        warp_range: 0.125 + random() * 0.0625,
      }),
    },

    invert: {
      post: () => [Effect('convolve', { kernel: mask.conv2d_invert })],
    },

    'is-this-anything': {
      layers: ['soup'],
      settings: () => ({
        refract_range: 2.5 + random() * 1.25,
        voronoi_point_freq: 1,
      }),
    },

    'its-the-fuzz': {
      layers: ['multires-low', 'muppet-fur'],
      settings: () => ({
        worms_behavior: worms.unruly,
        worms_drunkenness: 0.5 + random() * 0.25,
        worms_duration: 2.0 + random(),
      }),
    },

    jorts: {
      layers: [
        'glyph-map',
        'funhouse',
        'skew',
        'shadow',
        'brightness-post',
        'contrast-post',
        'vignette-dark',
        'grain',
        'saturation',
      ],
      settings: () => ({
        angle: 0,
        freq: [128, 512],
        glyph_map_alpha: 0.5 + random() * 0.25,
        glyph_map_colorize: true,
        glyph_map_mask: mask.v_bar,
        glyph_map_spline_order: interp.linear,
        glyph_map_zoom: 4.0,
        hue_rotation: 0.5 + random() * 0.05,
        hue_range: 0.0625 + random() * 0.0625,
        palette_on: false,
        warp_freq: [randomInt(2, 3), randomInt(2, 3)],
        warp_range: 0.0075 + random() * 0.00625,
        warp_octaves: 1,
      }),
    },

    'jovian-clouds': {
      layers: [
        'voronoi',
        'worms',
        'brightness-post',
        'contrast-post',
        'shadow',
        'tint',
        'grain',
        'saturation',
        'lens',
      ],
      settings: () => ({
        contrast_post: 2.0,
        dist_metric: distance.euclidean,
        freq: [randomInt(4, 7), randomInt(1, 3)],
        hue_range: 0.333 + random() * 0.16667,
        hue_rotation: 0.5,
        voronoi_alpha: 0.175 + random() * 0.25,
        voronoi_diagram_type: voronoi.flow,
        voronoi_point_distrib: point.random,
        voronoi_point_freq: randomInt(8, 10),
        voronoi_refract: 5.0 + random() * 3.0,
        worms_behavior: worms.chaotic,
        worms_alpha: 0.175 + random() * 0.25,
        worms_density: 500,
        worms_duration: 2.0,
        worms_kink: 192,
        worms_stride: 1.0,
        worms_stride_deviation: 0.0625,
      }),
    },

    'just-refracts-maam': {
      layers: ['basic', 'refract-octaves', 'refract-post', 'shadow', 'lens'],
      settings: () => ({
        corners: true,
        refract_range: 0.5 + random() * 0.5,
        ridges: coinFlip(),
      }),
    },

    kaleido: {
      layers: ['voronoi-refract', 'wobble'],
      settings: () => ({
        colorSpace: color.hsv,
        freq: randomInt(8, 12),
        hue_range: 0.5 + random() * 2.5,
        kaleido_point_corners: true,
        kaleido_point_distrib: point.random,
        kaleido_point_freq: 1,
        kaleido_sdf_sides: randomInt(0, 10),
        kaleido_sides: randomInt(3, 16),
        kaleido_blend_edges: false,
        palette_on: false,
        speed: 0.125,
        voronoi_point_freq: randomInt(8, 12),
      }),
      post: (settings) => [
        Effect('kaleido', {
          blend_edges: settings.kaleido_blend_edges,
          point_corners: settings.kaleido_point_corners,
          point_distrib: settings.kaleido_point_distrib,
          point_freq: settings.kaleido_point_freq,
          sdf_sides: settings.kaleido_sdf_sides,
          sides: settings.kaleido_sides,
        }),
      ],
    },

    'knotty-clouds': {
      layers: ['basic', 'voronoi', 'worms'],
      settings: () => ({
        voronoi_alpha: 0.125 + random() * 0.25,
        voronoi_diagram_type: voronoi.color_range,
        voronoi_point_freq: randomInt(6, 10),
        worms_alpha: 0.666 + random() * 0.333,
        worms_behavior: worms.obedient,
        worms_density: 1000,
        worms_duration: 1,
        worms_kink: 4,
      }),
    },

    later: {
      layers: [
        'value-mask',
        'multires',
        'wobble',
        'voronoi',
        'funhouse',
        'glowing-edges',
        'crt',
        'vignette-dark',
      ],
      settings: () => ({
        dist_metric: distance.euclidean,
        freq: randomInt(2, 5),
        mask: randomMember(valueMaskProceduralMembers),
        spline_order: interp.constant,
        voronoi_diagram_type: voronoi.flow,
        voronoi_point_distrib: point.random,
        voronoi_point_freq: randomInt(3, 6),
        voronoi_refract: 1.0 + random() * 0.5,
        warp_freq: randomInt(2, 4),
        warp_spline_order: interp.bicubic,
        warp_octaves: 2,
        warp_range: 0.05 + random() * 0.025,
      }),
    },

    'lattice-noise': {
      layers: [
        'basic',
        'derivative-octaves',
        'derivative-post',
        'density-map',
        'shadow',
        'grain',
        'saturation',
        'vignette-dark',
      ],
      settings: () => ({
        dist_metric: randomMember(distanceMetricAbsoluteMembers()),
        freq: randomInt(2, 5),
        lattice_drift: 1.0,
        octaves: randomInt(2, 3),
        ridges: coinFlip(),
      }),
    },

    lcd: {
      layers: ['value-mask', 'invert', 'skew', 'shadow', 'vignette-bright', 'grain'],
      settings: () => ({
        mask: randomMember([mask.lcd, mask.lcd_binary]),
        mask_repeat: randomInt(8, 12),
        saturation: 0.0,
      }),
    },

    lens: {
      layers: ['lens-distortion', 'aberration', 'vaseline', 'tint', 'vignette-dark'],
      settings: () => ({
        lens_brightness: 0.05 + random() * 0.025,
        lens_contrast: 1.05 + random() * 0.025,
      }),
      final: (settings) => [
        Preset('brightness-final', { brightness_final: settings.lens_brightness }),
        Preset('contrast-final', { contrast_final: settings.lens_contrast }),
      ],
    },

    'lens-distortion': {
      final: () => [
        Effect('lens_distortion', {
          displacement: (0.125 + random() * 0.0625) * (coinFlip() ? 1 : -1),
        }),
      ],
    },

    'lens-warp': {
      post: () => [
        Effect('lens_warp', { displacement: 0.125 + random() * 0.0625 }),
        Effect('lens_distortion', {
          displacement: 0.25 + random() * 0.125 * (coinFlip() ? 1 : -1),
        }),
      ],
    },

    'light-leak': {
      layers: ['vignette-bright'],
      settings: () => ({
        light_leak_alpha: 0.25 + random() * 0.125,
      }),
      final: (settings) => [
        Effect('light_leak', { alpha: settings.light_leak_alpha }),
      ],
    },

    'look-up': {
      layers: [
        'multires-alpha',
        'brightness-post',
        'contrast-post',
        'contrast-final',
        'saturation',
        'lens',
        'bloom',
      ],
      settings: () => ({
        brightness_post: -0.075,
        colorSpace: color.hsv,
        contrast_final: 1.5,
        distrib: distrib.exp,
        freq: randomInt(30, 40),
        hue_range: 0.333 + random() * 0.333,
        lattice_drift: 0,
        mask: mask.sparsest,
        octaves: 10,
        ridges: true,
        saturation: 0.5,
        speed: 0.025,
      }),
    },

    'low-poly': {
      settings: () => ({
        lowpoly_distrib: randomMember(pointCircularMembers),
        lowpoly_freq: randomInt(10, 20),
      }),
      post: (settings) => [
        Effect('lowpoly', {
          distrib: settings.lowpoly_distrib,
          freq: settings.lowpoly_freq,
        }),
      ],
    },

    'low-poly-regions': {
      layers: ['voronoi', 'low-poly'],
      settings: () => ({
        voronoi_diagram_type: voronoi.color_regions,
        voronoi_point_freq: randomInt(2, 3),
      }),
    },

    lsd: {
      layers: ['basic', 'refract-post', 'invert', 'random-hue', 'lens', 'grain'],
      settings: () => ({
        brightness_distrib: distrib.ones,
        freq: randomInt(3, 4),
        hue_range: randomInt(3, 4),
        speed: 0.025,
      }),
    },

    'magic-smoke': {
      layers: ['multires', 'worms', 'lens'],
      settings: () => ({
        octaves: randomInt(2, 3),
        worms_alpha: 1,
        worms_behavior: randomMember([worms.obedient, worms.crosshatch]),
        worms_density: 750,
        worms_duration: 0.25,
        worms_kink: randomInt(1, 3),
        worms_stride: randomInt(64, 256),
      }),
    },

    mcpaint: {
      layers: [
        'glyph-map',
        'skew',
        'grain',
        'vignette-dark',
        'brightness-final',
        'contrast-final',
        'saturation',
      ],
      settings: () => ({
        corners: true,
        freq: randomInt(2, 8),
        glyph_map_colorize: false,
        glyph_map_mask: mask.mcpaint,
        glyph_map_zoom: randomInt(2, 4),
        spline_order: interp.cosine,
      }),
    },

    'moire-than-a-feeling': {
      layers: ['basic', 'wormhole', 'density-map', 'invert', 'contrast-post'],
      settings: () => ({
        octaves: randomInt(1, 2),
        saturation: 0,
        wormhole_kink: 128,
        wormhole_stride: 0.0005,
      }),
    },

    'molten-glass': {
      layers: [
        'basic',
        'sine-octaves',
        'octave-warp-post',
        'brightness-post',
        'contrast-post',
        'bloom',
        'shadow',
        'normalize',
        'lens',
      ],
      settings: () => ({
        hue_range: random() * 3.0,
      }),
    },

    multires: {
      layers: ['basic'],
      settings: () => ({
        octaves: randomInt(6, 8),
      }),
    },

    'multires-alpha': {
      layers: ['multires'],
      settings: () => ({
        distrib: distrib.exp,
        lattice_drift: 1,
        octave_blending: blend.alpha,
        octaves: 5,
        palette_on: false,
      }),
    },

    'multires-low': {
      layers: ['basic'],
      settings: () => ({
        octaves: randomInt(2, 4),
      }),
    },

    'multires-ridged': {
      layers: ['multires'],
      settings: () => ({
        lattice_drift: random(),
        ridges: true,
      }),
    },

    'muppet-fur': {
      layers: ['basic', 'worms', 'rotate', 'bloom', 'lens'],
      settings: () => ({
        colorSpace: randomMember([color.oklab, color.hsv]),
        freq: randomInt(2, 3),
        hue_range: random() * 0.25,
        hue_rotation: random(),
        lattice_drift: random() * 0.333,
        palette_on: false,
        worms_alpha: 0.875 + random() * 0.125,
        worms_behavior: worms.unruly,
        worms_density: randomInt(500, 1250),
        worms_drunkenness: random() * 0.025,
        worms_duration: 2.0 + random() * 1.0,
        worms_stride: 1.0,
        worms_stride_deviation: 0.0,
      }),
    },

    mycelium: {
      layers: [
        'multires',
        'grayscale',
        'octave-warp-octaves',
        'derivative-post',
        'normalize',
        'fractal-seed',
        'vignette-dark',
        'contrast-post',
      ],
      settings: () => ({
        colorSpace: color.grayscale,
        distrib: distrib.ones,
        freq: [randomInt(3, 4), randomInt(3, 4)],
        lattice_drift: 1.0,
        mask: mask.h_tri,
        mask_static: true,
        speed: 0.05,
        warp_freq: [randomInt(2, 3), randomInt(2, 3)],
        warp_range: 2.5 + random() * 1.25,
        worms_behavior: worms.random,
      }),
    },

    nausea: {
      layers: ['value-mask', 'ripple', 'normalize', 'aberration'],
      settings: () => ({
        colorSpace: color.rgb,
        mask: randomMember([mask.h_bar, mask.v_bar]),
        mask_repeat: randomInt(5, 8),
        ripple_kink: 1.25 + random() * 1.25,
        ripple_freq: randomInt(2, 3),
        ripple_range: 1.25 + random(),
        spline_order: interp.constant,
      }),
    },

    nebula: {
      final: () => [Effect('nebula')],
    },

    nerdvana: {
      layers: ['symmetry', 'voronoi', 'density-map', 'reverb', 'bloom'],
      settings: () => ({
        dist_metric: distance.euclidean,
        palette_on: false,
        reverb_octaves: 2,
        reverb_ridges: false,
        voronoi_diagram_type: voronoi.color_range,
        voronoi_point_distrib: randomMember(pointCircularMembers),
        voronoi_point_freq: randomInt(5, 10),
        voronoi_nth: 1,
      }),
    },

    'neon-cambrian': {
      layers: [
        'voronoi',
        'posterize',
        'wormhole',
        'derivative-post',
        'brightness-final',
        'bloom',
        'contrast-final',
        'aberration',
      ],
      settings: () => ({
        contrast_final: 4.0,
        dist_metric: distance.euclidean,
        freq: 12,
        hue_range: 4,
        posterize_levels: randomInt(20, 25),
        voronoi_diagram_type: voronoi.color_flow,
        voronoi_point_distrib: point.random,
        wormhole_stride: 0.2 + random() * 0.1,
      }),
    },

    'noise-blaster': {
      layers: ['multires', 'reindex-octaves', 'reindex-post', 'grain'],
      settings: () => ({
        freq: randomInt(3, 4),
        lattice_drift: 1,
        reindex_range: 3,
        speed: 0.025,
      }),
    },

    'noise-lake': {
      layers: ['multires-low', 'value-refract', 'snow', 'lens'],
      settings: () => ({
        hue_range: 0.75 + random() * 0.375,
        freq: randomInt(4, 6),
        lattice_drift: 1.0,
        ridges: true,
        value_freq: randomInt(4, 6),
        value_refract_range: 0.25 + random() * 0.125,
      }),
    },

    'noise-tunnel': {
      layers: ['basic', 'periodic-distance', 'periodic-refract', 'lens'],
      settings: () => ({
        hue_range: 2.0 + random(),
        speed: 1.0,
      }),
    },

    noirmaker: {
      layers: ['grain', 'grayscale', 'light-leak', 'bloom', 'contrast-final', 'vignette-dark'],
    },

    normals: {
      final: () => [Effect('normalMap')],
    },

    now: {
      layers: [
        'multires-low',
        'normalize',
        'wobble',
        'voronoi',
        'funhouse',
        'outline',
        'grain',
        'saturation',
      ],
      settings: () => ({
        dist_metric: distance.euclidean,
        freq: randomInt(3, 10),
        hue_range: random(),
        lattice_drift: coinFlip(),
        saturation: 0.5 + random() * 0.5,
        spline_order: interp.constant,
        voronoi_diagram_type: voronoi.flow,
        voronoi_point_distrib: point.random,
        voronoi_point_freq: randomInt(3, 10),
        voronoi_refract: 2.0 + random(),
        warp_freq: randomInt(2, 4),
        warp_octaves: 1,
        warp_range: 0.0375 + random() * 0.0375,
        warp_spline_order: interp.bicubic,
      }),
    },

    'nudge-hue': {
      final: () => [Effect('adjustHue', { amount: -0.125 })],
    },

    numberwang: {
      layers: [
        'value-mask',
        'funhouse',
        'posterize',
        'palette',
        'maybe-invert',
        'random-hue',
        'grain',
        'saturation',
      ],
      settings: () => ({
        mask: mask.alphanum_numeric,
        mask_repeat: randomInt(5, 10),
        posterize_levels: 2,
        spline_order: interp.cosine,
        warp_range: 0.25 + random() * 0.75,
        warp_freq: randomInt(2, 4),
        warp_octaves: 1,
        warp_spline_order: interp.bicubic,
      }),
    },

    'octave-blend': {
      layers: ['multires-alpha'],
      settings: () => ({
        corners: true,
        distrib: randomMember([distrib.ones, distrib.uniform]),
        freq: randomInt(2, 5),
        lattice_drift: 0,
        mask: randomMember(valueMaskProceduralMembers),
        spline_order: interp.constant,
      }),
    },

    'octave-warp-octaves': {
      settings: () => ({
        warp_freq: [randomInt(2, 4), randomInt(2, 4)],
        warp_octaves: randomInt(1, 4),
        warp_range: 0.5 + random() * 0.25,
        warp_signed_range: false,
        warp_spline_order: interp.bicubic,
      }),
      octaves: (settings) => [
        Effect('warp', {
          displacement: settings.warp_range,
          freq: settings.warp_freq,
          octaves: settings.warp_octaves,
          signed_range: settings.warp_signed_range,
          spline_order: settings.warp_spline_order,
        }),
      ],
    },

    'octave-warp-post': {
      settings: () => ({
        speed: 0.025 + random() * 0.0125,
        warp_freq: randomInt(2, 3),
        warp_octaves: randomInt(2, 4),
        warp_range: 2.0 + random(),
        warp_spline_order: interp.bicubic,
      }),
      post: (settings) => [
        Effect('warp', {
          displacement: settings.warp_range,
          freq: settings.warp_freq,
          octaves: settings.warp_octaves,
          spline_order: settings.warp_spline_order,
        }),
      ],
    },

    oldschool: {
      layers: ['voronoi', 'normalize', 'random-hue', 'saturation', 'distressed'],
      settings: () => ({
        colorSpace: color.rgb,
        corners: true,
        dist_metric: distance.euclidean,
        distrib: distrib.ones,
        freq: randomInt(2, 5) * 2,
        mask: mask.chess,
        spline_order: interp.constant,
        speed: 0.05,
        voronoi_diagram_type: voronoi.flow,
        voronoi_point_distrib: point.random,
        voronoi_point_freq: randomInt(4, 8),
        voronoi_refract: randomInt(8, 12) * 0.5,
      }),
    },

    'one-art-please': {
      layers: ['contrast-post', 'grain', 'light-leak', 'saturation', 'texture'],
    },

    oracle: {
      layers: ['value-mask', 'random-hue', 'maybe-invert', 'distressed'],
      settings: () => ({
        corners: true,
        mask: mask.iching,
        mask_repeat: randomInt(1, 8),
        spline_order: interp.constant,
      }),
    },

    'outer-limits': {
      layers: [
        'symmetry',
        'reindex-post',
        'normalize',
        'grain',
        'be-kind-rewind',
        'vignette-dark',
        'contrast-post',
      ],
      settings: () => ({
        palette_on: false,
        reindex_range: randomInt(8, 16),
        saturation: 0,
      }),
    },

    outline: {
      settings: () => ({
        dist_metric: distance.euclidean,
        outline_invert: false,
      }),
      post: (settings) => [
        Effect('outline', {
          sobelMetric: settings.dist_metric,
          invert: settings.outline_invert,
        }),
        Effect('fxaa'),
      ],
    },

    oxidize: {
      layers: ['multires', 'refract-post', 'contrast-post', 'bloom', 'shadow', 'saturation', 'lens'],
      settings: () => ({
        distrib: distrib.exp,
        freq: 4,
        hue_range: 0.875 + random() * 0.25,
        lattice_drift: 1,
        octave_blending: blend.reduce_max,
        octaves: 8,
        refract_range: 0.1 + random() * 0.05,
        saturation_final: 0.5,
        speed: 0.05,
      }),
    },

    'paintball-party': {
      layers: [
        'basic',
        ...Array.from({ length: randomInt(1, 4) }, () => 'spatter-post'),
        ...Array.from({ length: randomInt(1, 4) }, () => 'spatter-final'),
        'bloom',
      ],
      settings: () => ({
        distrib: randomMember([distrib.zeros, distrib.ones]),
      }),
    },

    painterly: {
      layers: ['value-mask', 'ripple', 'funhouse', 'maybe-rotate', 'saturation', 'grain'],
      settings: () => ({
        distrib: distrib.uniform,
        hue_range: 0.333 + random() * 0.666,
        mask: randomMember(valueMaskGridMembers),
        mask_repeat: 1,
        octaves: 8,
        ridges: true,
        ripple_freq: randomInt(4, 6),
        ripple_kink: 0.0625 + random() * 0.125,
        ripple_range: 0.0625 + random() * 0.125,
        spline_order: interp.linear,
        warp_freq: randomInt(5, 7),
        warp_octaves: 8,
        warp_range: 0.0625 + random() * 0.125,
      }),
    },

    palette: {
      layers: ['maybe-palette'],
      settings: () => ({
        palette_name: randomMember(PALETTES),
        palette_on: true,
      }),
    },

    pantheon: {
      layers: ['runes-of-arecibo'],
      settings: () => ({
        mask: randomMember([
          mask.invaders_square,
          randomMember(valueMaskGlyphMembers),
        ]),
        mask_repeat: randomInt(2, 3) * 2,
        octaves: 2,
        posterize_levels: randomInt(3, 6),
        refract_range: randomMember([0, random() * 0.05]),
        refract_signed_range: false,
        refract_y_from_offset: true,
        spline_order: interp.cosine,
      }),
    },

    pearlescent: {
      layers: [
        'voronoi',
        'normalize',
        'refract-post',
        'brightness-final',
        'bloom',
        'shadow',
        'lens',
      ],
      settings: () => ({
        brightness_final: 0.05,
        dist_metric: distance.euclidean,
        freq: [2, 2],
        hue_range: randomInt(3, 5),
        octaves: randomInt(3, 5),
        refract_range: 0.5 + random() * 0.25,
        ridges: coinFlip(),
        saturation: 0.175 + random() * 0.25,
        tint_alpha: 0.0125 + random() * 0.0625,
        voronoi_alpha: 0.333 + random() * 0.333,
        voronoi_diagram_type: voronoi.flow,
        voronoi_point_freq: randomInt(3, 5),
        voronoi_refract: 0.25 + random() * 0.125,
      }),
    },

    'periodic-distance': {
      layers: ['basic'],
      settings: () => ({
        freq: randomInt(1, 6),
        distrib: randomMember(
          Object.values(distrib).filter((m) => isCenterDistribution(m))
        ),
        hue_range: 0.25 + random() * 0.125,
      }),
      post: () => [Effect('normalize')],
    },

    'periodic-refract': {
      layers: ['value-refract'],
      settings: () => ({
        value_distrib: randomMember(
          Object.values(distrib).filter(
            (m) => isCenterDistribution(m) || isScan(m)
          )
        ),
      }),
    },

    'pink-diamond': {
      layers: [
        'periodic-distance',
        'periodic-refract',
        'refract-octaves',
        'refract-post',
        'nudge-hue',
        'bloom',
        'lens',
      ],
      settings: () => ({
        color_space: color.hsv,
        bloom_alpha: 0.333 + random() * 0.16667,
        brightness_distrib: distrib.uniform,
        freq: 2,
        hue_range: 0.2 + random() * 0.1,
        hue_rotation: 0.9 + random() * 0.05,
        palette_on: false,
        refract_range: 0.0125 + random() * 0.00625,
        refract_y_from_offset: false,
        ridges: true,
        saturation_distrib: distrib.ones,
        speed: -0.125,
        value_distrib: randomMember(
          Object.values(distrib).filter((m) => isCenterDistribution(m))
        ),
        vaseline_alpha: 0.125 + random() * 0.0625,
      }),
      generator: (settings) => ({
        distrib: settings.value_distrib,
      }),
    },

    'pixel-sort': {
      settings: () => ({
        pixel_sort_angled: coinFlip(),
        pixel_sort_darkest: coinFlip(),
      }),
      final: (settings) => [
        Effect('pixel_sort', {
          angled: settings.pixel_sort_angled,
          darkest: settings.pixel_sort_darkest,
        }),
      ],
    },

    plaid: {
      layers: [
        'multires-low',
        'derivative-octaves',
        'funhouse',
        'maybe-rotate',
        'grain',
        'vignette-dark',
        'saturation',
      ],
      settings: () => ({
        dist_metric: distance.chebyshev,
        distrib: distrib.ones,
        freq: randomInt(2, 4) * 2,
        hue_range: random() * 0.5,
        mask: mask.chess,
        spline_order: randomInt(1, 3),
        vignette_dark_alpha: 0.25 + random() * 0.125,
        warp_freq: randomInt(2, 3),
        warp_range: random() * 0.125,
        warp_octaves: 1,
      }),
    },

    pluto: {
      layers: [
        'multires-ridged',
        'derivative-octaves',
        'voronoi',
        'refract-post',
        'bloom',
        'shadow',
        'contrast-post',
        'grain',
        'saturation',
        'lens',
      ],
      settings: () => ({
        deriv_alpha: 0.333 + random() * 0.16667,
        dist_metric: distance.euclidean,
        distrib: distrib.exp,
        freq: randomInt(4, 8),
        hue_rotation: 0.575,
        octave_blending: blend.reduce_max,
        palette_on: false,
        refract_range: 0.01 + random() * 0.005,
        saturation: 0.75 + random() * 0.25,
        shadow_alpha: 1.0,
        tint_alpha: 0.0125 + random() * 0.00625,
        vignette_dark_alpha: 0.125 + random() * 0.0625,
        voronoi_alpha: 0.925 + random() * 0.075,
        voronoi_diagram_type: voronoi.color_range,
        voronoi_nth: 2,
        voronoi_point_distrib: point.random,
      }),
    },

    'posterize-outline': {
      layers: ['posterize', 'outline'],
    },

    'precision-error': {
      layers: [
        'symmetry',
        'derivative-octaves',
        'reflect-octaves',
        'derivative-post',
        'density-map',
        'invert',
        'shadows',
        'contrast-post',
      ],
      settings: () => ({
        palette_on: false,
        reflect_range: 0.75 + random() * 2.0,
      }),
    },

    'procedural-mask': {
      layers: ['value-mask', 'skew', 'bloom', 'crt', 'vignette-dark', 'contrast-final'],
      settings: () => ({
        spline_order: interp.cosine,
        mask: randomMember(valueMaskProceduralMembers),
        mask_repeat: randomInt(10, 20),
      }),
    },

    prophesy: {
      layers: [
        'value-mask',
        'refract-octaves',
        'posterize',
        'emboss',
        'maybe-invert',
        'tint',
        'shadows',
        'saturation',
        'dexter',
        'texture',
        'maybe-skew',
        'grain',
      ],
      settings: () => ({
        grain_brightness: 0.125,
        grain_contrast: 1.125,
        mask: mask.invaders_square,
        mask_repeat: randomInt(1, 3) * 2,
        octaves: 2,
        palette_on: false,
        posterize_levels: randomInt(4, 8),
        saturation: 0.25 + random() * 0.125,
        spline_order: interp.cosine,
        refract_range: 0.25 + random() * 0.125,
        refract_signed_range: false,
        refract_y_from_offset: true,
        tint_alpha: 0.01 + random() * 0.005,
        vignette_dark_alpha: 0.25 + random() * 0.125,
      }),
    },

    pull: {
      layers: ['basic-voronoi', 'erosion-worms'],
      settings: () => ({
        voronoi_alpha: 0.25 + random() * 0.5,
        voronoi_diagram_type: randomMember([
          voronoi.range,
          voronoi.color_range,
          voronoi.range_regions,
        ]),
      }),
    },

    'pull-quantize': {
      layers: ['pull', 'lens', 'grain'],
      settings: () => ({
        dist_metric: randomMember([distance.manhattan, distance.chebyshev]),
        erosion_worms_alpha: 1.0,
        erosion_worms_quantize: true,
        saturation: 0.0,
        voronoi_point_freq: 3,
        voronoi_alpha: 1.0,
        voronoi_point_distrib: point.random,
      }),
    },

    puzzler: {
      layers: ['basic-voronoi', 'maybe-invert', 'wormhole', 'distressed'],
      settings: () => ({
        speed: 0.025,
        voronoi_diagram_type: voronoi.color_regions,
        voronoi_point_distrib: randomMember(
          point,
          valueMaskNonproceduralMembers
        ),
        voronoi_point_freq: 10,
      }),
    },

    quadrants: {
      layers: ['basic', 'reindex-post'],
      settings: () => ({
        color_space: color.rgb,
        freq: [2, 2],
        reindex_range: 2,
        spline_order: randomMember([interp.cosine, interp.bicubic]),
      }),
    },

    quilty: {
      layers: ['voronoi', 'skew', 'bloom', 'grain'],
      settings: () => ({
        dist_metric: randomMember([distance.manhattan, distance.chebyshev]),
        freq: randomInt(2, 4),
        saturation: random() * 0.5,
        spline_order: interp.constant,
        voronoi_diagram_type: randomMember([voronoi.range, voronoi.color_range]),
        voronoi_nth: randomInt(0, 4),
        voronoi_point_distrib: randomMember(pointGridMembers),
        voronoi_point_freq: randomInt(2, 4),
        voronoi_refract: randomInt(1, 3) * 0.5,
        voronoi_refract_y_from_offset: true,
      }),
    },

    'random-hue': {
      final: () => [Effect('adjust_hue', { amount: random() })],
    },

    rasteroids: {
      layers: [
        'basic',
        'funhouse',
        'sobel',
        'invert',
        'pixel-sort',
        'maybe-rotate',
        'bloom',
        'crt',
        'vignette-dark',
      ],
      settings: () => ({
        distrib: randomMember([distrib.uniform, distrib.ones]),
        freq: 6 * randomInt(2, 3),
        mask: randomMember(mask),
        pixel_sort_angled: false,
        pixel_sort_darkest: false,
        spline_order: interp.constant,
        vignette_dark_alpha: 0.125 + random() * 0.0625,
        warp_freq: randomInt(3, 5),
        warp_octaves: randomInt(3, 5),
        warp_range: 0.125 + random() * 0.0625,
        warp_spline_order: interp.constant,
      }),
    },

    'reflect-octaves': {
      settings: () => ({
        reflect_range: 5 + random() * 0.25,
      }),
      octaves: (settings) => [
        Effect('refract', {
          displacement: settings.reflect_range,
          from_derivative: true,
        }),
      ],
    },

    'reflect-post': {
      settings: () => ({
        reflect_range: 0.5 + random() * 12.5,
      }),
      post: (settings) => [
        Effect('refract', {
          displacement: settings.reflect_range,
          from_derivative: true,
        }),
      ],
    },

    reflecto: {
      layers: ['basic', 'reflect-octaves', 'reflect-post', 'grain'],
    },

    'refract-octaves': {
      settings: () => ({
        refract_range: 0.5 + random() * 0.25,
        refract_signed_range: true,
        refract_y_from_offset: false,
      }),
      octaves: (settings) => [
        Effect('refract', {
          displacement: settings.refract_range,
          signed_range: settings.refract_signed_range,
          y_from_offset: settings.refract_y_from_offset,
        }),
      ],
    },

    'refract-post': {
      settings: () => ({
        refract_range: 0.125 + random() * 1.25,
        refract_signed_range: true,
        refract_y_from_offset: true,
      }),
      post: (settings) => [
        Effect('refract', {
          displacement: settings.refract_range,
          signed_range: settings.refract_signed_range,
          y_from_offset: settings.refract_y_from_offset,
        }),
      ],
    },

    regional: {
      layers: ['voronoi', 'glyph-map', 'bloom', 'crt', 'contrast-post'],
      settings: () => ({
        glyph_map_colorize: coinFlip(),
        glyph_map_zoom: randomInt(4, 8),
        hue_range: 0.25 + random(),
        voronoi_diagram_type: voronoi.color_regions,
        voronoi_nth: 0,
      }),
    },

    'reindex-octaves': {
      settings: () => ({
        reindex_range: 0.125 + random() * 2.5,
      }),
      octaves: (settings) => [
        Effect('reindex', { displacement: settings.reindex_range }),
      ],
    },

    'remember-logo': {
      layers: ['symmetry', 'voronoi', 'derivative-post', 'density-map', 'crt', 'vignette-dark'],
      settings: () => ({
        voronoi_alpha: 1.0,
        voronoi_diagram_type: voronoi.regions,
        voronoi_nth: randomInt(0, 4),
        voronoi_point_distrib: randomMember(pointCircularMembers),
        voronoi_point_freq: randomInt(3, 7),
      }),
    },

    reverb: {
      layers: ['normalize'],
      settings: () => ({
        reverb_iterations: 1,
        reverb_ridges: coinFlip(),
        reverb_octaves: randomInt(3, 6),
      }),
      post: (settings) => [
        Effect('reverb', {
          iterations: settings.reverb_iterations,
          octaves: settings.reverb_octaves,
          ridges: settings.reverb_ridges,
        }),
      ],
    },

    'ride-the-rainbow': {
      layers: ['basic', 'swerve-v', 'scuff', 'distressed', 'contrast-post'],
      settings: () => ({
        brightness_distrib: distrib.ones,
        corners: true,
        distrib: distrib.column_index,
        freq: randomInt(6, 12),
        hue_range: 0.9,
        palette_on: false,
        saturation_distrib: distrib.ones,
        spline_order: interp.constant,
      }),
    },

    ridge: {
      post: () => [Effect('ridge')],
    },

    ripple: {
      settings: () => ({
        ripple_range: 0.025 + random() * 0.1,
        ripple_freq: randomInt(2, 3),
        ripple_kink: randomInt(3, 18),
      }),
      post: (settings) => [
        Effect('ripple', {
          displacement: settings.ripple_range,
          freq: settings.ripple_freq,
          kink: settings.ripple_kink,
        }),
      ],
    },

    rotate: {
      settings: () => ({
        angle: random() * 360.0,
      }),
      post: (settings) => [Effect('rotate', { angle: settings.angle })],
    },

  'runes-of-arecibo': {
      layers: [
        'value-mask',
        'refract-octaves',
        'posterize',
        'emboss',
        'maybe-invert',
        'contrast-post',
        'skew',
        'grain',
        'texture',
        'vaseline',
        'brightness-final',
        'contrast-final',
      ],
      settings: () => ({
        brightness_final: -0.1,
        color_space: color.grayscale,
        corners: true,
        mask: randomMember([
          mask.arecibo_num,
          mask.arecibo_bignum,
          mask.arecibo_nucleotide,
        ]),
        mask_repeat: randomInt(4, 12),
        palette_on: false,
        posterize_levels: randomInt(1, 3),
        refract_range: 0.025 + random() * 0.0125,
        refract_signed_range: false,
        refract_y_from_offset: true,
        spline_order: randomMember([interp.linear, interp.cosine]),
      }),
    },
    'sands-of-time': {
      layers: ['basic', 'worms', 'lens'],
      settings: () => ({
        freq: randomInt(3, 5),
        octaves: randomInt(1, 3),
        worms_behavior: worms.unruly,
        worms_alpha: 1,
        worms_density: 750,
        worms_duration: 0.25,
        worms_kink: randomInt(1, 2),
        worms_stride: randomInt(128, 256),
      }),
    },

    satori: {
      layers: [
        'multires-low',
        'sine-octaves',
        'voronoi',
        'contrast-post',
        'grain',
        'saturation',
      ],
      settings: () => ({
        color_space: randomMember(colorSpaceMembers()),
        dist_metric: randomMember(distanceMetricAbsoluteMembers()),
        freq: randomInt(3, 4),
        hue_range: random(),
        lattice_drift: 1,
        ridges: true,
        speed: 0.05,
        voronoi_alpha: 1.0,
        voronoi_diagram_type: voronoi.flow,
        voronoi_refract: randomInt(6, 12) * 0.25,
        voronoi_point_distrib: randomMember([point.random, ...pointCircularMembers]),
        voronoi_point_freq: randomInt(2, 8),
      }),
    },

    saturation: {
      settings: () => ({
        saturation_final: 0.333 + random() * 0.16667,
      }),
      final: (settings) => [
        Effect('adjust_saturation', { amount: settings.saturation_final }),
      ],
    },

    sblorp: {
      layers: ['basic', 'posterize', 'invert', 'grain', 'saturation'],
      settings: () => ({
        color_space: color.rgb,
        distrib: distrib.ones,
        freq: randomInt(5, 9),
        lattice_drift: 1.25 + random() * 1.25,
        mask: mask.sparse,
        octave_blending: blend.reduce_max,
        octaves: randomInt(2, 3),
        posterize_levels: 1,
      }),
    },

    sbup: {
      layers: ['basic', 'posterize', 'funhouse', 'falsetto', 'palette', 'distressed'],
      settings: () => ({
        distrib: distrib.ones,
        freq: [2, 2],
        mask: mask.square,
        posterize_levels: randomInt(1, 2),
        warp_range: 1.5 + random(),
      }),
    },

    'scanline-error': {
      final: () => [Effect('scanline_error')],
    },

    scratches: {
      final: () => [Effect('scratches')],
    },

    scribbles: {
      layers: [
        'basic',
        'derivative-octaves',
        'derivative-post',
        'derivative-post',
        'contrast-post',
        'invert',
        'sketch',
      ],
      settings: () => ({
        color_space: color.grayscale,
        deriv_alpha: 0.925,
        freq: randomInt(2, 4),
        lattice_drift: 1.0,
        octaves: randomInt(3, 4),
        palette_on: false,
        ridges: true,
      }),
    },

    scuff: {
      final: () => [Effect('scratches')],
    },

    serene: {
      layers: ['basic-water', 'periodic-refract', 'refract-post', 'lens'],
      settings: () => ({
        freq: randomInt(2, 3),
        octaves: 3,
        refract_range: 0.0025 + random() * 0.00125,
        refract_y_from_offset: false,
        value_distrib: distrib.center_circle,
        value_freq: randomInt(2, 3),
        value_refract_range: 0.025 + random() * 0.0125,
        speed: 0.25,
      }),
    },

    shadow: {
      final: () => [Effect('shadow')],
    },

    shadows: {
      final: () => [Effect('shadow')],
    },

    'shake-it-like': {
      layers: [
        'multires-low',
        'falsetto',
        'shake-it',
        'shake-it',
        'worms',
        'distressed',
      ],
      settings: () => ({
        hue_range: random(),
        mask: mask.rings,
        octaves: randomInt(1, 3),
        shake_it_octaves: 1,
        speed: 0.025,
        worms_alpha: 0.5,
        worms_behavior: worms.deviant,
        worms_density: 75,
        worms_duration: 0.5,
        worms_kink: randomInt(4, 8),
        worms_stride: randomInt(50, 100),
      }),
    },

    'shape-party': {
      layers: ['value-mask', 'posterize', 'warp', 'grain', 'saturation'],
      settings: () => ({
        distrib: distrib.uniform,
        mask: randomMember(valueMaskProceduralMembers),
        mask_repeat: randomInt(1, 5),
        palette_on: false,
        posterize_levels: randomInt(1, 2),
        warp_freq: randomInt(2, 4),
        warp_octaves: randomInt(2, 4),
        warp_range: 0.125 + random() * 0.125,
        warp_spline_order: interp.constant,
      }),
    },

    shatter: {
      layers: [
        'multires-low',
        'ghost-diagram',
        'refract-post',
        'refract-post',
        'contrast-post',
        'scanline-error',
        'grain',
      ],
      settings: () => ({
        dist_metric: distance.chebyshev,
        freq: randomInt(4, 6),
        refract_range: 0.1 + random() * 0.05,
        refract_signed_range: true,
        refract_y_from_offset: true,
      }),
    },

    shimmer: {
      layers: ['basic', 'derivative-octaves', 'voronoi', 'refract-post', 'lens'],
      settings: () => ({
        dist_metric: distance.euclidean,
        freq: randomInt(2, 3),
        hue_range: 3.0 + random() * 1.5,
        lattice_drift: 1.0,
        refract_range: 1.25 * random() * 0.625,
        ridges: true,
        voronoi_alpha: 0.25 + random() * 0.125,
        voronoi_diagram_type: voronoi.color_flow,
        voronoi_point_freq: 10,
      }),
    },

    shmoo: {
      layers: [
        'basic',
        'refract-post',
        'contrast-post',
        'shrink-triangulate',
        'bloom',
        'grain',
        'vignette-dark',
      ],
      settings: () => ({
        freq: randomInt(3, 6),
        mask: mask.void,
        refract_range: 0.333 + random() * 0.16667,
        saturation: 0.0,
        spline_order: interp.bicubic,
        vignette_dark_alpha: 0.125 + random() * 0.125,
      }),
    },

    sideways: {
      settings: () => ({
        sideways_displacement: 0.5 + random() * 0.5,
        sideways_freq: [randomInt(2, 5), 1],
        sideways_octaves: 1,
        sideways_spline_order: interp.bicubic,
      }),
      post: (settings) => [
        Effect('warp', {
          displacement: settings.sideways_displacement,
          freq: settings.sideways_freq,
          octaves: settings.sideways_octaves,
          spline_order: settings.sideways_spline_order,
        }),
      ],
    },

    'simple-frame': {
      final: () => [Effect('frame')],
    },

    'sined-multifractal': {
      layers: [
        'sine-octaves',
        'voronoi',
        'refract',
        'posterize',
        'contrast-post',
        'grain',
        'saturation',
      ],
      settings: () => ({
        freq: randomInt(4, 6),
        hue_range: random() * 2.0,
        octaves: randomInt(2, 4),
        posterize_levels: randomInt(8, 24),
        ridges: true,
        sine_range: randomInt(5, 10) * 0.25,
        sine_freq: randomInt(2, 4),
        voronoi_diagram_type: voronoi.range,
        voronoi_point_freq: randomInt(1, 4),
      }),
    },

    'sine-octaves': {
      settings: () => ({
        sine_range: randomInt(1, 4) * 0.125,
        sine_freq: [1, randomInt(1, 4)],
        sine_octaves: randomInt(1, 3),
      }),
      octaves: (settings) => [
        Effect('sine', {
          displacement: settings.sine_range,
          freq: settings.sine_freq,
          octaves: settings.sine_octaves,
        }),
      ],
    },

    'sine-post': {
      settings: () => ({
        sine_post_range: randomInt(1, 4) * 0.125,
        sine_post_freq: [1, randomInt(1, 4)],
        sine_post_octaves: randomInt(1, 3),
      }),
      post: (settings) => [
        Effect('sine', {
          displacement: settings.sine_post_range,
          freq: settings.sine_post_freq,
          octaves: settings.sine_post_octaves,
        }),
      ],
    },

    singularity: {
      layers: [
        'multires',
        'maybe-mask',
        'invert',
        'ghost',
        'vignette-bright',
        'brightness-final',
      ],
      settings: () => ({
        brightness_final: random() * -0.333,
        hue_range: 1.0,
        octaves: randomInt(2, 3),
        palette_on: false,
        value_invert: true,
      }),
    },

    sketch: {
      layers: ['basic', 'symmetry', 'sobel', 'contrast-post', 'sketch'],
      settings: () => ({
        color_space: color.grayscale,
        freq: randomInt(2, 3),
        mask: mask.inside,
        mask_static: true,
        octaves: 2,
        palette_on: false,
      }),
    },

    skew: {
      settings: () => ({
        skew_angle: random() * 360.0,
        skew_range: randomInt(1, 4) * 0.333,
      }),
      post: (settings) => [
        Effect('skew', {
          angle: settings.skew_angle,
          range: settings.skew_range,
        }),
      ],
    },

    smoothstep: {
      settings: () => ({
        smoothstep_low: random(),
        smoothstep_high: random(),
      }),
      post: (settings) => [
        Effect('smoothstep', {
          low: settings.smoothstep_low,
          high: settings.smoothstep_high,
        }),
      ],
    },

    'smoothstep-narrow': {
      settings: () => ({
        smoothstep_narrow_low: random() * 0.05,
        smoothstep_narrow_high: 0.95 + random() * 0.05,
      }),
      post: (settings) => [
        Effect('smoothstep', {
          low: settings.smoothstep_narrow_low,
          high: settings.smoothstep_narrow_high,
        }),
      ],
    },

    'smoothstep-wide': {
      settings: () => ({
        smoothstep_wide_low: random() * 0.25,
        smoothstep_wide_high: 0.75 + random() * 0.25,
      }),
      post: (settings) => [
        Effect('smoothstep', {
          low: settings.smoothstep_wide_low,
          high: settings.smoothstep_wide_high,
        }),
      ],
    },

    snow: {
      layers: ['basic', 'subpixels', 'maybe-invert', 'contrast-post', 'vignette-dark'],
      settings: () => ({
        brightness_distrib: distrib.ones,
        color_space: color.grayscale,
        saturation: 0.0,
      }),
    },

    sobel: {
      settings: () => ({
        sobel_alpha: 1.0,
      }),
      post: (settings) => [
        Effect('sobel', { alpha: settings.sobel_alpha }),
      ],
    },

    'soft-cells': {
      layers: [
        'basic-voronoi',
        'voronoi',
        'emboss',
        'contrast-post',
        'bloom',
        'grain',
        'saturation',
      ],
      settings: () => ({
        dist_metric: distance.manhattan,
        freq: randomInt(5, 7),
        mask: mask.square,
        octaves: randomInt(2, 3),
        palette_on: false,
        spline_order: randomMember([interp.linear, interp.cosine]),
        voronoi_diagram_type: voronoi.color_regions,
        voronoi_nth: randomInt(1, 3),
        voronoi_point_distrib: point.grid,
        voronoi_point_freq: randomInt(3, 4),
        warp_range: 0.125,
      }),
    },

    soup: {
      layers: [
        'multires-low',
        'derivative-octaves',
        'wormhole',
        'posterize-outline',
        'distressed',
      ],
      settings: () => ({
        freq: 2,
        lattice_drift: 0.333,
        octaves: 5,
        speed: 0.05,
        worms_alpha: 1.0,
        worms_behavior: worms.deviant,
        worms_density: 150,
        worms_duration: 0.2,
        worms_kink: 0,
        worms_stride: 25,
      }),
    },

    spaghettification: {
      layers: ['maybe-hyperspace', 'spatter-post', 'vignette-dark'],
      settings: () => ({
        spatter_alpha: random() * 0.5 + 0.25,
        spatter_density: randomInt(1500, 3000),
        spatter_iterations: randomInt(100, 250),
        spatter_kink: random(),
        spatter_signs: true,
        spatter_stride: randomInt(2, 4),
      }),
    },

    spectrogram: {
      layers: ['basic', 'grain', 'filthy'],
      settings: () => ({
        distrib: distrib.row_index,
        freq: randomInt(256, 512),
        hue_range: 0.5 + random() * 0.5,
        mask: mask.bar_code,
        spline_order: interp.constant,
      }),
    },

    'spatter-post': {
      settings: () => ({
        speed: 0.0333 + random() * 0.016667,
        spatter_post_color: true,
      }),
      post: (settings) => [
        Effect('spatter', { color: settings.spatter_post_color }),
      ],
    },

    'spatter-final': {
      settings: () => ({
        speed: 0.0333 + random() * 0.016667,
        spatter_final_color: true,
      }),
      final: (settings) => [
        Effect('spatter', { color: settings.spatter_final_color }),
      ],
    },

    splork: {
      layers: ['voronoi', 'posterize', 'distressed'],
      settings: () => ({
        color_space: color.rgb,
        dist_metric: distance.chebyshev,
        distrib: distrib.ones,
        freq: 33,
        mask: mask.bank_ocr,
        palette_on: true,
        posterize_levels: randomInt(1, 3),
        spline_order: interp.cosine,
        voronoi_diagram_type: voronoi.color_range,
        voronoi_nth: 1,
        voronoi_point_freq: 2,
        voronoi_refract: 0.125,
      }),
    },

    'spooky-ticker': {
      final: () => [Effect('spooky_ticker')],
    },

    'stackin-bricks': {
      layers: ['voronoi'],
      settings: () => ({
        dist_metric: distance.triangular,
        voronoi_diagram_type: voronoi.color_range,
        voronoi_inverse: true,
        voronoi_point_freq: 10,
      }),
    },

    starfield: {
      layers: [
        'multires-low',
        'brightness-post',
        'nebula',
        'contrast-post',
        'lens',
        'grain',
        'vignette-dark',
        'contrast-final',
      ],
      settings: () => ({
        brightness_post: -0.075,
        color_space: color.hsv,
        contrast_post: 2.0,
        distrib: distrib.exp,
        freq: randomInt(400, 500),
        hue_range: 1.0,
        mask: mask.sparser,
        mask_static: true,
        palette_on: false,
        saturation: 0.75,
        spline_order: interp.linear,
      }),
    },

    'stray-hair': {
      final: () => [Effect('stray_hair')],
    },

    'string-theory': {
      layers: ['multires-low', 'erosion-worms', 'bloom', 'lens'],
      settings: () => ({
        color_space: color.rgb,
        erosion_worms_alpha: 0.875 + random() * 0.125,
        erosion_worms_contraction: 4.0 + random() * 2.0,
        erosion_worms_density: 0.25 + random() * 0.125,
        erosion_worms_iterations: randomInt(1250, 2500),
        octaves: randomInt(2, 4),
        palette_on: false,
        ridges: false,
      }),
    },

    subpixelator: {
      layers: ['basic', 'subpixels', 'funhouse'],
      settings: () => ({
        palette_on: false,
      }),
    },

    subpixels: {
      post: () => [
        Effect('glyph_map', {
          mask: randomMember(
            Object.values(mask).filter((m) => isValueMaskRgb(m))
          ),
          zoom: randomMember([8, 16]),
        }),
      ],
    },

    symmetry: {
      layers: ['basic'],
      settings: () => ({
        corners: true,
        freq: [2, 2],
      }),
    },

    'swerve-h': {
      settings: () => ({
        swerve_h_displacement: 0.5 + random() * 0.5,
        swerve_h_freq: [randomInt(2, 5), 1],
        swerve_h_octaves: 1,
        swerve_h_spline_order: interp.bicubic,
      }),
      post: (settings) => [
        Effect('warp', {
          displacement: settings.swerve_h_displacement,
          freq: settings.swerve_h_freq,
          octaves: settings.swerve_h_octaves,
          spline_order: settings.swerve_h_spline_order,
        }),
      ],
    },

    'swerve-v': {
      settings: () => ({
        swerve_v_displacement: 0.5 + random() * 0.5,
        swerve_v_freq: [1, randomInt(2, 5)],
        swerve_v_octaves: 1,
        swerve_v_spline_order: interp.bicubic,
      }),
      post: (settings) => [
        Effect('warp', {
          displacement: settings.swerve_v_displacement,
          freq: settings.swerve_v_freq,
          octaves: settings.swerve_v_octaves,
          spline_order: settings.swerve_v_spline_order,
        }),
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
