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
