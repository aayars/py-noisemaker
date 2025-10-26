import * as constants from '../constants.js';
import { PALETTES as _PALETTES } from '../palettes.js';
import {
  coin_flip as _coin_flip,
  enum_range as _enum_range,
  random_member as _random_member,
  stash as _stash,
  PRESETS as _PRESETS,
} from '../presets.js';
import { Preset as _Preset } from '../composer.js';
import { random as _random, randomInt as _random_int } from '../util.js';
import { random as _rngRandom } from '../rng.js';
import { maskShape as _maskShape, squareMasks as _squareMasks } from '../masks.js';

export * from '../constants.js';

const settingsProxy = new Proxy(
  {},
  {
    get: (_, prop) => (settings) => settings[prop],
  },
);

export const surfaces = Object.freeze({ settings: settingsProxy });

export function coin_flip(...args) {
  if (args.length !== 0) {
    throw new Error(`coin_flip() takes no arguments, received ${args.length}`);
  }
  return _coin_flip();
}

export function enum_range(...args) {
  if (args.length !== 2) {
    throw new Error(`enum_range(a, b) requires exactly 2 arguments, received ${args.length}`);
  }
  const [a, b] = args;
  if (typeof a !== 'number' || typeof b !== 'number') {
    throw new Error('enum_range(a, b) requires numeric arguments');
  }
  return _enum_range(a, b);
}

export function random_member(...collections) {
  if (collections.length === 0) {
    throw new Error('random_member() requires at least one iterable argument');
  }
  return _random_member(...collections);
}

export function stash(...args) {
  if (args.length === 0 || args.length > 2) {
    throw new Error(`stash(key[, value]) expects 1 or 2 arguments, received ${args.length}`);
  }
  const [key, value] = args;
  if (typeof key !== 'string') {
    throw new Error('stash(key[, value]) key must be a string');
  }
  return (settings = {}) =>
    _stash(key, typeof value === 'function' ? value(settings) : value);
}

export function random(...args) {
  if (args.length !== 0) {
    throw new Error(`random() takes no arguments, received ${args.length}`);
  }
  return _random();
}

export function random_int(...args) {
  if (args.length !== 2) {
    throw new Error(`random_int(a, b) requires exactly 2 arguments, received ${args.length}`);
  }
  const [a, b] = args;
  if (typeof a !== 'number' || typeof b !== 'number') {
    throw new Error('random_int(a, b) requires numeric arguments');
  }
  return _random_int(a, b);
}

coin_flip.__thunk = true;
random_member.__thunk = true;
random.__thunk = true;
random_int.__thunk = true;

export function mask_freq(...args) {
  if (args.length !== 2) {
    throw new Error(`mask_freq(mask, repeat) requires exactly 2 arguments, received ${args.length}`);
  }
  const [mask, repeat] = args;
  const [h, w] = _maskShape(mask);
  return [Math.floor(h * 0.5 + h * repeat), Math.floor(w * 0.5 + w * repeat)];
}

export function preset(...args) {
  if (args.length === 0 || args.length > 2) {
    throw new Error(`preset(name[, settings]) expects 1 or 2 arguments, received ${args.length}`);
  }
  const [name, settings = {}] = args;
  if (typeof name !== 'string') {
    throw new Error('preset(name[, settings]) name must be a string');
  }
  // Return a thunk that will instantiate the preset when invoked. This
  // avoids recursive construction while the preset table itself is being
  // evaluated from the DSL. The thunk accepts parent settings so nested
  // presets can resolve dynamic values.
  return (parentSettings = {}) => {
    const resolved = {};
    for (const [k, v] of Object.entries(settings)) {
      resolved[k] = typeof v === 'function' ? v(parentSettings) : v;
    }
    const presets = _PRESETS();
    // Python's DSL ``preset()`` helper routes through ``noisemaker.presets.Preset``,
    // which rebuilds the preset table when instantiating nested presets. Consume
    // the same three RNG samples here to mirror that extra ``PRESETS()`` call
    // without re-evaluating the table in JavaScript.
    _rngRandom();
    _rngRandom();
    _rngRandom();
    return new _Preset(name, presets, resolved);
  };
}

export const operations = Object.freeze({
  coin_flip,
  random_member,
  enum_range,
  stash,
  random,
  random_int,
  mask_freq,
  preset,
  // expose helper functions used via enum method-style calls
  distance_metric_absolute_members: constants.distanceMetricAbsoluteMembers,
  distance_metric_all: constants.distanceMetricAll,
  color_space_members: constants.colorSpaceMembers,
  value_mask_procedural_members: constants.valueMaskProceduralMembers,
  value_mask_grid_members: constants.valueMaskGridMembers,
  value_mask_glyph_members: constants.valueMaskGlyphMembers,
  value_mask_nonprocedural_members: constants.valueMaskNonproceduralMembers,
  value_mask_rgb_members: constants.valueMaskRgbMembers,
  circular_members: constants.circularMembers,
  grid_members: constants.gridMembers,
  worm_behavior_all: constants.wormBehaviorAll,
  mask_shape: _maskShape,
  square_masks: _squareMasks,
});

export const enums = { ...constants, PALETTES: _PALETTES };

// Map enum/object method names used in the DSL to functions exposed in
// `operations`.  This allows expressions like `DistanceMetric.absolute_members()`
// to resolve to the appropriate helper without mutating the frozen enum objects.
export const enumMethods = Object.freeze({
  DistanceMetric: {
    absolute_members: operations.distance_metric_absolute_members,
    all: operations.distance_metric_all,
  },
  PointDistribution: {
    grid_members: () => operations.grid_members,
    circular_members: () => operations.circular_members,
  },
  ColorSpace: {
    color_members: operations.color_space_members,
  },
  ValueMask: {
    procedural_members: () => operations.value_mask_procedural_members,
    grid_members: () => operations.value_mask_grid_members,
    glyph_members: () => operations.value_mask_glyph_members,
    nonprocedural_members: () => operations.value_mask_nonprocedural_members,
    rgb_members: () => operations.value_mask_rgb_members,
  },
  WormBehavior: {
    all: () => operations.worm_behavior_all,
  },
  masks: {
    mask_shape: operations.mask_shape,
    square_masks: operations.square_masks,
  },
});

export const defaultContext = {
  surfaces,
  operations,
  enums,
  enumMethods,
};
