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
coin_flip.__thunk = true;

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
random_member.__thunk = true;

export function stash(...args) {
  if (args.length === 0 || args.length > 2) {
    throw new Error(`stash(key[, value]) expects 1 or 2 arguments, received ${args.length}`);
  }
  const [key, value] = args;
  if (typeof key !== 'string') {
    throw new Error('stash(key[, value]) key must be a string');
  }
  return _stash(key, value);
}

export function random(...args) {
  if (args.length !== 0) {
    throw new Error(`random() takes no arguments, received ${args.length}`);
  }
  return _random();
}
random.__thunk = true;

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
    return new _Preset(name, _PRESETS(), resolved);
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
  distanceMetricAbsoluteMembers: constants.distanceMetricAbsoluteMembers,
  distanceMetricAll: constants.distanceMetricAll,
  colorSpaceMembers: constants.colorSpaceMembers,
  valueMaskProceduralMembers: constants.valueMaskProceduralMembers,
  valueMaskGridMembers: constants.valueMaskGridMembers,
  valueMaskGlyphMembers: constants.valueMaskGlyphMembers,
  valueMaskNonproceduralMembers: constants.valueMaskNonproceduralMembers,
  valueMaskRgbMembers: constants.valueMaskRgbMembers,
  circularMembers: constants.circularMembers,
  gridMembers: constants.gridMembers,
  wormBehaviorAll: constants.wormBehaviorAll,
  maskShape: _maskShape,
  squareMasks: _squareMasks,
});

export const enums = { ...constants, PALETTES: _PALETTES };

// Map enum/object method names used in the DSL to functions exposed in
// `operations`.  This allows expressions like `DistanceMetric.absolute_members()`
// to resolve to the appropriate helper without mutating the frozen enum objects.
export const enumMethods = Object.freeze({
  DistanceMetric: {
    absolute_members: operations.distanceMetricAbsoluteMembers,
    all: operations.distanceMetricAll,
  },
  PointDistribution: {
    grid_members: () => operations.gridMembers,
    circular_members: () => operations.circularMembers,
  },
  ColorSpace: {
    color_members: operations.colorSpaceMembers,
  },
  ValueMask: {
    procedural_members: () => operations.valueMaskProceduralMembers,
    grid_members: () => operations.valueMaskGridMembers,
    glyph_members: () => operations.valueMaskGlyphMembers,
    nonprocedural_members: () => operations.valueMaskNonproceduralMembers,
    rgb_members: () => operations.valueMaskRgbMembers,
  },
  WormBehavior: {
    all: () => operations.wormBehaviorAll,
  },
  masks: {
    mask_shape: operations.maskShape,
    square_masks: operations.squareMasks,
  },
});

export const defaultContext = {
  surfaces,
  operations,
  enums,
  enumMethods,
};
