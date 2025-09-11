import * as constants from '../constants.js';
import {
  coin_flip as _coin_flip,
  enum_range as _enum_range,
  random_member as _random_member,
  stash as _stash,
  PRESETS as _PRESETS,
} from '../presets.js';
import { Preset as _Preset } from '../composer.js';
import { random as _random, randomInt as _random_int } from '../util.js';
import { maskShape as _maskShape } from '../masks.js';

export * from '../constants.js';

export const surfaces = Object.freeze({ });

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
  return _stash(key, value);
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
  return new _Preset(name, _PRESETS(), settings);
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
});

export const enums = constants;

export const defaultContext = {
  surfaces,
  operations,
  enums,
};
