import * as constants from '../constants.js';
import {
  coin_flip as _coin_flip,
  enum_range as _enum_range,
  random_member as _random_member,
  stash as _stash,
} from '../presets.js';
import { random as _random, randomInt as _randomInt } from '../util.js';

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
  return _randomInt(a, b);
}

export const operations = Object.freeze({
  coin_flip,
  random_member,
  enum_range,
  stash,
  random,
  random_int,
});

export const enums = constants;

export const defaultContext = {
  surfaces,
  operations,
  enums,
};
