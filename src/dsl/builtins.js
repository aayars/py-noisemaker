import * as constants from '../constants.js';
import {
  coinFlip as _coinFlip,
  enumRange as _enumRange,
  randomMember as _randomMember,
  stash as _stash,
} from '../presets.js';
import { random as _random, randomInt as _randomInt } from '../util.js';

export * from '../constants.js';

export const surfaces = Object.freeze({ });

export function coinFlip(...args) {
  if (args.length !== 0) {
    throw new Error(`coinFlip() takes no arguments, received ${args.length}`);
  }
  return _coinFlip();
}

export function enumRange(...args) {
  if (args.length !== 2) {
    throw new Error(`enumRange(a, b) requires exactly 2 arguments, received ${args.length}`);
  }
  const [a, b] = args;
  if (typeof a !== 'number' || typeof b !== 'number') {
    throw new Error('enumRange(a, b) requires numeric arguments');
  }
  return _enumRange(a, b);
}

export function randomMember(...collections) {
  if (collections.length === 0) {
    throw new Error('randomMember() requires at least one iterable argument');
  }
  return _randomMember(...collections);
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

export function randomInt(...args) {
  if (args.length !== 2) {
    throw new Error(`randomInt(a, b) requires exactly 2 arguments, received ${args.length}`);
  }
  const [a, b] = args;
  if (typeof a !== 'number' || typeof b !== 'number') {
    throw new Error('randomInt(a, b) requires numeric arguments');
  }
  return _randomInt(a, b);
}

export const operations = Object.freeze({
  coinFlip,
  randomMember,
  enumRange,
  stash,
  random,
  randomInt,
});

export const enums = constants;

export const defaultContext = {
  surfaces,
  operations,
  enums,
};
