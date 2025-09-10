import * as constants from '../constants.js';
import { random } from '../util.js';

export const surfaces = Object.freeze({
  synth1: 'synth1',
  synth2: 'synth2',
  mixer: 'mixer',
  post1: 'post1',
  post2: 'post2',
  post3: 'post3',
  final: 'final',
});

function coinFlip() {
  return random() < 0.5;
}

function enumRange(a, b) {
  const out = [];
  for (let i = a; i <= b; i++) {
    out.push(i);
  }
  return out;
}

function randomMember(...collections) {
  const out = [];
  for (const c of collections) {
    if (Array.isArray(c)) {
      out.push(...c.slice().sort());
    } else if (c && typeof c === 'object' && !(c instanceof Map)) {
      const keys = Object.keys(c).sort();
      const values = keys.map((k) => c[k]);
      const primitives = values.every(
        (v) => v === null || ['number', 'string', 'boolean'].includes(typeof v),
      );
      if (primitives) {
        out.push(...values);
      } else {
        out.push(...keys);
      }
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
function stash(key, value) {
  if (value !== undefined) {
    _STASH.set(key, value);
  }
  return _STASH.get(key);
}

export const operations = Object.freeze({
  coinFlip,
  randomMember,
  enumRange,
  stash,
});

export const enums = constants;

export const defaultContext = {
  surfaces,
  operations,
  enums,
};
