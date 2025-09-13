import { parsePresetDSL } from './dsl/index.js';
import { random, randomInt, setSeed, getSeed } from './util.js';
import { Effect } from './composer.js';
import * as constants from './constants.js';
export { setSeed };

// Precompute enum value lookups so that arrays of enum numeric values can be
// sorted deterministically by their enum member names, mirroring Python's
// Enum sorting behaviour.
const ENUM_LOOKUPS = [];
for (const obj of Object.values(constants)) {
  if (
    obj &&
    typeof obj === 'object' &&
    !Array.isArray(obj) &&
    Object.values(obj).every((v) => typeof v === 'number')
  ) {
    const values = new Set(Object.values(obj));
    const nameMap = new Map(Object.entries(obj).map(([k, v]) => [v, k]));
    ENUM_LOOKUPS.push({ values, nameMap });
  }
}

export function coin_flip() {
  return randomInt(0, 1) === 1;
}

export function enum_range(a, b) {
  const out = [];
  for (let i = a; i <= b; i++) {
    out.push(i);
  }
  return out;
}

export function random_member(...collections) {
  const out = [];
  for (const c of collections) {
    if (Array.isArray(c)) {
      const arr = c.slice();
      sortArray(arr);
      out.push(...arr);
    } else if (c && typeof c === 'object' && !(c instanceof Map)) {
      const keys = Object.keys(c).sort();
      const values = keys.map((k) => c[k]);
      const primitives = values.every(
        (v) => v === null || ['number', 'string', 'boolean'].includes(typeof v)
      );
      if (primitives) {
        out.push(...values);
      } else {
        out.push(...keys);
      }
    } else if (c && typeof c[Symbol.iterator] === 'function') {
      const arr = Array.from(c);
      sortArray(arr);
      out.push(...arr);
    } else {
      throw new Error('random_member(arg) should be iterable');
    }
  }
  if (out.every((v) => typeof v === 'boolean')) {
    out.sort((a, b) => Number(a) - Number(b));
  }
  const idx = Math.floor(random() * out.length);
  return out[idx];
}

function sortArray(arr) {
  let selected = null;
  for (const { values, nameMap } of ENUM_LOOKUPS) {
    if (arr.every((v) => values.has(v))) {
      if (!selected || values.size < selected.size) {
        selected = { map: nameMap, size: values.size };
      }
    }
  }
  arr.sort((a, b) => {
    if (selected) {
      const na = selected.map.get(a);
      const nb = selected.map.get(b);
      return na < nb ? -1 : na > nb ? 1 : 0;
    }
    const ta = typeof a;
    const tb = typeof b;
    if (ta === 'number' && tb === 'number') {
      return a - b;
    }
    const sa = String(a);
    const sb = String(b);
    return sa < sb ? -1 : sa > sb ? 1 : 0;
  });
}

const _STASH = new Map();
export function stash(key, value) {
  if (value !== undefined) {
    _STASH.set(key, value);
  }
  return _STASH.get(key);
}

export function mapEffect(e, settings) {
  const seen = new Set();
  let depth = 0;
  while (
    typeof e === 'function' &&
    !e.__effectName &&
    !e.post_effects &&
    !e.final_effects
  ) {
    if (seen.has(e) || depth++ > 64) {
      throw new Error('Runaway dynamic preset function');
    }
    seen.add(e);
    e = e(settings);
  }
  if (typeof e === 'function') {
    return e;
  }
  if (e && typeof e === 'object' && e.__effectName) {
    const params = {};
    if (e.__paramNames) {
      for (const k of e.__paramNames) {
        const v = e.__params[k];
        params[k] = typeof v === 'function' ? v(settings) : v;
      }
    }
    return Effect(e.__effectName, params);
  }
  return e;
}

let _SOURCE;
{
  const url = new URL('../dsl/presets.dsl', import.meta.url);
  if (typeof process !== 'undefined' && process.versions && process.versions.node) {
    const fs = await import('node:fs/promises');
    _SOURCE = await fs.readFile(url, 'utf8');
  } else {
    const res = await fetch(url);
    _SOURCE = await res.text();
  }
}

function buildPresets(names) {
  // Preserve the caller's RNG state. The DSL parser currently uses the
  // seeded RNG during parsing, so parse using a fixed seed and then restore
  // the original seed to avoid consuming random values that would affect
  // downstream preset evaluation.
  const seedBefore = getSeed();
  setSeed(0);
  const parsed = parsePresetDSL(_SOURCE);
  setSeed(seedBefore);

  // Python's PRESETS() advances the RNG three times when invoked. Advance the
  // RNG equivalently here to maintain parity.
  random();
  random();
  random();
  const presets = {};

  function build(name) {
    if (presets[name]) return presets[name];
    const preset = parsed[name];
    if (!preset) return;

    // Recursively build any layer references that are also presets.
    if (Array.isArray(preset.layers)) {
      for (const layer of preset.layers) {
        if (typeof layer === 'string' && parsed[layer]) {
          build(layer);
        }
      }
    }

    const p = { ...preset };
    if (p.settings && typeof p.settings === 'object') {
      const s = p.settings;
      p.settings = () => ({ ...s });
    }
    if (p.generator && typeof p.generator === 'object') {
      const g = p.generator;
      p.generator = (settings) => {
        const out = {};
        for (const [k, v] of Object.entries(g)) {
          out[k] = typeof v === 'function' ? v(settings) : v;
        }
        return out;
      };
    }
    if (Array.isArray(p.octaves)) {
      const o = p.octaves;
      p.octaves = (settings) => o.map((e) => mapEffect(e, settings));
    } else if (typeof p.octaves === 'function') {
      const ofn = p.octaves;
      p.octaves = (settings) => {
        const arr = ofn(settings);
        return Array.isArray(arr) ? arr.map((e) => mapEffect(e, settings)) : arr;
      };
    }
    if (Array.isArray(p.post)) {
      const post = p.post;
      p.post = (settings) => post.map((e) => mapEffect(e, settings));
    } else if (typeof p.post === 'function') {
      const pfn = p.post;
      p.post = (settings) => {
        const arr = pfn(settings);
        return Array.isArray(arr) ? arr.map((e) => mapEffect(e, settings)) : arr;
      };
    }
    if (Array.isArray(p.final)) {
      const fin = p.final;
      p.final = (settings) => fin.map((e) => mapEffect(e, settings));
    } else if (typeof p.final === 'function') {
      const ffn = p.final;
      p.final = (settings) => {
        const arr = ffn(settings);
        return Array.isArray(arr) ? arr.map((e) => mapEffect(e, settings)) : arr;
      };
    }

    presets[name] = p;
    return p;
  }

  if (!names) {
    names = Object.keys(parsed);
  } else if (!Array.isArray(names)) {
    names = [names];
  }

  for (const name of names) {
    build(name);
  }

  return presets;
}

export function PRESETS(names) {
  return buildPresets(names);
}

export default PRESETS;

