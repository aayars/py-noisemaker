import { parsePresetDSL } from './dsl/index.js';
import { random, randomInt, setSeed, getSeed } from './util.js';
import { Effect } from './composer.js';
import * as constants from './constants.js';
export { setSeed };

// Precompute enum value lookups so that arrays of enum numeric values can be
// sorted deterministically by their enum member names, mirroring Python's
// Enum sorting behaviour.
const ENUM_LOOKUPS = [];
for (const [name, obj] of Object.entries(constants)) {
  if (
    obj &&
    typeof obj === 'object' &&
    !Array.isArray(obj) &&
    Object.values(obj).every((v) => typeof v === 'number')
  ) {
    const values = new Set(Object.values(obj));
    const nameMap = new Map(Object.entries(obj).map(([k, v]) => [v, k]));
    ENUM_LOOKUPS.push({ name, values, nameMap });
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
  if (!Object.prototype.hasOwnProperty.call(out, '__literal')) {
    Object.defineProperty(out, '__literal', { value: true });
  }
  return out;
}

export function random_member(...collections) {
  const out = [];
  for (const c of collections) {
    if (Array.isArray(c)) {
      const arr = c.slice();
      if (c.__enum && !Object.prototype.hasOwnProperty.call(arr, '__enum')) {
        Object.defineProperty(arr, '__enum', { value: c.__enum });
      }
      if (c.__literal && !Object.prototype.hasOwnProperty.call(arr, '__literal')) {
        Object.defineProperty(arr, '__literal', { value: c.__literal });
      }
      let enumName = arr.__enum || null;
      if (!enumName && !arr.__literal) {
        for (const lookup of ENUM_LOOKUPS) {
          if (arr.every((item) => lookup.values.has(item))) {
            enumName = lookup.name;
            break;
          }
        }
      }
      if (enumName && !Object.prototype.hasOwnProperty.call(arr, '__enum')) {
        Object.defineProperty(arr, '__enum', { value: enumName });
      }
      sortArray(arr);
      out.push(...arr);
    } else if (c && typeof c === 'object' && !(c instanceof Map)) {
      const entries = Object.entries(c);
      const values = entries.map(([, v]) => v);
      const primitives = values.every(
        (v) => v === null || ['number', 'string', 'boolean'].includes(typeof v),
      );
      if (primitives) {
        const enumLookup = ENUM_LOOKUPS.find((lookup) =>
          values.every((item) => lookup.values.has(item)),
        );
        if (enumLookup) {
          for (const [, value] of entries) {
            out.push(value);
          }
        } else {
          const keys = entries.map(([k]) => k).sort();
          for (const key of keys) {
            out.push(c[key]);
          }
        }
      } else {
        const keys = entries.map(([k]) => k).sort();
        out.push(...keys);
      }
    } else if (c && typeof c[Symbol.iterator] === 'function') {
      const arr = Array.from(c);
      if (c.__enum && !Object.prototype.hasOwnProperty.call(arr, '__enum')) {
        Object.defineProperty(arr, '__enum', { value: c.__enum });
      }
      if (c.__literal && !Object.prototype.hasOwnProperty.call(arr, '__literal')) {
        Object.defineProperty(arr, '__literal', { value: c.__literal });
      }
      let enumName = arr.__enum || null;
      if (!enumName && !arr.__literal) {
        for (const lookup of ENUM_LOOKUPS) {
          if (arr.every((item) => lookup.values.has(item))) {
            enumName = lookup.name;
            break;
          }
        }
      }
      if (enumName && !Object.prototype.hasOwnProperty.call(arr, '__enum')) {
        Object.defineProperty(arr, '__enum', { value: enumName });
      }
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
  if (arr.__enum) {
    selected = ENUM_LOOKUPS.find((e) => e.name === arr.__enum) || null;
  }
  if (selected) {
    arr.sort((a, b) => {
      const na = selected.nameMap.get(a);
      const nb = selected.nameMap.get(b);
      return na < nb ? -1 : na > nb ? 1 : 0;
    });
  } else {
    arr.sort((a, b) => {
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

function resolveSettingValue(value, settings) {
  if (typeof value === 'function') {
    try {
      const arity = value.length;
      if (arity === 0) return resolveSettingValue(value(), settings);
      if (arity === 1) return resolveSettingValue(value(settings), settings);
      return value;
    } catch {
      return value;
    }
  }
  if (Array.isArray(value)) return value.map((v) => resolveSettingValue(v, settings));
  if (value && typeof value === 'object') {
    const out = {};
    for (const [k, v] of Object.entries(value)) {
      out[k] = resolveSettingValue(v, settings);
    }
    return out;
  }
  return value;
}

function evaluateSettings(template) {
  const resolved = {};
  for (const [key, value] of Object.entries(template)) {
    resolved[key] = resolveSettingValue(value, resolved);
  }
  return resolved;
}

let _SOURCE;
if (typeof process !== 'undefined' && process.env?.NOISEMAKER_EMBEDDED_DSL) {
  // Embedded DSL for SEA runtime
  _SOURCE = Buffer.from(process.env.NOISEMAKER_EMBEDDED_DSL, 'base64').toString('utf-8');
} else if (typeof NOISEMAKER_PRESETS_DSL !== 'undefined') {
  // Bundled DSL (browser or esbuild define)
  _SOURCE = NOISEMAKER_PRESETS_DSL;
} else {
  // Load from file system (Node.js ESM only - requires top-level await support)
  // This path should never execute in bundled/SEA builds
  const fs = (await import('node:fs')).default || (await import('node:fs'));
  const url = new URL('../../dsl/presets.dsl', import.meta.url);
  _SOURCE = fs.readFileSync(url, 'utf8');
}

function buildPresets(names) {
  // The reference Python implementation builds the preset table without
  // consuming any RNG state. Parse the DSL using a fixed seed and then restore
  // the caller's seed so downstream random sequences remain unchanged.
  const seedBefore = getSeed();
  setSeed(0);
  const parsed = parsePresetDSL(_SOURCE);
  const presets = {};

  function build(name) {
    if (presets[name]) return presets[name];
    const preset = parsed[name];
    if (!preset) return;

    let resolvedLayers = null;
    if (preset.layers !== undefined) {
      resolvedLayers = resolveSettingValue(preset.layers, Object.create(null));
      if (Array.isArray(resolvedLayers)) {
        // Recursively build any layer references that are also presets.
        for (const layer of resolvedLayers) {
          if (typeof layer === 'string' && parsed[layer]) {
            build(layer);
          }
        }
      }
    }

    const p = { ...preset };
    if (resolvedLayers !== null) {
      // Ensure downstream consumers always see a concrete array of layer names,
      // matching the behaviour of the reference Python implementation where
      // dynamic layer expressions are resolved during preset table
      // construction.
      if (Array.isArray(resolvedLayers)) {
        p.layers = resolvedLayers.slice();
      } else {
        p.layers = resolvedLayers;
      }
    } else if (typeof p.layers === 'function') {
      // Allow layers to be specified as a function in the DSL. Evaluate the
      // function once when building the preset so downstream consumers always
      // see a concrete array of layer names, matching the Python behaviour.
      p.layers = p.layers();
    }
    if (p.settings && typeof p.settings === 'object') {
      const s = p.settings;
      p.settings = () => evaluateSettings(s);
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

  setSeed(seedBefore);
  // The Python preset table construction advances RNG three times for dynamic
  // layer expressions even though the resulting values are unused when only a
  // single preset is requested. Advance the RNG here to keep sequences aligned.
  random();
  random();
  random();

  return presets;
}

export function PRESETS(names) {
  return buildPresets(names);
}

export default PRESETS;

