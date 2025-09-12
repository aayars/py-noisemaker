import { parsePresetDSL } from './dsl/index.js';
import { random, getSeed, setSeed } from './util.js';
import { Effect } from './composer.js';
export { setSeed };

export function coin_flip() {
  return random() < 0.5;
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
      out.push(...c.slice().sort());
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
      arr.sort();
      out.push(...arr);
    } else {
      throw new Error('random_member(arg) should be iterable');
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

let _PRESETS;
{
  const url = new URL('../dsl/presets.dsl', import.meta.url);
  let source;
  if (typeof process !== 'undefined' && process.versions && process.versions.node) {
    const fs = await import('node:fs/promises');
    source = await fs.readFile(url, 'utf8');
  } else {
    const res = await fetch(url);
    source = await res.text();
  }
  const seed = getSeed();
  const parsed = parsePresetDSL(source);
  _PRESETS = {};
  for (const [name, preset] of Object.entries(parsed)) {
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
      const mapEffect = (e, settings) => {
        while (
          typeof e === 'function' &&
          !e.__effectName &&
          !e.post_effects &&
          !e.final_effects
        ) {
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
      };
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
    _PRESETS[name] = p;
  }
  setSeed(seed);
}

export function PRESETS() {
  return _PRESETS;
}

export default PRESETS;

