import { Context } from './context.js';
import { FULLSCREEN_VS } from './value.js';
import { multires } from './generators.js';
import { ColorSpace } from './constants.js';
import { shapeFromParams } from './util.js';
import { EFFECTS } from './effectsRegistry.js';
import { SettingsDict } from './settings.js';

const SETTINGS_KEY = 'settings';
const ALLOWED_KEYS = ['layers', SETTINGS_KEY, 'generator', 'octaves', 'post', 'final', 'ai', 'unique'];
const UNUSED_OKAY = [
  'ai',
  'angle',
  'paletteAlpha',
  'paletteName',
  'speed',
  'voronoiSdfSides',
  'voronoiInverse',
  'voronoiRefract',
  'voronoiRefractYFromOffset',
];

function toCamel(str) {
  return str.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
}

function toCamelKeys(obj = {}) {
  const out = {};
  for (const [k, v] of Object.entries(obj)) {
    out[toCamel(k)] = v;
  }
  return out;
}

export class Preset {
  constructor(presetName, presets, settings = {}) {
    this.name = presetName;
    const prototype = presets[presetName];
    if (!prototype) {
      throw new Error(`Preset "${presetName}" was not found among the available presets.`);
    }
    if (typeof prototype !== 'object') {
      throw new Error(`Preset "${presetName}" should be an object, not "${typeof prototype}"`);
    }
    for (const key of Object.keys(prototype)) {
      if (!ALLOWED_KEYS.includes(key)) {
        throw new Error(
          `Preset "${presetName}": Key "${key}" is not permitted. Allowed keys are: ${ALLOWED_KEYS}`
        );
      }
    }
    this.layers = prototype.layers || [];
    this.flattened_layers = [];
    _flattenAncestors(presetName, presets, {}, this.flattened_layers);

    this.settings = new SettingsDict(
      _flattenAncestorMetadata(this, null, SETTINGS_KEY, {}, presets)
    );
    Object.assign(this.settings, toCamelKeys(settings));

    this.generator = _flattenAncestorMetadata(this, this.settings, 'generator', {}, presets);
    this.octave_effects = _flattenAncestorMetadata(this, this.settings, 'octaves', [], presets);
    this.post_effects = _flattenAncestorMetadata(this, this.settings, 'post', [], presets);
    this.final_effects = _flattenAncestorMetadata(this, this.settings, 'final', [], presets);

    try {
      this.settings.raiseIfUnaccessed(UNUSED_OKAY);
    } catch (e) {
      throw new Error(`Preset "${presetName}": ${e.message}`);
    }
  }

  render(seed = 0, opts = {}) {
    opts = toCamelKeys(opts);
    const {
      ctx = new Context(null),
      width = 256,
      height = 256,
      time = 0,
      speed = 1,
      withAlpha = false,
    } = opts;

    const g = this.generator || {};
    const colorSpace =
      g.color_space ?? g.colorSpace ??
      this.settings.color_space ?? this.settings.colorSpace ??
      ColorSpace.hsv;
    const shape = shapeFromParams(
      width,
      height,
      colorSpace === ColorSpace.grayscale ? 'grayscale' : 'rgb',
      withAlpha
    );
    const freq = g.freq !== undefined ? g.freq : 1;
    let tensor = multires(freq, shape, {
      ...g,
      color_space: colorSpace,
      ctx,
      seed,
      time,
      speed,
      octaveEffects: this.octave_effects,
      postEffects: this.post_effects,
      finalEffects: this.final_effects,
    });

    // Present to canvas if available
    if (ctx.canvas) {
      const [h, w, c] = tensor.shape;
      if (ctx.gl && !ctx.isCPU) {
        const gl = ctx.gl;
        ctx.canvas.width = w;
        ctx.canvas.height = h;
        const fs = `#version 300 es\nprecision highp float;\nuniform sampler2D u_tex;\nout vec4 outColor;\nvoid main(){\n vec2 uv = vec2(gl_FragCoord.x / ${w}.0, 1.0 - gl_FragCoord.y / ${h}.0);\n vec4 color = texture(u_tex, uv);\n outColor = ${c > 3 ? 'color' : 'vec4(color.rgb, 1.0)'};\n}`;
        const prog = ctx.createProgram(FULLSCREEN_VS, fs);
        gl.useProgram(prog);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, tensor.handle);
        gl.uniform1i(gl.getUniformLocation(prog, 'u_tex'), 0);
        ctx.bindFramebuffer(null, w, h);
        ctx.drawQuad();
        gl.bindTexture(gl.TEXTURE_2D, null);
        gl.deleteProgram(prog);
      } else if (ctx.canvas.getContext) {
        const data = tensor.read();
        const ctx2d = ctx.canvas.getContext('2d', { willReadFrequently: true });
        if (ctx2d) {
          ctx.canvas.width = w;
          ctx.canvas.height = h;
          const img = ctx2d.createImageData(w, h);
          for (let i = 0; i < h * w; i++) {
            const src = i * c;
            const r = data[src];
            const g = c > 2 ? data[src + 1] : r;
            const b = c > 2 ? data[src + 2] : r;
            const aVal = c > 3 ? data[src + 3] : c === 2 ? data[src + 1] : 1;
            const a = Number.isFinite(aVal) ? aVal : 1;
            const base = i * 4;
            img.data[base] = Math.max(0, Math.min(255, Math.round(r * 255)));
            img.data[base + 1] = Math.max(0, Math.min(255, Math.round(g * 255)));
            img.data[base + 2] = Math.max(0, Math.min(255, Math.round(b * 255)));
            img.data[base + 3] = Math.max(0, Math.min(255, Math.round(a * 255)));
          }
          ctx2d.putImageData(img, 0, 0);
        }
      }
    }

    return tensor;
  }
}

export function Effect(effectName, params = {}) {
  const effect = EFFECTS[effectName];
  if (!effect) {
    throw new Error(`"${effectName}" is not a registered effect name.`);
  }
  const keys = Object.keys(effect).filter((k) => k !== 'func');

  // Normalize parameter keys: convert snake_case to camelCase and apply effect-specific aliases
  const aliasMap = {
    sine: { displacement: 'amount' },
    smoothstep: { low: 'a', high: 'b' },
  };

  const mapped = {};
  for (const [k, v] of Object.entries(params)) {
    const camel = toCamel(k);
    const finalKey = aliasMap[effectName]?.[camel] || camel;
    mapped[finalKey] = v;
  }

  for (const k of Object.keys(mapped)) {
    if (!keys.includes(k)) {
      throw new Error(`Effect "${effectName}" does not accept a parameter named "${k}"`);
    }
  }

  const applied = {};
  for (const k of keys) {
    applied[k] = mapped[k] !== undefined ? mapped[k] : effect[k];
  }

  const fn = function (tensor, shape, time, speed) {
    const args = keys.map((k) => applied[k]);
    return effect.func(tensor, shape, time, speed, ...args);
  };
  fn.__effectName = effectName;
  fn.__paramNames = keys;
  fn.__params = { ...applied };
  return fn;
}

export function render(presetName, seed = 0, opts = {}) {
  const { presets = {}, settings } = opts;
  const preset =
    typeof presetName === 'string'
      ? new Preset(presetName, presets, settings)
      : presetName;
  return preset.render(seed, opts);
}

function _flattenAncestors(presetName, presets, unique, ancestors) {
  const layers = presets[presetName].layers || [];
  for (const ancestorName of layers) {
    if (!(ancestorName in presets)) {
      throw new Error(`"${ancestorName}" was not found among the available presets.`);
    }
    if (unique[ancestorName]) continue;
    if (presets[ancestorName].unique) unique[ancestorName] = true;
    _flattenAncestors(ancestorName, presets, unique, ancestors);
  }
  ancestors.push(presetName);
}

  function _flattenAncestorMetadata(preset, settings, key, defaultVal, presets) {
    const flattened = Array.isArray(defaultVal) ? [] : {};
    for (const ancestorName of preset.flattened_layers) {
      const prototype = presets[ancestorName][key];
      let ancestor;
      if (prototype) {
        if (typeof prototype !== 'function') {
          throw new Error(
            `${ancestorName}: Key "${key}" wasn't wrapped in a function. This can cause unexpected results for the given seed.`
          );
        }
        try {
          ancestor = key === SETTINGS_KEY ? prototype() : prototype(settings);
        } catch (e) {
          if (ancestorName === preset.name) throw e;
          throw new Error(`In ancestor "${ancestorName}": ${e}`);
        }
      } else {
        ancestor = defaultVal;
      }
      if (Array.isArray(defaultVal)) {
        if (!Array.isArray(ancestor)) {
          throw new Error(
            `${ancestorName}: Key "${key}" should be an array, not ${typeof ancestor}.`
          );
        }
        flattened.push(...ancestor);
      } else {
        if (typeof ancestor !== 'object') {
          throw new Error(
            `${ancestorName}: Key "${key}" should be object, not ${typeof ancestor}.`
          );
        }
        Object.assign(flattened, toCamelKeys(ancestor));
      }
    }
    return flattened;
  }

