import { Context } from './context.js';
import { FULLSCREEN_VS } from './value.js';
import { multires } from './generators.js';
import { ColorSpace } from './constants.js';
import { shapeFromParams, setSeed, withTensorData } from './util.js';
// Ensure all built-in effects register themselves with the registry.
// The import is intentionally side-effectful.
import './effects.js';
import { EFFECTS } from './effectsRegistry.js';
import { SettingsDict } from './settings.js';
import { resetCallCount, getCallCount, setSeed as setRngSeed } from './rng.js';

const SETTINGS_KEY = 'settings';
const ALLOWED_KEYS = ['layers', SETTINGS_KEY, 'generator', 'octaves', 'post', 'final', 'ai', 'unique'];
const UNUSED_OKAY = [
  'ai',
  'angle',
  'paletteAlpha',
  'paletteName',
  'paletteOn',
  'speed',
  'voronoiSdfSides',
  'voronoiInverse',
];

function resolveValue(v, settings) {
  if (typeof v === 'function') {
    // Mirror Python's arity-sensitive resolution: functions expecting zero
    // args are invoked with no params, those expecting one arg receive the
    // settings object, and others are left untouched.
    try {
      const arity = v.length;
      if (arity === 0) return resolveValue(v(), settings);
      if (arity === 1) return resolveValue(v(settings), settings);
      return v;
    } catch {
      return v;
    }
  }
  if (Array.isArray(v)) return v.map((x) => resolveValue(x, settings));
  if (v && typeof v === 'object') {
    const out = {};
    for (const [k, val] of Object.entries(v)) {
      out[k] = resolveValue(val, settings);
    }
    return out;
  }
  return v;
}

function toCamel(str) {
  return str.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
}

function toSnake(str) {
  return str
    .replace(/([A-Z])/g, '_$1')
    .replace(/-/g, '_')
    .toLowerCase();
}

function toCamelKeys(obj = {}) {
  const out = {};
  for (const [k, v] of Object.entries(obj)) {
    out[toCamel(k)] = v;
  }
  return out;
}

function debugLog(enabled, ...args) {
  if (enabled) {
    console.log('[noisemaker]', ...args);
  }
}

export class Preset {
  constructor(presetName, presets, settings = {}, seed, opts = {}) {
    if (seed !== undefined) setSeed(seed);
    this.debug = opts.debug || false;
    debugLog(this.debug, 'Constructing preset', presetName);
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
    debugLog(this.debug, 'layers', this.layers);
    this.flattened_layers = [];
    _flattenAncestors(presetName, presets, {}, this.flattened_layers, [], this.debug);
    debugLog(this.debug, 'flattened layers', this.flattened_layers);

    this.settings = new SettingsDict(
      _flattenAncestorMetadata(this, null, SETTINGS_KEY, {}, presets, this.debug)
    );
    Object.assign(this.settings, toCamelKeys(settings));
    debugLog(this.debug, 'settings', this.settings);

    this.generator = _flattenAncestorMetadata(this, this.settings, 'generator', {}, presets, this.debug);
    this.octave_effects = _flattenAncestorMetadata(this, this.settings, 'octaves', [], presets, this.debug);
    this.post_effects = _flattenAncestorMetadata(this, this.settings, 'post', [], presets, this.debug);
    this.final_effects = _flattenAncestorMetadata(this, this.settings, 'final', [], presets, this.debug);
    debugLog(this.debug, 'generator', this.generator);
    debugLog(this.debug, 'octaves', this.octave_effects);
    debugLog(this.debug, 'post', this.post_effects);
    debugLog(this.debug, 'final', this.final_effects);

    try {
      this.settings.raiseIfUnaccessed(UNUSED_OKAY);
    } catch (e) {
      throw new Error(`Preset "${presetName}": ${e.message}`);
    }
  }

  async render(seed = 0, opts = {}) {
    opts = toCamelKeys(opts);
    const {
      ctx = new Context(null),
      width = 256,
      height = 256,
      time = 0,
      speed = 1,
      withAlpha = false,
      debug = this.debug,
    } = opts;

    if (debug) {
      debugLog(true, `render start: seed=${seed}`, {
        width,
        height,
        time,
        speed,
        withAlpha,
      });
      resetCallCount();
    }

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

    const merged = { ...this.settings, ...g };
    const freq = merged.freq ?? 1;
    debugLog(debug, 'render merged settings', merged);
    debugLog(debug, 'render shape', shape);

    let tensor = await multires(freq, shape, {
      ...merged,
      color_space: colorSpace,
      ctx,
      seed,
      time,
      speed,
      octaveEffects: this.octave_effects,
      postEffects: this.post_effects,
      finalEffects: this.final_effects,
    });
    if (debug) {
      debugLog(true, 'render complete');
      const effectNames = [
        ...this.octave_effects,
        ...this.post_effects,
        ...this.final_effects,
      ].map((e) => e.__effectName || e.name || '');
      const calls = getCallCount();
      debugLog(true, 'effect order', effectNames, 'rng calls', calls);
    }

    // Present to canvas if available
    let drawPromise = null;
    if (ctx.canvas) {
      const [h, w, c] = tensor.shape;

      const draw2D = (data) => {
        const ctx2d = ctx.canvas.getContext('2d', { willReadFrequently: true });
        if (!ctx2d) return;
        ctx.canvas.width = w;
        ctx.canvas.height = h;
        const img = new ImageData(new Uint8ClampedArray(w * h * 4), w, h);
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
      };

      const drawArray = (arrPromise) =>
        Promise.resolve(arrPromise).then((arr) => draw2D(arr));

      let gpuCtx = null;
      if (ctx.canvas.getContext) {
        try {
          gpuCtx = ctx.canvas.getContext('webgpu');
        } catch (_) {
          gpuCtx = null;
        }
      }
      if (gpuCtx) {
        ctx.gpu = gpuCtx;
        if (!ctx.device) await ctx.initWebGPU();
      }
      if (!ctx.isCPU && ctx.device && gpuCtx) {
        ctx.canvas.width = w;
        ctx.canvas.height = h;
        const renderTex = (tex) => ctx.renderTexture(tex, gpuCtx.getCurrentTexture());
        let tex = tensor.handle;
        const isTex = typeof GPUTexture !== 'undefined' && tex instanceof GPUTexture;
        if (!isTex || c !== 4) {
          const texRes = withTensorData(tensor, (data) => {
            if (c !== 4) {
              const padded = new Float32Array(w * h * 4);
              for (let i = 0; i < w * h; i++) {
                const src = i * c;
                const dst = i * 4;
                const r = data[src];
                const g = c > 2 ? data[src + 1] : r;
                const b = c > 2 ? data[src + 2] : r;
                const aVal = c > 3 ? data[src + 3] : c === 2 ? data[src + 1] : 1;
                const a = Number.isFinite(aVal) ? aVal : 1;
                padded[dst] = r;
                padded[dst + 1] = g;
                padded[dst + 2] = b;
                padded[dst + 3] = a;
              }
              data = padded;
            }
            return ctx.createTexture(w, h, data);
          });
          if (texRes && typeof texRes.then === 'function') {
            drawPromise = texRes.then(renderTex);
          } else {
            tex = texRes;
            renderTex(tex);
          }
        } else {
          renderTex(tex);
        }
      } else if (ctx.gl && !ctx.isCPU) {
        const gl = ctx.gl;
        ctx.canvas.width = w;
        ctx.canvas.height = h;
        const colorExpr =
          c > 3
            ? 'color'
            : c === 1
            ? 'vec4(color.rrr, 1.0)'
            : c === 2
            ? 'vec4(color.rrr, color.g)'
            : 'vec4(color.rgb, 1.0)';
        const fs = `#version 300 es
precision highp float;
uniform sampler2D u_tex;
out vec4 outColor;
void main(){
 vec2 uv = vec2(gl_FragCoord.x / ${w}.0, 1.0 - gl_FragCoord.y / ${h}.0);
 vec4 color = texture(u_tex, uv);
 outColor = ${colorExpr};
}`;
        const prog = ctx.getProgram(FULLSCREEN_VS, fs);
        gl.useProgram(prog);
        gl.activeTexture(gl.TEXTURE0);
        let tex = tensor.handle;
        const isTex = typeof GPUTexture !== 'undefined' && tex instanceof GPUTexture;
        if (!isTex) {
          const texRes = withTensorData(tensor, (data) => {
            if (c !== 4) {
              const padded = new Float32Array(w * h * 4);
              for (let i = 0; i < w * h; i++) {
                const src = i * c;
                const dst = i * 4;
                const r = data[src];
                const g = c > 2 ? data[src + 1] : r;
                const b = c > 2 ? data[src + 2] : r;
                const aVal = c > 3 ? data[src + 3] : c === 2 ? data[src + 1] : 1;
                const a = Number.isFinite(aVal) ? aVal : 1;
                padded[dst] = r;
                padded[dst + 1] = g;
                padded[dst + 2] = b;
                padded[dst + 3] = a;
              }
              data = padded;
            }
            return ctx.createTexture(w, h, data);
          });
          if (texRes && typeof texRes.then === 'function') {
            drawPromise = texRes.then((t) => {
              gl.bindTexture(gl.TEXTURE_2D, t);
              gl.uniform1i(gl.getUniformLocation(prog, 'u_tex'), 0);
              ctx.bindFramebuffer(null, w, h);
              ctx.drawQuad();
              gl.bindTexture(gl.TEXTURE_2D, null);
            });
          } else {
            tex = texRes;
            gl.bindTexture(gl.TEXTURE_2D, tex);
            gl.uniform1i(gl.getUniformLocation(prog, 'u_tex'), 0);
            ctx.bindFramebuffer(null, w, h);
            ctx.drawQuad();
            gl.bindTexture(gl.TEXTURE_2D, null);
          }
        } else {
          gl.bindTexture(gl.TEXTURE_2D, tex);
          gl.uniform1i(gl.getUniformLocation(prog, 'u_tex'), 0);
          ctx.bindFramebuffer(null, w, h);
          ctx.drawQuad();
          gl.bindTexture(gl.TEXTURE_2D, null);
        }
      } else if (ctx.canvas.getContext) {
        drawPromise = drawArray(tensor.read());
      }
    }

    return drawPromise ? drawPromise.then(() => tensor) : tensor;
  }
}

export function Effect(effectName, params = {}) {
  let effect = EFFECTS[effectName];
  if (!effect) {
    effect = EFFECTS[toSnake(effectName)];
  }
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

export async function render(presetName, seed = 0, opts = {}) {
  setRngSeed(seed);
  const { presets = {}, settings } = opts;
  const preset =
    typeof presetName === 'string'
      ? new Preset(presetName, presets, settings, seed, opts)
      : presetName;
  return preset.render(seed, opts);
}

function _flattenAncestors(presetName, presets, unique, ancestors, stack = [], debug = false) {
  debugLog(debug, '_flattenAncestors enter', presetName, 'stack', stack);
  if (stack.includes(presetName)) {
    const cycle = stack.slice(stack.indexOf(presetName)).concat(presetName).join(' -> ');
    debugLog(debug, 'cycle detected', cycle);
    throw new Error(`Cycle detected in preset layers: ${cycle}`);
  }
  stack.push(presetName);
  const layers = presets[presetName].layers || [];
  debugLog(debug, 'layers for', presetName, layers);
  for (const ancestorName of layers) {
    if (!(ancestorName in presets)) {
      throw new Error(`"${ancestorName}" was not found among the available presets.`);
    }
    if (unique[ancestorName]) continue;
    if (presets[ancestorName].unique) unique[ancestorName] = true;
    _flattenAncestors(ancestorName, presets, unique, ancestors, stack, debug);
  }
  stack.pop();
  ancestors.push(presetName);
  debugLog(debug, '_flattenAncestors exit', presetName, 'ancestors', ancestors);
}

function _flattenAncestorMetadata(preset, settings, key, defaultVal, presets, debug = false) {
  debugLog(debug, '_flattenAncestorMetadata', key);
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
        ancestor = key === SETTINGS_KEY ? resolveValue(ancestor, settings) : ancestor;
        debugLog(debug, `metadata from ${ancestorName}`, ancestor);
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
  debugLog(debug, '_flattenAncestorMetadata result', key, flattened);
  return flattened;
}

