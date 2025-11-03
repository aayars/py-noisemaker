import { Context } from './context.js';
import { multires } from './generators.js';
import { ColorSpace } from './constants.js';
import { shapeFromParams } from './util.js';
// Ensure all built-in effects register themselves with the registry.
// The import is intentionally side-effectful.
import './effects.js';
import { EFFECTS } from './effectsRegistry.js';
import { SettingsDict } from './settings.js';
import { resetCallCount, getCallCount } from './rng.js';
import { setSeed as setValueSeed } from './value.js';

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

function valueToArray(value) {
  if (Array.isArray(value)) {
    return value;
  }
  if (ArrayBuffer.isView(value)) {
    return Array.from(value);
  }
  if (value === undefined) {
    return [];
  }
  return [value];
}

function cloneStageParams(params = {}) {
  const out = {};
  if (!params || typeof params !== 'object') {
    return out;
  }
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined) continue;
    if (Array.isArray(value)) {
      out[key] = value.slice();
    } else if (ArrayBuffer.isView(value)) {
      out[key] = Array.from(value);
    } else {
      out[key] = value;
    }
  }
  return out;
}

function gatherStageSnapshots(preset) {
  const stages = [];
  if (!preset) return stages;

  const generatorParams = cloneStageParams(preset.generator || {});
  const generatorName = generatorParams.generator || 'multires';
  stages.push({
    signature: `generator:${generatorName}`,
    params: generatorParams,
  });

  const buckets = [
    ['octave_effects', 'octave'],
    ['post_effects', 'post'],
    ['final_effects', 'final'],
  ];

  for (const [key, category] of buckets) {
    const effects = Array.isArray(preset[key]) ? preset[key] : [];
    for (const effect of effects) {
      let name = 'anonymous';
      let params = {};
      if (typeof effect === 'function' && effect.__effectName) {
        name = effect.__effectName;
        params = cloneStageParams(effect.__params || {});
      } else if (effect && typeof effect === 'object') {
        name = effect.name || 'nested';
        params = cloneStageParams(effect.params || {});
      }
      stages.push({
        signature: `${category}:${name}`,
        params,
      });
    }
  }

  return stages;
}

function normalizeUniformComponents(primary, secondary, fallback, components, bool) {
  const primaryArr = valueToArray(primary);
  const secondaryArr = valueToArray(secondary);
  const fallbackArr = valueToArray(fallback);
  const result = new Array(Math.max(components, 0));
  for (let i = 0; i < result.length; i += 1) {
    let value = primaryArr[i];
    if (value === undefined) value = secondaryArr[i];
    if (value === undefined && secondaryArr.length) {
      value = secondaryArr[Math.min(i, secondaryArr.length - 1)];
    }
    if (value === undefined) value = fallbackArr[i];
    if (value === undefined && fallbackArr.length) {
      value = fallbackArr[Math.min(i, fallbackArr.length - 1)];
    }
    if (bool) {
      result[i] = value ? 1 : 0;
    } else {
      const num = Number(value);
      result[i] = Number.isFinite(num) ? num : 0;
    }
  }
  return result;
}

function writeUniformLayout(view, layout, params, defaults) {
  if (!view || !layout || !Array.isArray(layout.fields)) {
    return;
  }
  for (const field of layout.fields) {
    const values = normalizeUniformComponents(
      params ? params[field.name] : undefined,
      defaults ? defaults[field.name] : undefined,
      field.defaultValue,
      field.components,
      Boolean(field.bool),
    );
    const slots = Math.max(1, Math.floor(field.size / 4));
    for (let i = 0; i < slots; i += 1) {
      const offset = field.offset + i * 4;
      let value = values[i] !== undefined ? values[i] : 0;
      if (field.bool) {
        value = value ? 1 : 0;
        view.setUint32(offset, value >>> 0, true);
      } else if (field.scalarType === 'u32') {
        const uintVal = Number.isFinite(value) ? Math.trunc(value) : 0;
        view.setUint32(offset, uintVal >>> 0, true);
      } else if (field.scalarType === 'i32') {
        const intVal = Number.isFinite(value) ? Math.trunc(value) : 0;
        view.setInt32(offset, intVal, true);
      } else {
        const floatVal = Number.isFinite(value) ? value : 0;
        view.setFloat32(offset, floatVal, true);
      }
    }
  }
}

function writeProgramUniforms(program, stageSnapshots, frameIndex = 0) {
  if (!program || typeof program.stageCount !== 'number') {
    return;
  }
  const stageMap = new Map();
  for (const stage of stageSnapshots || []) {
    if (stage && stage.signature) {
      stageMap.set(stage.signature, stage);
    }
  }
  const bufferIndex = Number.isFinite(frameIndex) ? frameIndex : 0;
  for (let i = 0; i < program.stageCount; i += 1) {
    const descriptor = program.getStageDescriptor(i);
    if (!descriptor || descriptor.gpuSupported === false) {
      continue;
    }
    if (!descriptor.uniformLayout) {
      continue;
    }
    const snapshot = stageMap.get(descriptor.signature) || null;
    const uniformView = program.getUniformBufferView(i, bufferIndex);
    if (!uniformView || !uniformView.view) {
      continue;
    }
    const rawParams = snapshot?.params || {};
    const resolvedParams =
      typeof descriptor.resolveUniformParams === 'function'
        ? descriptor.resolveUniformParams(rawParams, descriptor)
        : rawParams;
    writeUniformLayout(
      uniformView.view,
      descriptor.uniformLayout,
      resolvedParams,
      descriptor.uniformDefaults || {},
    );
    descriptor._lastResolvedParams = resolvedParams;
    if (
      descriptor.shaderId === 'MULTIRES_WGSL' &&
      Array.isArray(resolvedParams?.options0)
    ) {
      const channelCount = resolvedParams.options0[2];
      if (
        Number.isFinite(channelCount) &&
        descriptor.specialization &&
        descriptor.specialization.constants
      ) {
        descriptor.specialization.constants.channelCount = channelCount;
      }
    }
  }
}

export class Preset {
  static _getProgramCache(ctx) {
    if (!ctx || typeof ctx !== 'object') {
      return null;
    }
    if (!this._programCacheMap) {
      this._programCacheMap = new WeakMap();
    }
    let cache = this._programCacheMap.get(ctx);
    if (!cache) {
      cache = new Map();
      this._programCacheMap.set(ctx, cache);
    }
    return cache;
  }

  constructor(presetName, presets, settings = {}, seed, opts = {}) {
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

    const generatorMetadata = _flattenAncestorMetadata(
      this,
      this.settings,
      'generator',
      {},
      presets,
      this.debug,
    );
    this.generator = Object.keys(generatorMetadata).length ? generatorMetadata : null;
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

  is_generator() {
    return this.generator;
  }

  is_effect() {
    const post = Array.isArray(this.post_effects) ? this.post_effects : [];
    if (post.length) {
      return post;
    }
    const final = Array.isArray(this.final_effects) ? this.final_effects : [];
    return final;
  }

  async render(seed = 0, opts = {}) {
    opts = toCamelKeys(opts);
    const {
      ctx: ctxOpt,
      width = 256,
      height = 256,
      time = 0,
      speed = 1,
      withAlpha = false,
      withSupersample = false,
      withFxaa = false,
      withAi = false,
      withUpscale = false,
      stabilityModel = null,
      styleFilename = null,
      tensor: initialTensor = null,
      debug: debugOpt = this.debug,
      collectDebug: collectDebugOpt = false,
      powerPreference = 'high-performance',
      frameIndex: frameIndexOpt = 0,
      frame: frameOpt,
      progressCallback = null,
    } = opts;
    const debug = Boolean(debugOpt);
    const collectDebug = Boolean(collectDebugOpt);
    const gatherDebug = debug || collectDebug;
    const ctx = ctxOpt || new Context(null, debug, powerPreference);

    if (ctx && typeof ctx.forceCPU === 'undefined') {
      ctx.forceCPU = true;
    }
    if (ctx && typeof ctx.isCPU === 'undefined') {
      ctx.isCPU = true;
    }

    const numericFrameIndex = Number(frameIndexOpt);
    const numericFrameAlt = Number(frameOpt);
    if (Number.isFinite(numericFrameIndex) && numericFrameIndex >= 0) {
      Math.floor(numericFrameIndex);
    } else if (Number.isFinite(numericFrameAlt) && numericFrameAlt >= 0) {
      Math.floor(numericFrameAlt);
    }

    if (debug) {
      debugLog(true, `render start: seed=${seed}`, {
        width,
        height,
        time,
        speed,
        withAlpha,
      });
    }
    if (gatherDebug) {
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

    let tensor = null;
    let pipelineInitialTensor = initialTensor;

    if (!tensor) {
      tensor = await multires(freq, shape, {
        withAlpha,
        withSupersample,
        withFxaa,
        withAi,
        withUpscale,
        stabilityModel,
        styleFilename,
        tensor: pipelineInitialTensor,
        color_space: colorSpace,
        ctx,
        seed,
        time,
        speed,
        octaveEffects: this.octave_effects,
        postEffects: this.post_effects,
        finalEffects: this.final_effects,
        progressCallback,
        ...merged,
      });
    }
    if (gatherDebug) {
      if (debug) {
        debugLog(true, 'render complete (cpu)');
      }
      const effectNames = [
        ...this.octave_effects,
        ...this.post_effects,
        ...this.final_effects,
      ].map((e) => e.__effectName || e.name || '');
      const calls = getCallCount();
      if (debug) {
        debugLog(true, 'effect order', effectNames, 'rng calls', calls);
      }
      return { effects: effectNames, rngCalls: calls };
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
        const imgSource = ctx2d.createImageData(w, h);
        const img = imgSource;
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

      if (ctx.canvas.getContext) {
        ctx.canvas.width = w;
        ctx.canvas.height = h;
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

  const fn = function (tensor, shape, time, speed, settings = {}) {
    // Resolve parameters at call time in case they contain unresolved functions
    const resolvedApplied = {};
    for (const k of keys) {
      resolvedApplied[k] = resolveValue(applied[k], settings);
    }
    const args = keys.map((k) => resolvedApplied[k]);
    return effect.func(tensor, shape, time, speed, ...args);
  };
  fn.__effectName = effectName;
  fn.__paramNames = keys;
  fn.__params = { ...applied };
  return fn;
}

export async function render(presetOrName, seed = 0, opts = {}) {
  if (Number.isFinite(seed) && seed !== 0) {
    setValueSeed(seed);
  }
  const { presets = {}, settings } = opts;
  const preset =
    presetOrName instanceof Preset
      ? presetOrName
      : new Preset(presetOrName, presets, settings, seed, opts);
  return preset.render(seed, opts);
}

export { gatherStageSnapshots, writeProgramUniforms };

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

