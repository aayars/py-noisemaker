import { Context } from './context.js';
import { FULLSCREEN_VS, freqForShape } from './value.js';
import { multires } from './generators.js';
import { ColorSpace, OctaveBlending, ValueDistribution } from './constants.js';
import { shapeFromParams, withTensorData } from './util.js';
import { Tensor, markPresentationNormalized } from './tensor.js';
import { compilePreset, buildTopologySignatureFromPreset } from './webgpu/pipeline.js';
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

function coerceNumber(value, fallback = 0) {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function coerceUnsigned(value, fallback = 0, min = 0) {
  const num = Math.floor(Number(value));
  if (!Number.isFinite(num)) {
    return Math.max(min, Math.floor(Number(fallback)) || min);
  }
  return Math.max(min, num);
}

function coerceBooleanFlag(value) {
  if (typeof value === 'string') {
    const lowered = value.trim().toLowerCase();
    if (lowered === 'true' || lowered === '1') return true;
    if (lowered === 'false' || lowered === '0' || lowered === '') return false;
  }
  return Boolean(value);
}

function normalizeColorSpaceValue(value, fallback = ColorSpace.hsv) {
  if (Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string') {
    const lowered = value.trim().toLowerCase();
    if (lowered === 'grayscale' || lowered === 'grey' || lowered === 'gray') {
      return ColorSpace.grayscale;
    }
    if (lowered === 'rgb') {
      return ColorSpace.rgb;
    }
    if (lowered === 'hsv') {
      return ColorSpace.hsv;
    }
    if (lowered === 'oklab') {
      return ColorSpace.oklab;
    }
  }
  return Number.isFinite(fallback) ? fallback : ColorSpace.hsv;
}

function prepareMultiresUniformParams(params = {}, defaults = {}) {
  const out = { ...params };
  const defaultOptions1 = valueToArray(defaults.options1);
  const defaultColorSpace = defaultOptions1.length >= 3 ? defaultOptions1[2] : ColorSpace.hsv;
  const colorSpaceValue = normalizeColorSpaceValue(
    params.color_space ?? params.colorSpace,
    normalizeColorSpaceValue(defaultColorSpace, ColorSpace.hsv),
  );
  out.color_space = colorSpaceValue;
  out.colorSpace = colorSpaceValue;

  const withAlpha = coerceBooleanFlag(params.withAlpha ?? params.with_alpha ?? false);
  const width = coerceNumber(params.width, 0);
  const height = coerceNumber(params.height, 0);
  const shape = shapeFromParams(
    width,
    height,
    colorSpaceValue === ColorSpace.grayscale ? 'grayscale' : 'rgb',
    withAlpha,
  );

  const freqShape = [shape[0], shape[1]];
  let rawFreq = null;
  if (Array.isArray(params.freq) || ArrayBuffer.isView(params.freq)) {
    rawFreq = Array.from(params.freq);
  } else if (params.freq !== undefined) {
    const freqNumber = Number(params.freq);
    if (Number.isFinite(freqNumber)) {
      rawFreq = freqForShape(freqNumber, freqShape);
    }
  }
  if (!rawFreq || rawFreq.length === 0) {
    const defaultFreq = valueToArray(defaults.freq);
    if (defaultFreq.length >= 2) {
      rawFreq = defaultFreq.slice(0, 2);
    } else if (defaultFreq.length === 1) {
      rawFreq = [defaultFreq[0], defaultFreq[0]];
    } else {
      rawFreq = [1, 1];
    }
  }
  const freq0 = coerceNumber(rawFreq[0], 1);
  const freq1Source = rawFreq.length > 1 ? rawFreq[1] : rawFreq[0];
  const freq1 = coerceNumber(freq1Source, freq0);
  out.freq = [freq0, freq1];

  const speedDefault = defaults.speed !== undefined ? defaults.speed : 1;
  const speedValue = coerceNumber(params.speed ?? out.speed, speedDefault);
  out.speed = speedValue;

  const sinDefault = defaults.sin_amount !== undefined ? defaults.sin_amount : 0;
  const sinValue = coerceNumber(params.sin_amount ?? params.sin, sinDefault);
  out.sin_amount = sinValue;

  const defaultColorParams0 = valueToArray(defaults.color_params0);
  const hueRangeDefault = defaultColorParams0.length > 0 ? defaultColorParams0[0] : 0.125;
  const hueRotationDefault = defaultColorParams0.length > 1 ? defaultColorParams0[1] : 0;
  const saturationDefault = defaultColorParams0.length > 2 ? defaultColorParams0[2] : 1;
  const extraDefault = defaultColorParams0.length > 3 ? defaultColorParams0[3] : 0;
  const hueRange = coerceNumber(params.hueRange ?? params.hue_range, hueRangeDefault);
  const hueRotation = coerceNumber(params.hueRotation ?? params.hue_rotation, hueRotationDefault);
  const saturation = coerceNumber(params.saturation, saturationDefault);
  out.color_params0 = [hueRange, hueRotation, saturation, extraDefault];

  const defaultColorParams1 = valueToArray(defaults.color_params1);
  const colorParams1Source =
    Array.isArray(params.color_params1) || ArrayBuffer.isView(params.color_params1)
      ? Array.from(params.color_params1)
      : Array.isArray(params.colorParams1) || ArrayBuffer.isView(params.colorParams1)
      ? Array.from(params.colorParams1)
      : defaultColorParams1;
  out.color_params1 = [
    coerceNumber(colorParams1Source[0], defaultColorParams1[0] ?? 0),
    coerceNumber(colorParams1Source[1], defaultColorParams1[1] ?? 0),
    coerceNumber(colorParams1Source[2], defaultColorParams1[2] ?? 0),
    coerceNumber(colorParams1Source[3], defaultColorParams1[3] ?? 0),
  ];

  const defaultOptions0 = valueToArray(defaults.options0);
  const octavesDefault = defaultOptions0.length > 0 ? defaultOptions0[0] : 1;
  const blendingDefault = defaultOptions0.length > 1 ? defaultOptions0[1] : OctaveBlending.falloff;
  const ridgesDefault = defaultOptions0.length > 3 ? defaultOptions0[3] : 0;
  const octaves = Math.max(1, coerceUnsigned(params.octaves, octavesDefault, 1));
  const octaveBlending = coerceUnsigned(
    params.octaveBlending ?? params.octave_blending,
    blendingDefault,
  );
  const ridges = coerceBooleanFlag(params.ridges ?? ridgesDefault);
  let channelCount = colorSpaceValue === ColorSpace.grayscale ? 1 : 3;
  if (withAlpha) {
    channelCount += 1;
  }
  if (octaveBlending === OctaveBlending.alpha && (channelCount === 1 || channelCount === 3)) {
    channelCount += 1;
  }
  channelCount = coerceUnsigned(params.channelCount, channelCount, 1);
  out.options0 = [octaves >>> 0, octaveBlending >>> 0, channelCount >>> 0, ridges ? 1 : 0];

  const seedOffsetDefault = defaultOptions1.length > 0 ? defaultOptions1[0] : 0;
  const distribDefault = defaultOptions1.length > 1 ? defaultOptions1[1] : ValueDistribution.simplex;
  const seedOffset = coerceUnsigned(
    params.seedOffset ?? params.seed_offset,
    seedOffsetDefault,
  );
  const distrib = coerceUnsigned(params.distrib, distribDefault);
  out.options1 = [seedOffset >>> 0, distrib >>> 0, colorSpaceValue >>> 0, 0];

  return out;
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
<<<<<<< ours
    const rawParams = snapshot?.params || {};
    const resolvedParams =
      typeof descriptor.resolveUniformParams === 'function'
        ? descriptor.resolveUniformParams(rawParams, descriptor)
        : rawParams;
    writeUniformLayout(
      uniformView.view,
      descriptor.uniformLayout,
      resolvedParams,
=======
    let uniformParams = snapshot?.params || {};
    if (descriptor.shaderId === 'MULTIRES_WGSL') {
      uniformParams = prepareMultiresUniformParams(uniformParams, descriptor.uniformDefaults || {});
    }
    writeUniformLayout(
      uniformView.view,
      descriptor.uniformLayout,
      uniformParams,
>>>>>>> theirs
      descriptor.uniformDefaults || {},
    );
  }
}

function makeProgramCacheKey(name, topology, width, height, colorSpace, withAlpha) {
  const safeName = name || 'anonymous';
  const safeTopology = topology || 'none';
  const w = Math.max(1, Math.floor(Number(width) || 0));
  const h = Math.max(1, Math.floor(Number(height) || 0));
  const space = colorSpace ?? 'unknown';
  return `${safeName}|${safeTopology}|${w}x${h}|${space}|alpha:${withAlpha ? 1 : 0}`;
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
      presentationTarget = undefined,
      readback: readbackOpt = false,
    } = opts;
    const debug = Boolean(debugOpt);
    const collectDebug = Boolean(collectDebugOpt);
    const gatherDebug = debug || collectDebug;
    const ctx = ctxOpt || new Context(null, debug, powerPreference);

    const numericFrameIndex = Number(frameIndexOpt);
    const numericFrameAlt = Number(frameOpt);
    const frameIndex = Number.isFinite(numericFrameIndex) && numericFrameIndex >= 0
      ? Math.floor(numericFrameIndex)
      : Number.isFinite(numericFrameAlt) && numericFrameAlt >= 0
      ? Math.floor(numericFrameAlt)
      : 0;
    const readback = Boolean(readbackOpt);

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
    let usedGPU = false;

    const canUseWebGPU = Boolean(ctx && ctx.device && ctx.queue && !ctx.isCPU);
    if (canUseWebGPU) {
      const stageSnapshots = gatherStageSnapshots(this);
      if (stageSnapshots.length) {
        const generatorParams = cloneStageParams(merged);
        const generatorStage = stageSnapshots[0];
        generatorStage.params = generatorParams;
        if (generatorParams.color_space === undefined) generatorParams.color_space = colorSpace;
        if (generatorParams.colorSpace === undefined) generatorParams.colorSpace = colorSpace;
        if (generatorParams.withAlpha === undefined) generatorParams.withAlpha = withAlpha ? 1 : 0;
        if (generatorParams.width === undefined) generatorParams.width = width;
        if (generatorParams.height === undefined) generatorParams.height = height;
        if (generatorParams.withSupersample === undefined) generatorParams.withSupersample = withSupersample ? 1 : 0;
        if (generatorParams.withFxaa === undefined) generatorParams.withFxaa = withFxaa ? 1 : 0;
        if (generatorParams.withAi === undefined) generatorParams.withAi = withAi ? 1 : 0;
        if (generatorParams.withUpscale === undefined) generatorParams.withUpscale = withUpscale ? 1 : 0;
      }
      const dynamicParams = { time, speed, seed, frameIndex };
      for (const stage of stageSnapshots) {
        if (!stage || !stage.params) continue;
        for (const [key, value] of Object.entries(dynamicParams)) {
          if (stage.params[key] === undefined) {
            stage.params[key] = value;
          }
        }
      }
      try {
        const topologySignature = buildTopologySignatureFromPreset(this);
        const cache = Preset._getProgramCache(ctx);
        const cacheKey = cache
          ? makeProgramCacheKey(this.name, topologySignature, width, height, colorSpace, withAlpha)
          : null;
        let program = null;
        if (cache && cacheKey) {
          let entry = cache.get(cacheKey);
          if (entry && entry.program && !entry.program.matchesPreset(this)) {
            if (typeof entry.program.dispose === 'function') {
              try {
                entry.program.dispose();
              } catch (_) {
                /* ignore dispose failures */
              }
            }
            cache.delete(cacheKey);
            entry = null;
          }
          if (!entry) {
            entry = { program: null, unsupported: false };
            try {
              const compiled = compilePreset(this, ctx);
              if (compiled && compiled.stageCount > 0) {
                let unsupportedStage = false;
                if (Array.isArray(compiled.stages)) {
                  unsupportedStage = compiled.stages.some(
                    (stage) => stage && stage.gpuSupported === false,
                  );
                } else {
                  for (let i = 0; i < compiled.stageCount; i += 1) {
                    const descriptor = compiled.getStageDescriptor(i);
                    if (descriptor && descriptor.gpuSupported === false) {
                      unsupportedStage = true;
                      break;
                    }
                  }
                }
                if (unsupportedStage) {
                  entry.unsupported = true;
                  if (typeof compiled.dispose === 'function') {
                    try {
                      compiled.dispose();
                    } catch (_) {
                      /* ignore dispose failures */
                    }
                  }
                } else {
                  entry.program = compiled;
                }
              } else {
                entry.unsupported = true;
                if (compiled && typeof compiled.dispose === 'function') {
                  try {
                    compiled.dispose();
                  } catch (_) {
                    /* ignore dispose failures */
                  }
                }
              }
            } catch (err) {
              entry.unsupported = true;
              if (debug) {
                debugLog(true, 'WebGPU preset compilation failed', err);
              }
            }
            cache.set(cacheKey, entry);
          }
          if (entry && entry.program) {
            program = entry.program;
          }
        }
        if (program) {
          try {
            writeProgramUniforms(program, stageSnapshots, frameIndex);
            const result = await program.execute(ctx, {
              width,
              height,
              time,
              frameIndex,
              seed,
              present: false,
              readback,
              presentationTarget,
            });
            if (result?.texture) {
              const gpuTensor = new Tensor(ctx, result.texture, shape, null);
              if (result.texture && typeof result.texture === 'object') {
                try {
                  result.texture._noisemakerShape = [height, width, 4];
                  result.texture._noisemakerChannels = 4;
                } catch (_) {
                  /* ignore metadata errors */
                }
              }
              markPresentationNormalized(gpuTensor, true);
              tensor = gpuTensor;
              usedGPU = true;
            } else if (result?.readback) {
              const readShape = [height, width, shape[2] ?? 4];
              tensor = Tensor.fromArray(ctx, result.readback, readShape);
              markPresentationNormalized(tensor, true);
              usedGPU = true;
            }
          } catch (err) {
            if (debug) {
              debugLog(true, 'WebGPU execution failed', err);
            }
          }
        }
      } catch (err) {
        if (debug) {
          debugLog(true, 'WebGPU pipeline unavailable', err);
        }
      }
    }

    if (!tensor) {
      tensor = await multires(freq, shape, {
        withAlpha,
        withSupersample,
        withFxaa,
        withAi,
        withUpscale,
        stabilityModel,
        styleFilename,
        tensor: initialTensor,
        color_space: colorSpace,
        ctx,
        seed,
        time,
        speed,
        octaveEffects: this.octave_effects,
        postEffects: this.post_effects,
        finalEffects: this.final_effects,
        ...merged,
      });
    }
    if (gatherDebug) {
      if (debug) {
        debugLog(true, `render complete${usedGPU ? ' (webgpu)' : ''}`);
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
      if (!ctx.isCPU && ctx.canvas.getContext) {
        ctx.canvas.width = w;
        ctx.canvas.height = h;
        try {
          gpuCtx = ctx.canvas.getContext('webgpu');
        } catch (_) {
          gpuCtx = null;
        }
      }
      if (!ctx.isCPU && gpuCtx) {
        ctx.gpu = gpuCtx;
        if (!ctx.device) {
          await ctx.initWebGPU();
        } else {
          ctx.gpu.configure({ device: ctx.device, format: ctx.presentationFormat });
        }
      }
      if (!ctx.isCPU && ctx.device && gpuCtx) {
        ctx.canvas.width = w;
        ctx.canvas.height = h;
        ctx.gpu.configure({ device: ctx.device, format: ctx.presentationFormat });
        const renderTex = (tex) => {
          const res = ctx.renderTexture(tex, () => gpuCtx.getCurrentTexture());
          return res && typeof res.then === 'function' ? res : Promise.resolve(res);
        };
        let tex = tensor.handle;
        const isTex = typeof GPUTexture !== 'undefined' && tex instanceof GPUTexture;
        const storageChannels =
          isTex && tex && typeof tex._noisemakerChannels === 'number' ? tex._noisemakerChannels : c;
        if (!isTex || storageChannels !== 4) {
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
            drawPromise = texRes.then((created) => renderTex(created));
          } else {
            tex = texRes;
            drawPromise = renderTex(tex);
          }
        } else {
          drawPromise = renderTex(tex);
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

export async function render(presetOrName, seed = 0, opts = {}) {
  if (Number.isFinite(seed) && seed !== 0) {
    setValueSeed(seed);
  }
  const { presets = {}, settings } = opts;
  if (presetOrName instanceof Preset) {
    return presetOrName.render(seed, opts);
  }
  const preset =
    typeof presetOrName === 'string'
      ? new Preset(presetOrName, presets, settings, seed, opts)
      : presetOrName;
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

