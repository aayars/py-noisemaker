import { Context } from './context.js';
import { FULLSCREEN_VS } from './value.js';
import { multires } from './generators.js';
import { ColorSpace } from './constants.js';
import { shapeFromParams, setSeed, withTensorData } from './util.js';
import { Tensor, markPresentationNormalized } from './tensor.js';
import { compilePreset, buildTopologySignatureFromPreset } from './webgpu/pipeline.js';
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
    writeUniformLayout(uniformView.view, descriptor.uniformLayout, snapshot?.params || {}, descriptor.uniformDefaults || {});
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
      ctx: ctxOpt,
      width = 256,
      height = 256,
      time = 0,
      speed = 1,
      withAlpha = false,
      debug = this.debug,
      powerPreference = 'high-performance',
      frameIndex: frameIndexOpt = 0,
      frame: frameOpt,
      presentationTarget = undefined,
      readback: readbackOpt = false,
    } = opts;
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
    }
    if (debug) {
      debugLog(true, `render complete${usedGPU ? ' (webgpu)' : ''}`);
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
  setRngSeed(seed);
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

