import { Context } from './context.js';
import { values, rgbToHsv, hsvToRgb, FULLSCREEN_VS } from './value.js';
import { rgbToOklab, oklabToRgb } from './oklab.js';
import { ColorSpace } from './constants.js';
import { shapeFromParams } from './util.js';
import { EFFECTS } from './effectsRegistry.js';

const SETTINGS_KEY = 'settings';
const ALLOWED_KEYS = ['layers', SETTINGS_KEY, 'generator', 'octaves', 'post', 'final', 'ai', 'unique'];

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

    this.settings = _flattenAncestorMetadata(this, null, SETTINGS_KEY, {}, presets);
    Object.assign(this.settings, settings);

    this.generator = _flattenAncestorMetadata(this, this.settings, 'generator', {}, presets);
    this.octave_effects = _flattenAncestorMetadata(this, this.settings, 'octaves', [], presets);
    this.post_effects = _flattenAncestorMetadata(this, this.settings, 'post', [], presets);
    this.final_effects = _flattenAncestorMetadata(this, this.settings, 'final', [], presets);
  }

  render(seed = 0, opts = {}) {
    const {
      ctx = new Context(null),
      width = 256,
      height = 256,
      time = 0,
      speed = 1,
      withAlpha = false,
    } = opts;

    const colorSpace = this.settings.colorSpace || ColorSpace.rgb;
    const shape = shapeFromParams(
      width,
      height,
      colorSpace === ColorSpace.grayscale ? 'grayscale' : 'rgb',
      withAlpha
    );
    const g = this.generator || {};
    const freq = g.freq !== undefined ? g.freq : 1;
    const tensorOpts = { ctx, seed, time, speed, ...g };
    let tensor = values(freq, shape, tensorOpts);

    if (colorSpace === ColorSpace.hsv) {
      tensor = rgbToHsv(tensor);
    } else if (colorSpace === ColorSpace.oklab) {
      tensor = rgbToOklab(tensor);
    }

    for (const e of this.octave_effects) {
      tensor = _applyOctaveEffectOrPreset(e, tensor, shape, time, speed, 0);
    }

    let finalEffects = [];
    for (const e of this.post_effects) {
      const { tensor: t, final } = _applyPostEffectOrPreset(e, tensor, shape, time, speed);
      tensor = t;
      finalEffects = finalEffects.concat(final);
    }

    finalEffects = finalEffects.concat(this.final_effects);

    for (const e of finalEffects) {
      tensor = _applyFinalEffectOrPreset(e, tensor, shape, time, speed);
    }

    if (colorSpace === ColorSpace.hsv) {
      tensor = hsvToRgb(tensor);
    } else if (colorSpace === ColorSpace.oklab) {
      tensor = oklabToRgb(tensor);
    }

    // Present to canvas if available
    if (ctx.canvas) {
      const [h, w, c] = tensor.shape;
      if (ctx.gl && !ctx.isCPU) {
        const gl = ctx.gl;
        ctx.canvas.width = w;
        ctx.canvas.height = h;
        const fs = `#version 300 es\nprecision highp float;\nuniform sampler2D u_tex;\nout vec4 outColor;\nvoid main(){\n vec2 uv = gl_FragCoord.xy / vec2(${w}.0, ${h}.0);\n vec4 color = texture(u_tex, uv);\n outColor = ${c > 3 ? 'color' : 'vec4(color.rgb, 1.0)'};\n}`;
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
            const r = data[i * c];
            const gch = data[i * c + 1] || 0;
            const b = data[i * c + 2] || 0;
            const aVal = c > 3 ? data[i * c + 3] : 1;
            const a = Number.isFinite(aVal) ? aVal : 1;
            img.data[i * 4] = Math.max(0, Math.min(255, Math.round(r * 255)));
            img.data[i * 4 + 1] = Math.max(0, Math.min(255, Math.round(gch * 255)));
            img.data[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(b * 255)));
            img.data[i * 4 + 3] = Math.max(0, Math.min(255, Math.round(a * 255)));
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
  for (const k of Object.keys(params)) {
    if (!keys.includes(k)) {
      throw new Error(`Effect "${effectName}" does not accept a parameter named "${k}"`);
    }
  }
  const applied = {};
  for (const k of keys) {
    applied[k] = params[k] !== undefined ? params[k] : effect[k];
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

function _applyOctaveEffectOrPreset(effectOrPreset, tensor, shape, time, speed, octave) {
  if (typeof effectOrPreset === 'function') {
    if (effectOrPreset.__params && 'displacement' in effectOrPreset.__params) {
      const params = { ...effectOrPreset.__params };
      params.displacement = params.displacement / 2 ** octave;
      const effect = EFFECTS[effectOrPreset.__effectName];
      const args = effectOrPreset.__paramNames.map((k) => params[k]);
      return effect.func(tensor, shape, time, speed, ...args);
    }
    return effectOrPreset(tensor, shape, time, speed);
  } else {
    for (const e of effectOrPreset.octave_effects) {
      tensor = _applyOctaveEffectOrPreset(e, tensor, shape, time, speed, octave);
    }
    return tensor;
  }
}

function _applyPostEffectOrPreset(effectOrPreset, tensor, shape, time, speed) {
  if (typeof effectOrPreset === 'function') {
    return { tensor: effectOrPreset(tensor, shape, time, speed), final: [] };
  } else {
    let final = [...effectOrPreset.final_effects];
    for (const e of effectOrPreset.post_effects) {
      const res = _applyPostEffectOrPreset(e, tensor, shape, time, speed);
      tensor = res.tensor;
      final = final.concat(res.final);
    }
    return { tensor, final };
  }
}

function _applyFinalEffectOrPreset(effectOrPreset, tensor, shape, time, speed) {
  if (typeof effectOrPreset === 'function') {
    return effectOrPreset(tensor, shape, time, speed);
  } else {
    for (const e of effectOrPreset.post_effects.concat(effectOrPreset.final_effects)) {
      tensor = _applyFinalEffectOrPreset(e, tensor, shape, time, speed);
    }
    return tensor;
  }
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
  let flattened = Array.isArray(defaultVal) ? [] : {};
  for (const ancestorName of preset.flattened_layers) {
    const prototype = presets[ancestorName][key];
    let ancestor;
    if (key === SETTINGS_KEY) {
      ancestor = prototype ? prototype() : defaultVal;
    } else {
      ancestor = prototype ? prototype(settings) : defaultVal;
    }
    if (Array.isArray(defaultVal)) {
      if (!Array.isArray(ancestor)) {
        throw new Error(
          `${ancestorName}: Key "${key}" should be an array, not ${typeof ancestor}.`
        );
      }
      flattened = flattened.concat(ancestor);
    } else {
      if (typeof ancestor !== 'object') {
        throw new Error(
          `${ancestorName}: Key "${key}" should be object, not ${typeof ancestor}.`
        );
      }
      Object.assign(flattened, ancestor);
    }
  }
  return flattened;
}

