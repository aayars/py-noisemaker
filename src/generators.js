import { ColorSpace, InterpolationType, OctaveBlending, ValueDistribution } from './constants.js';
import { values, hsvToRgb, rgbToHsv, ridge, refract, normalize, freqForShape } from './value.js';
import { oklabToRgb } from './oklab.js';
import { Tensor } from './tensor.js';
import { random as simplexRandom } from './simplex.js';

function _applyOctaveEffectOrPreset(effect, tensor, shape, time, speed, octave) {
  if (typeof effect === 'function') {
    return effect(tensor, shape, time, speed);
  } else if (effect && effect.octave_effects) {
    for (const e of effect.octave_effects) {
      tensor = _applyOctaveEffectOrPreset(e, tensor, shape, time, speed, octave);
    }
    return tensor;
  }
  return tensor;
}

export function basic(freq, shape, opts = {}) {
  const {
    ridges = false,
    sin = 0,
    splineOrder = InterpolationType.bicubic,
    distrib = ValueDistribution.uniform,
    corners = false,
    mask = null,
    maskInverse = false,
    maskStatic = false,
    latticeDrift = 0,
    colorSpace = ColorSpace.hsv,
    hueRange = 0.125,
    hueRotation = null,
    saturation = 1.0,
    hueDistrib = null,
    brightnessDistrib = null,
    brightnessFreq = null,
    saturationDistrib = null,
    speed = 1,
    time = 0,
    octaveEffects = null,
    octave = 1,
    ctx = null,
    seed = 0,
  } = opts;

  const f = Array.isArray(freq) ? freq : freqForShape(freq, shape);
  const common = {
    ctx,
    corners,
    mask,
    maskInverse,
    maskStatic,
    splineOrder,
    time,
    speed,
    seed,
  };

  let tensor = values(f, shape, { distrib, ...common });

  if (latticeDrift) {
    tensor = refract(tensor, null, null, latticeDrift / Math.min(f[0], f[1]));
  }

  if (octaveEffects) {
    for (const e of octaveEffects) {
      tensor = _applyOctaveEffectOrPreset(e, tensor, shape, time, speed, octave);
    }
  }

  let alpha = null;
  if (shape[2] === 4) {
    const data = tensor.read();
    const rgb = new Float32Array(shape[0] * shape[1] * 3);
    alpha = new Float32Array(shape[0] * shape[1]);
    for (let i = 0; i < shape[0] * shape[1]; i++) {
      const b4 = i * 4;
      const b3 = i * 3;
      rgb[b3] = data[b4];
      rgb[b3 + 1] = data[b4 + 1];
      rgb[b3 + 2] = data[b4 + 2];
      alpha[i] = data[b4 + 3];
    }
    tensor = Tensor.fromArray(ctx, rgb, [shape[0], shape[1], 3]);
  } else if (shape[2] === 2) {
    const data = tensor.read();
    const rgb = new Float32Array(shape[0] * shape[1]);
    alpha = new Float32Array(shape[0] * shape[1]);
    for (let i = 0; i < shape[0] * shape[1]; i++) {
      const b2 = i * 2;
      rgb[i] = data[b2];
      alpha[i] = data[b2 + 1];
    }
    tensor = Tensor.fromArray(ctx, rgb, [shape[0], shape[1], 1]);
  }

  let cSpace = colorSpace;
  const originalColorSpace = colorSpace;
  if (cSpace === ColorSpace.oklab) {
    tensor = oklabToRgb(tensor);
    cSpace = ColorSpace.rgb;
  }
  if (cSpace === ColorSpace.rgb) {
    tensor = rgbToHsv(tensor);
    cSpace = ColorSpace.hsv;
  }

  if (cSpace === ColorSpace.hsv) {
    const [h, w] = shape;
    const data = tensor.read();
    const out = new Float32Array(h * w * 3);
    let vMin = Infinity;
    let vMax = -Infinity;
    const hueNoise = hueDistrib
      ? values(f, [h, w, 1], { ...common, distrib: hueDistrib }).read()
      : null;
    const satNoise = saturationDistrib
      ? values(f, [h, w, 1], { ...common, distrib: saturationDistrib }).read()
      : null;
    let brightFreqArr = brightnessFreq;
    if (typeof brightFreqArr === 'number') {
      brightFreqArr = freqForShape(brightFreqArr, shape);
    }
    const brightNoise =
      brightnessDistrib || brightFreqArr
        ? values(brightFreqArr || f, [h, w, 1], {
            ...common,
            distrib: brightnessDistrib || ValueDistribution.uniform,
          }).read()
        : null;
    const hueRot =
      hueRotation === null || hueRotation === undefined
        ? originalColorSpace === ColorSpace.hsv
          ? simplexRandom(time, seed, speed)
          : 0
        : hueRotation;
    for (let i = 0; i < h * w; i++) {
      const base = i * 3;
      let hVal = hueNoise
        ? hueNoise[i]
        : (data[base] * (originalColorSpace === ColorSpace.hsv ? hueRange : 1.0) +
            (originalColorSpace === ColorSpace.hsv ? hueRot : 0)) % 1.0;
      let sVal = satNoise ? satNoise[i] : data[base + 1];
      sVal *= saturation;
      let vVal = brightNoise ? brightNoise[i] : data[base + 2];
      if (ridges && splineOrder) {
        vVal = 1 - Math.abs(vVal * 2 - 1);
      }
      if (sin) {
        vVal = Math.sin(sin * vVal);
        if (vVal < vMin) vMin = vVal;
        if (vVal > vMax) vMax = vVal;
      }
      out[base] = hVal;
      out[base + 1] = sVal;
      out[base + 2] = vVal;
    }
    if (sin) {
      const range = vMax - vMin || 1;
      for (let i = 0; i < h * w; i++) {
        out[i * 3 + 2] = (out[i * 3 + 2] - vMin) / range;
      }
    }
    tensor = Tensor.fromArray(ctx, out, [h, w, 3]);
    tensor = hsvToRgb(tensor);
  } else if (cSpace === ColorSpace.grayscale) {
    if (ridges && splineOrder) {
      tensor = ridge(tensor);
    }
    if (sin) {
      const data = tensor.read();
      const out = new Float32Array(data.length);
      let min = Infinity,
        max = -Infinity;
      for (let i = 0; i < data.length; i++) {
        const v = Math.sin(sin * data[i]);
        out[i] = v;
        if (v < min) min = v;
        if (v > max) max = v;
      }
      const range = max - min || 1;
      for (let i = 0; i < out.length; i++) {
        out[i] = (out[i] - min) / range;
      }
      tensor = Tensor.fromArray(ctx, out, tensor.shape);
    }
  }

  if (alpha) {
    const [h, w] = shape;
    const c = tensor.shape[2];
    const data = tensor.read();
    const outC = c + 1;
    const out = new Float32Array(h * w * outC);
    for (let i = 0; i < h * w; i++) {
      const baseOut = i * outC;
      const baseIn = i * c;
      for (let k = 0; k < c; k++) {
        out[baseOut + k] = data[baseIn + k];
      }
      out[baseOut + c] = alpha[i];
    }
    tensor = Tensor.fromArray(ctx, out, [h, w, outC]);
  }

  return tensor;
}

export function multires(freq, shape, opts = {}) {
  const {
    octaves = 1,
    ridges = false,
    sin = 0,
    splineOrder = InterpolationType.bicubic,
    distrib = ValueDistribution.uniform,
    corners = false,
    mask = null,
    maskInverse = false,
    maskStatic = false,
    latticeDrift = 0,
    colorSpace = ColorSpace.hsv,
    hueRange = 0.125,
    hueRotation = null,
    saturation = 1.0,
    hueDistrib = null,
    saturationDistrib = null,
    brightnessDistrib = null,
    brightnessFreq = null,
    octaveBlending = OctaveBlending.falloff,
    octaveEffects = [],
    postEffects = [],
    finalEffects = [],
    time = 0,
    speed = 1,
    ctx = null,
    seed = 0,
  } = opts;

  const f = Array.isArray(freq) ? freq : freqForShape(freq, shape);
  const zero = new Float32Array(shape[0] * shape[1] * shape[2]);
  let tensor = Tensor.fromArray(ctx, zero, shape);

  for (let octave = 1; octave <= octaves; octave++) {
    const multiplier = 2 ** octave;
    const baseFreq = [
      Math.floor(f[0] * 0.5 * multiplier),
      Math.floor(f[1] * 0.5 * multiplier),
    ];
    if (baseFreq[0] > shape[0] && baseFreq[1] > shape[1]) break;
    const layer = basic(baseFreq, shape, {
      ridges,
      sin,
      splineOrder,
      distrib,
      corners,
      mask,
      maskInverse,
      maskStatic,
      latticeDrift,
      colorSpace,
      hueRange,
      hueRotation,
      saturation,
      hueDistrib,
      saturationDistrib,
      brightnessDistrib,
      brightnessFreq,
      octaveEffects,
      octave,
      time,
      speed,
      ctx,
      seed,
    });
    const data = tensor.read();
    const layerData = layer.read();
    const c = shape[2];
    const out = new Float32Array(data.length);
    if (octaveBlending === OctaveBlending.reduce_max) {
      for (let i = 0; i < data.length; i++) {
        out[i] = Math.max(data[i], layerData[i]);
      }
    } else if (octaveBlending === OctaveBlending.alpha && c >= 1) {
      for (let i = 0; i < shape[0] * shape[1]; i++) {
        const base = i * c;
        const a = layerData[base + c - 1];
        for (let k = 0; k < c; k++) {
          out[base + k] = data[base + k] * (1 - a) + layerData[base + k] * a;
        }
      }
    } else {
      for (let i = 0; i < data.length; i++) {
        out[i] = data[i] + layerData[i] / multiplier;
      }
    }
    tensor = Tensor.fromArray(ctx, out, shape);
  }
  tensor = normalize(tensor);

  let final = [];
  for (const e of postEffects) {
    const res = _applyPostEffectOrPreset(e, tensor, shape, time, speed);
    tensor = res.tensor;
    final = final.concat(res.final);
  }

  final = final.concat(finalEffects);
  for (const e of final) {
    tensor = _applyFinalEffectOrPreset(e, tensor, shape, time, speed);
  }

  tensor = normalize(tensor);
  return tensor;
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
