import { ColorSpace, InterpolationType, OctaveBlending, ValueDistribution } from './constants.js';
import {
  values,
  hsvToRgb,
  rgbToHsv,
  ridge,
  refract,
  normalize,
  clamp01,
  combineOctaves,
  OctaveCombineMode,
  freqForShape,
  proportionalDownsample,
  fxaa,
  setSeed as setValueSeed,
} from './value.js';
import { oklabToRgb } from './oklab.js';
import { Tensor } from './tensor.js';
import { random as simplexRandom } from './simplex.js';
import { setSeed as setRngSeed } from './rng.js';
import { EFFECTS } from './effectsRegistry.js';

async function _applyOctaveEffectOrPreset(effect, tensor, shape, time, speed, octave) {
  if (typeof effect === 'function') {
    if (
      effect.__params &&
      Object.prototype.hasOwnProperty.call(effect.__params, 'displacement')
    ) {
      const params = {
        ...effect.__params,
        displacement: effect.__params.displacement / 2 ** octave,
      };
      const args = effect.__paramNames.map((k) => params[k]);
      return await EFFECTS[effect.__effectName].func(
        tensor,
        shape,
        time,
        speed,
        ...args,
      );
    }
    return await effect(tensor, shape, time, speed);
  } else if (effect && effect.octave_effects) {
    for (const e of effect.octave_effects) {
      tensor = await _applyOctaveEffectOrPreset(e, tensor, shape, time, speed, octave);
    }
    return tensor;
  }
  return tensor;
}

export async function basic(freq, shape, opts = {}) {
  const {
    ridges = false,
    sin = 0,
    splineOrder = InterpolationType.bicubic,
    distrib = ValueDistribution.simplex,
    corners = false,
    mask = null,
    maskInverse = false,
    maskStatic = false,
    latticeDrift = 0,
    color_space = ColorSpace.hsv,
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
    seed = undefined,
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
  };

  // Python treats a seed value of ``0`` as falsy and therefore skips reseeding
  // the RNG modules. Mirror that behaviour here so that ``seed=0`` preserves
  // the existing generator state instead of forcing a fresh sequence.
  const numericSeed = Number.isFinite(seed) ? seed : null;
  if (numericSeed) {
    setRngSeed(numericSeed);
    setValueSeed(numericSeed);
  }

  // Python seeds the global RNG and value modules externally before invoking
  // ``generators.basic``.  Passing an explicit seed down to ``values`` would
  // bypass the internal seed counters (notably ``simplex.getSeed``) and yield a
  // different sequence of lattice permutations.  To mirror Python's behaviour we
  // rely solely on the global seed state configured above and avoid forwarding a
  // per-call seed here.
  let tensor = await values(f, shape, { distrib, ...common });

  if (latticeDrift) {
    const warpShape = [shape[0], shape[1], 1];
    const warpOpts = {
      ctx,
      splineOrder,
      time,
      speed,
      distrib: ValueDistribution.simplex,
    };
    const refX = await values(f, warpShape, warpOpts);
    const refY = await values(f, warpShape, warpOpts);
    tensor = await refract(
      tensor,
      refX,
      refY,
      latticeDrift / Math.min(f[0], f[1]),
      splineOrder,
      false,
    );
  }

  if (octaveEffects) {
    for (const e of octaveEffects) {
      tensor = await _applyOctaveEffectOrPreset(e, tensor, shape, time, speed, octave);
    }
  }

  let alpha = null;
  if (shape[2] === 4) {
    const data = await tensor.read();
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
    const data = await tensor.read();
    const rgb = new Float32Array(shape[0] * shape[1]);
    alpha = new Float32Array(shape[0] * shape[1]);
    for (let i = 0; i < shape[0] * shape[1]; i++) {
      const b2 = i * 2;
      rgb[i] = data[b2];
      alpha[i] = data[b2 + 1];
    }
    tensor = Tensor.fromArray(ctx, rgb, [shape[0], shape[1], 1]);
  }

  let cSpace = color_space;
  const originalColorSpace = color_space;
  if (cSpace === ColorSpace.oklab) {
    const [h, w, c] = tensor.shape;
    if (c !== 3) {
      throw new Error('oklab color space requires 3 channels');
    }
    const data = await tensor.read();
    const converted = new Float32Array(data.length);
    const scale = Math.fround(-0.509);
    const offsetA = Math.fround(0.276);
    const offsetB = Math.fround(0.198);
    for (let i = 0; i < h * w; i++) {
      const base = i * 3;
      const aVal = data[base + 1];
      const bVal = data[base + 2];
      converted[base] = data[base];
      converted[base + 1] = Math.fround(aVal * scale + offsetA);
      converted[base + 2] = Math.fround(bVal * scale + offsetB);
    }
    tensor = Tensor.fromArray(ctx, converted, [h, w, 3]);
    tensor = await oklabToRgb(tensor);
    tensor = await clamp01(tensor);
    cSpace = ColorSpace.rgb;
  }
  if (cSpace === ColorSpace.rgb) {
    tensor = await rgbToHsv(tensor);
    cSpace = ColorSpace.hsv;
  }

  if (cSpace === ColorSpace.hsv) {
    const [h, w] = shape;
    const data = await tensor.read();
    const out = new Float32Array(h * w * 3);
    const f32 = Math.fround;
    let vMin = Infinity;
    let vMax = -Infinity;
    const hueNoise = hueDistrib
      ? await (await values(f, [h, w, 1], { ...common, distrib: hueDistrib })).read()
      : null;
    const satNoise = saturationDistrib
      ? await (await values(f, [h, w, 1], { ...common, distrib: saturationDistrib })).read()
      : null;
    let brightFreqArr = brightnessFreq;
    if (typeof brightFreqArr === 'number') {
      brightFreqArr = freqForShape(brightFreqArr, shape);
    }
    const brightNoise =
      brightnessDistrib || brightFreqArr
        ? await (
            await values(brightFreqArr || f, [h, w, 1], {
              ...common,
              distrib: brightnessDistrib || ValueDistribution.simplex,
            })
          ).read()
        : null;
    // In the Python implementation, when ``hue_rotation`` is ``None`` the
    // value is generated by :func:`simplex.random` without an explicit seed.
    // This consumes a value from the global RNG that was previously seeded via
    // ``rng.set_seed``.  The previous JavaScript port passed the user supplied
    // ``seed`` directly to ``simplexRandom`` which bypassed the global RNG and
    // produced different results.  To maintain parity we let
    // ``simplexRandom`` derive its own seed so that it mirrors the Python
    // behaviour.
    const hueRangeF = f32(
      originalColorSpace === ColorSpace.hsv ? hueRange : 1.0,
    );
    let hueRotF = 0;
    if (!hueNoise) {
      let hueRot = hueRotation;
      if (originalColorSpace !== ColorSpace.hsv) {
        hueRot = 0;
      } else if (hueRot === null || hueRot === undefined) {
        hueRot = simplexRandom(time, undefined, speed);
      }
      hueRotF = f32(hueRot ?? 0);
    }
    const saturationF = f32(saturation);
    for (let i = 0; i < h * w; i++) {
      const base = i * 3;
      const hueSource = f32(data[base]);
      const satSource = f32(data[base + 1]);
      const valSource = f32(data[base + 2]);
      let hVal;
      if (hueNoise) {
        hVal = f32(hueNoise[i]);
      } else {
        const scaled = f32(hueSource * hueRangeF);
        hVal = f32((scaled + hueRotF) % 1.0);
        if (hVal < 0) {
          hVal = f32(hVal + 1);
        }
      }
      let sVal = satNoise ? f32(satNoise[i]) : satSource;
      sVal = f32(sVal * saturationF);
      let vVal = brightNoise ? f32(brightNoise[i]) : valSource;
      if (ridges && splineOrder) {
        const doubled = f32(vVal * 2);
        const diff = f32(Math.abs(doubled - 1));
        vVal = f32(1 - diff);
      }
      if (sin) {
        vVal = f32(Math.sin(f32(sin * vVal)));
        if (vVal < vMin) vMin = vVal;
        if (vVal > vMax) vMax = vVal;
      }
      out[base] = f32(hVal);
      out[base + 1] = f32(sVal);
      out[base + 2] = f32(vVal);
    }
    if (sin) {
      const vMinF = f32(vMin);
      const range = f32(vMax - vMin) || 1;
      for (let i = 0; i < h * w; i++) {
        const idx = i * 3 + 2;
        const adjusted = f32(out[idx] - vMinF);
        out[idx] = f32(adjusted / range);
      }
    }
    tensor = Tensor.fromArray(ctx, out, [h, w, 3]);
    tensor = hsvToRgb(tensor);
  } else if (cSpace === ColorSpace.grayscale) {
    if (ridges && splineOrder) {
      tensor = ridge(tensor);
    }
    if (sin) {
      const data = await tensor.read();
      const out = new Float32Array(data.length);
      for (let i = 0; i < data.length; i++) {
        out[i] = Math.sin(sin * data[i]);
      }
      tensor = Tensor.fromArray(ctx, out, tensor.shape);
    }
  }

  if (alpha) {
    const [h, w] = shape;
    const c = tensor.shape[2];
    const data = await tensor.read();
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

export async function multires(freq, shape, opts = {}) {
  const {
    octaves = 1,
    ridges = false,
    sin = 0,
    splineOrder = InterpolationType.bicubic,
    distrib = ValueDistribution.simplex,
    corners = false,
    mask = null,
    maskInverse = false,
    maskStatic = false,
    latticeDrift = 0,
    color_space = ColorSpace.hsv,
    hueRange = 0.125,
    hueRotation = null,
    saturation = 1.0,
    hueDistrib = null,
    saturationDistrib = null,
    brightnessDistrib = null,
    brightnessFreq = null,
    octaveBlending = OctaveBlending.falloff,
    time = 0,
    speed = 1,
    ctx = null,
    seed = undefined,
  } = opts;

  const octaveEffects = opts.octaveEffects ?? opts.octave_effects ?? [];
  const postEffects = opts.postEffects ?? opts.post_effects ?? [];
  const finalEffects = opts.finalEffects ?? opts.final_effects ?? [];

  const withSupersample =
    opts.withSupersample ?? opts.with_supersample ?? false;
  const withFxaa = opts.withFxaa ?? opts.with_fxaa ?? false;
  const withAi = opts.withAi ?? opts.with_ai ?? false;
  const withUpscale = opts.withUpscale ?? opts.with_upscale ?? false;
  const stabilityModel = opts.stabilityModel ?? opts.stability_model ?? null;
  const styleFilename = opts.styleFilename ?? opts.style_filename ?? null;
  const inputTensor = opts.tensor ?? null;

  // Preserve Python's behaviour where ``seed=0`` leaves the previous global
  // RNG state untouched instead of reseeding with zero.
  const numericSeed = Number.isFinite(seed) ? seed : null;
  if (numericSeed) {
    setRngSeed(numericSeed);
    setValueSeed(numericSeed);
  }

  if (withAi && withSupersample) {
    throw new Error('--with-ai and --with-supersample may not be used together.');
  }
  if (withAi) {
    throw new Error('AI post-processing is not supported in the JavaScript implementation.');
  }
  if (withUpscale) {
    throw new Error('withUpscale is not supported in the JavaScript implementation.');
  }
  void stabilityModel;
  void styleFilename;

  const originalShape = shape.slice();
  const freqShape = shape.slice();
  const rawFreq = Array.isArray(freq)
    ? freq.slice()
    : freqForShape(freq, freqShape);
  const freqArray = [
    rawFreq[0] ?? rawFreq,
    rawFreq[1] ?? rawFreq[0] ?? rawFreq,
  ];

  let workingShape = shape.slice();
  if (withSupersample) {
    workingShape[0] *= 2;
    workingShape[1] *= 2;
  }
  const generationShape = workingShape.slice();
  let combineShape = workingShape;
  const needsAlphaChannel =
    octaveBlending === OctaveBlending.alpha &&
    (combineShape[2] === 1 || combineShape[2] === 3);
  if (needsAlphaChannel) {
    combineShape = combineShape.slice();
    combineShape[2] += 1;
  }

  let tensor = inputTensor;
  let currentShape = tensor ? tensor.shape.slice() : combineShape.slice();

  if (!tensor) {
    const zero = new Float32Array(
      combineShape[0] * combineShape[1] * combineShape[2],
    );
    tensor = Tensor.fromArray(ctx, zero, combineShape);

    for (let octave = 1; octave <= octaves; octave++) {
      const multiplier = 2 ** octave;
      const baseFreq = [
        Math.floor(freqArray[0] * 0.5 * multiplier),
        Math.floor(freqArray[1] * 0.5 * multiplier),
      ];
      if (
        baseFreq[0] > generationShape[0] &&
        baseFreq[1] > generationShape[1]
      ) {
        break;
      }

      const layer = await basic(baseFreq, combineShape, {
        ridges,
        sin,
        splineOrder,
        distrib,
        corners,
        mask,
        maskInverse,
        maskStatic,
        latticeDrift,
        color_space,
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
      });

      let combineMode = OctaveCombineMode.falloff;
      let weight = 1 / multiplier;
      if (octaveBlending === OctaveBlending.reduce_max) {
        combineMode = OctaveCombineMode.reduceMax;
        weight = 0;
      } else if (octaveBlending === OctaveBlending.alpha && combineShape[2] >= 1) {
        combineMode = OctaveCombineMode.alpha;
        weight = 0;
      }
      tensor = await combineOctaves(tensor, layer, combineMode, weight);
      currentShape = tensor.shape.slice();
    }
  } else if (octaveEffects && octaveEffects.length) {
    for (const effect of octaveEffects) {
      tensor = await _applyOctaveEffectOrPreset(
        effect,
        tensor,
        currentShape,
        time,
        speed,
        1,
      );
      currentShape = tensor.shape.slice();
    }
  } else if (tensor) {
    currentShape = tensor.shape.slice();
  }

  if (needsAlphaChannel && (originalShape[2] === 1 || originalShape[2] === 3)) {
    const [h, w, c] = tensor.shape;
    const outChannels = originalShape[2];
    const data = await tensor.read();
    const out = new Float32Array(h * w * outChannels);
    for (let i = 0; i < h * w; i++) {
      const src = i * c;
      const dst = i * outChannels;
      const alpha = data[src + c - 1];
      if (outChannels === 1) {
        out[dst] = data[src] * alpha;
      } else {
        out[dst] = data[src] * alpha;
        out[dst + 1] = data[src + 1] * alpha;
        out[dst + 2] = data[src + 2] * alpha;
      }
    }
    tensor = Tensor.fromArray(ctx, out, [h, w, outChannels]);
    currentShape = tensor.shape.slice();
  }

  tensor = await normalize(tensor);
  currentShape = tensor.shape.slice();

  let final = [];
  for (const e of postEffects) {
    const res = await _applyPostEffectOrPreset(
      e,
      tensor,
      currentShape,
      time,
      speed,
    );
    tensor = res.tensor;
    currentShape = tensor.shape.slice();
    if (res.final && res.final.length) {
      final = final.concat(res.final);
    }
  }

  if (finalEffects && finalEffects.length) {
    final = final.concat(finalEffects);
  }

  for (const e of final) {
    tensor = await _applyFinalEffectOrPreset(
      e,
      tensor,
      currentShape,
      time,
      speed,
    );
    currentShape = tensor.shape.slice();
  }

  tensor = await normalize(tensor);
  currentShape = tensor.shape.slice();

  if (withFxaa) {
    tensor = await fxaa(tensor);
    currentShape = tensor.shape.slice();
  }

  if (withSupersample) {
    tensor = await proportionalDownsample(tensor, currentShape, originalShape);
    currentShape = tensor.shape.slice();
  }

  return tensor;
}

async function _applyPostEffectOrPreset(effectOrPreset, tensor, shape, time, speed) {
  if (typeof effectOrPreset === 'function') {
    return { tensor: await effectOrPreset(tensor, shape, time, speed), final: [] };
  } else {
    let final = [...effectOrPreset.final_effects];
    for (const e of effectOrPreset.post_effects) {
      const res = await _applyPostEffectOrPreset(e, tensor, shape, time, speed);
      tensor = res.tensor;
      shape = tensor.shape;
      final = final.concat(res.final);
    }
    return { tensor, final };
  }
}

async function _applyFinalEffectOrPreset(effectOrPreset, tensor, shape, time, speed) {
  if (typeof effectOrPreset === 'function') {
    return effectOrPreset(tensor, shape, time, speed);
  } else {
    for (const e of effectOrPreset.post_effects.concat(effectOrPreset.final_effects)) {
      tensor = await _applyFinalEffectOrPreset(e, tensor, shape, time, speed);
      shape = tensor.shape;
    }
    return tensor;
  }
}
