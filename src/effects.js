import { Tensor } from './tensor.js';
import {
  warp as warpOp,
  sobel,
  normalize,
  blend,
  values,
  adjustHue,
  rgbToHsv,
  hsvToRgb,
  clamp01,
  ridge,
  downsample,
  upsample,
} from './value.js';
import { PALETTES } from './palettes.js';
import { register } from './effectsRegistry.js';
import { random } from './util.js';
import { InterpolationType } from './constants.js';

export function warp(tensor, shape, time, speed, freq = 2, octaves = 1, displacement = 1) {
  let out = tensor;
  for (let octave = 0; octave < octaves; octave++) {
    const mult = 2 ** octave;
    const f = freq * mult;
    const flowX = values(f, [shape[0], shape[1], 1], { seed: 100 + octave, time });
    const flowY = values(f, [shape[0], shape[1], 1], { seed: 200 + octave, time });
    const dx = flowX.read();
    const dy = flowY.read();
    const flowData = new Float32Array(shape[0] * shape[1] * 2);
    for (let i = 0; i < shape[0] * shape[1]; i++) {
      flowData[i * 2] = dx[i] * 2 - 1;
      flowData[i * 2 + 1] = dy[i] * 2 - 1;
    }
    const flow = Tensor.fromArray(tensor.ctx, flowData, [shape[0], shape[1], 2]);
    out = warpOp(out, flow, displacement / mult);
  }
  return out;
}
register('warp', warp, { freq: 2, octaves: 1, displacement: 1 });

export function shadow(tensor, shape, time, speed, alpha = 1) {
  const shade = normalize(sobel(tensor));
  return blend(tensor, shade, alpha);
}
register('shadow', shadow, { alpha: 1 });

export function posterize(tensor, shape, time, speed, levels = 9) {
  if (levels <= 0) return tensor;
  const src = tensor.read();
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    out[i] = Math.floor(src[i] * levels + (1 / levels) * 0.5) / levels;
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register('posterize', posterize, { levels: 9 });

export function fbm(tensor, shape, time, speed, freq = 4, octaves = 4, lacunarity = 2, gain = 0.5) {
  const [h, w, c] = shape;
  let f = freq;
  let amp = 1;
  let total = 0;
  const data = new Float32Array(h * w * c);
  for (let o = 0; o < octaves; o++) {
    const layer = values(f, shape, { seed: o, time });
    const layerData = layer.read();
    for (let i = 0; i < data.length; i++) {
      data[i] += layerData[i] * amp;
    }
    total += amp;
    amp *= gain;
    f *= lacunarity;
  }
  for (let i = 0; i < data.length; i++) {
    data[i] /= total;
  }
  return Tensor.fromArray(tensor ? tensor.ctx : null, data, shape);
}
register('fbm', fbm, { freq: 4, octaves: 4, lacunarity: 2, gain: 0.5 });

const TAU = Math.PI * 2;

export function palette(tensor, shape, time, speed, name = null) {
  if (!name) return tensor;
  const p = PALETTES[name];
  if (!p) return tensor;
  const [h, w] = shape;
  const src = tensor.read();
  const out = new Float32Array(h * w * 3);
  for (let i = 0; i < h * w; i++) {
    const t = src[i];
    out[i * 3] = p.offset[0] + p.amp[0] * Math.cos(TAU * (p.freq[0] * t * 0.875 + 0.0625 + p.phase[0]));
    out[i * 3 + 1] = p.offset[1] + p.amp[1] * Math.cos(TAU * (p.freq[1] * t * 0.875 + 0.0625 + p.phase[1]));
    out[i * 3 + 2] = p.offset[2] + p.amp[2] * Math.cos(TAU * (p.freq[2] * t * 0.875 + 0.0625 + p.phase[2]));
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, 3]);
}
register('palette', palette, { name: null });

export function invert(tensor, shape, time, speed) {
  const src = tensor.read();
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    out[i] = 1 - src[i];
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register('invert', invert, {});

export function aberration(tensor, shape, time, speed, displacement = 0.005) {
  const [h, w, c] = shape;
  if (c !== 3) return tensor;
  const disp = Math.round(w * displacement * random());
  const hueShift = random() * 0.1 - 0.05;
  const shifted = adjustHue(tensor, hueShift);
  const src = shifted.read();
  const out = new Float32Array(h * w * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const base = (y * w + x) * 3;
      const rIdx = (y * w + Math.min(w - 1, x + disp)) * 3;
      const bIdx = (y * w + Math.max(0, x - disp)) * 3;
      out[base] = src[rIdx];
      out[base + 1] = src[base + 1];
      out[base + 2] = src[bIdx + 2];
    }
  }
  const displaced = Tensor.fromArray(tensor.ctx, out, shape);
  return adjustHue(displaced, -hueShift);
}
register('aberration', aberration, { displacement: 0.005 });

export function reindex(tensor, shape, time, speed, displacement = 0.5) {
  const [h, w, c] = shape;
  const src = tensor.read();
  const lum = new Float32Array(h * w);
  for (let i = 0; i < h * w; i++) {
    if (c === 1) {
      lum[i] = src[i];
    } else {
      const base = i * c;
      const r = src[base];
      const g = src[base + 1] || 0;
      const b = src[base + 2] || 0;
      lum[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }
  }
  const mod = Math.min(h, w);
  const out = new Float32Array(h * w * c);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const r = lum[idx];
      const xo = Math.floor((r * displacement * mod + r) % w);
      const yo = Math.floor((r * displacement * mod + r) % h);
      const srcIdx = (yo * w + xo) * c;
      for (let k = 0; k < c; k++) {
        out[idx * c + k] = src[srcIdx + k];
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register('reindex', reindex, { displacement: 0.5 });

export function vignette(tensor, shape, time, speed, brightness = 0.0, alpha = 1.0) {
  const [h, w, c] = shape;
  const norm = normalize(tensor);
  const edgeData = new Float32Array(h * w * c);
  edgeData.fill(brightness);
  const edges = Tensor.fromArray(tensor.ctx, edgeData, shape);
  const cx = (w - 1) / 2;
  const cy = (h - 1) / 2;
  const maxDist = Math.sqrt(cx * cx + cy * cy);
  const maskData = new Float32Array(h * w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy) / maxDist;
      maskData[y * w + x] = dist * dist;
    }
  }
  const mask = Tensor.fromArray(tensor.ctx, maskData, [h, w, 1]);
  const vignetted = blend(norm, edges, mask);
  return blend(norm, vignetted, alpha);
}
register('vignette', vignette, { brightness: 0.0, alpha: 1.0 });

export function dither(tensor, shape, time, speed, levels = 2) {
  const [h, w, c] = shape;
  const noise = values(Math.max(h, w), [h, w, 1], { time, seed: 0, speed: speed * 1000 });
  const n = noise.read();
  const src = tensor.read();
  const out = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    const d = n[i] - 0.5;
    for (let k = 0; k < c; k++) {
      let v = src[i * c + k] + d / levels;
      v = Math.floor(Math.min(1, Math.max(0, v)) * levels) / levels;
      out[i * c + k] = v;
    }
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register('dither', dither, { levels: 2 });

export function grain(tensor, shape, time, speed, alpha = 0.25) {
  const [h, w, c] = shape;
  const noise = values(Math.max(h, w), [h, w, 1], { time, speed: speed * 100 });
  const n = noise.read();
  const noiseData = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    for (let k = 0; k < c; k++) {
      noiseData[i * c + k] = n[i];
    }
  }
  const noiseTensor = Tensor.fromArray(tensor.ctx, noiseData, shape);
  return blend(tensor, noiseTensor, alpha);
}
register('grain', grain, { alpha: 0.25 });

export function saturation(tensor, shape, time, speed, amount = 0.75) {
  if (shape[2] !== 3) return tensor;
  const hsv = rgbToHsv(tensor);
  const data = hsv.read();
  for (let i = 0; i < shape[0] * shape[1]; i++) {
    data[i * 3 + 1] = Math.min(1, Math.max(0, data[i * 3 + 1] * amount));
  }
  return hsvToRgb(Tensor.fromArray(tensor.ctx, data, hsv.shape));
}
register('saturation', saturation, { amount: 0.75 });

export function randomHue(tensor, shape, time, speed, range = 0.05) {
  const shift = random() * range * 2 - range;
  return adjustHue(tensor, shift);
}
register('randomHue', randomHue, { range: 0.05 });

export function normalizeEffect(tensor, shape, time, speed) {
  return normalize(tensor);
}
register('normalize', normalizeEffect, {});

export function adjustBrightness(
  tensor,
  shape,
  time,
  speed,
  amount = 0
) {
  const src = tensor.read();
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const v = src[i] + amount;
    out[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register('adjustBrightness', adjustBrightness, { amount: 0 });

export function adjustContrast(
  tensor,
  shape,
  time,
  speed,
  amount = 1
) {
  const src = tensor.read();
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const v = (src[i] - 0.5) * amount + 0.5;
    out[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register('adjustContrast', adjustContrast, { amount: 1 });

export function adjustHueEffect(tensor, shape, time, speed, amount = 0.25) {
  if (shape[2] !== 3 || amount === 0 || amount === 1 || amount === null) return tensor;
  return adjustHue(tensor, amount);
}
register('adjustHue', adjustHueEffect, { amount: 0.25 });

export function ridgeEffect(tensor, shape, time, speed) {
  return ridge(tensor);
}
register('ridge', ridgeEffect, {});

export function sine(
  tensor,
  shape,
  time,
  speed,
  amount = 1.0,
  rgb = false
) {
  const [h, w, c] = shape;
  const src = tensor.read();
  const out = new Float32Array(h * w * c);
  const ns = (v) => (Math.sin(v) + 1) * 0.5;
  for (let i = 0; i < h * w; i++) {
    const base = i * c;
    if (c === 1) {
      out[i] = ns(src[i] * amount);
    } else if (c === 2) {
      out[base] = ns(src[base] * amount);
      out[base + 1] = src[base + 1];
    } else if (c === 3) {
      if (rgb) {
        out[base] = ns(src[base] * amount);
        out[base + 1] = ns(src[base + 1] * amount);
        out[base + 2] = ns(src[base + 2] * amount);
      } else {
        out[base] = src[base];
        out[base + 1] = src[base + 1];
        out[base + 2] = ns(src[base + 2] * amount);
      }
    } else if (c === 4) {
      if (rgb) {
        out[base] = ns(src[base] * amount);
        out[base + 1] = ns(src[base + 1] * amount);
        out[base + 2] = ns(src[base + 2] * amount);
        out[base + 3] = src[base + 3];
      } else {
        out[base] = src[base];
        out[base + 1] = src[base + 1];
        out[base + 2] = ns(src[base + 2] * amount);
        out[base + 3] = src[base + 3];
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register('sine', sine, { amount: 1.0, rgb: false });


export function blur(
  tensor,
  shape,
  time,
  speed,
  amount = 10.0,
  splineOrder = InterpolationType.bicubic
) {
  const [h, w] = shape;
  const targetH = Math.max(1, Math.floor(h / amount));
  const factor = Math.max(1, Math.floor(h / targetH));
  let small = downsample(tensor, factor);
  const data = small.read();
  for (let i = 0; i < data.length; i++) data[i] *= 4;
  small = Tensor.fromArray(tensor.ctx, data, small.shape);
  const out = upsample(small, factor);
  return out;
}
register('blur', blur, { amount: 10.0, splineOrder: InterpolationType.bicubic });
