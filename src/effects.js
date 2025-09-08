import { Tensor } from './tensor.js';
import { warp as warpOp, sobel, normalize, blend, values } from './value.js';
import { PALETTES } from './palettes.js';
import { register } from './effectsRegistry.js';

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
