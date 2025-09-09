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
  distance,
  ridge,
  downsample,
  upsample,
  FULLSCREEN_VS,
} from './value.js';
import { PALETTES } from './palettes.js';
import { register } from './effectsRegistry.js';
import { random as simplexRandom } from './simplex.js';
import { maskValues } from './masks.js';
import { random, randomInt } from './util.js';
import { InterpolationType, DistanceMetric, ValueMask } from './constants.js';

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

export function derivative(
  tensor,
  shape,
  time,
  speed,
  distMetric = DistanceMetric.euclidean,
  withNormalize = true,
  alpha = 1
) {
  let out = sobel(tensor);
  if (withNormalize) out = normalize(out);
  if (alpha === 1) return out;
  return blend(tensor, out, alpha);
}
register('derivative', derivative, {
  distMetric: DistanceMetric.euclidean,
  withNormalize: true,
  alpha: 1,
});

export function sobelOperator(
  tensor,
  shape,
  time,
  speed,
  distMetric = DistanceMetric.euclidean
) {
  const blurred = blur(tensor, shape, time, speed);
  let out = sobel(blurred);
  out = normalize(out);
  const data = out.read();
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.abs(data[i] * 2 - 1);
  }
  return Tensor.fromArray(tensor.ctx, data, shape);
}
register('sobel', sobelOperator, {
  distMetric: DistanceMetric.euclidean,
});

export function normalMap(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const src = tensor.read();
  const gray = new Float32Array(h * w);
  for (let i = 0; i < h * w; i++) {
    if (c === 1) {
      gray[i] = src[i];
    } else {
      const base = i * c;
      const r = src[base];
      const g = src[base + 1] || 0;
      const b = src[base + 2] || 0;
      gray[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }
  }
  const gxKernel = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const gyKernel = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  const gx = new Float32Array(h * w);
  const gy = new Float32Array(h * w);
  function get(x, y) {
    x = Math.max(0, Math.min(w - 1, x));
    y = Math.max(0, Math.min(h - 1, y));
    return gray[y * w + x];
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sx = 0, sy = 0, idx = 0;
      for (let yy = -1; yy <= 1; yy++) {
        for (let xx = -1; xx <= 1; xx++) {
          const v = get(x + xx, y + yy);
          sx += gxKernel[idx] * v;
          sy += gyKernel[idx] * v;
          idx++;
        }
      }
      const i = y * w + x;
      gx[i] = 1 - sx;
      gy[i] = sy;
    }
  }
  let xTensor = normalize(Tensor.fromArray(tensor.ctx, gx, [h, w, 1]));
  let yTensor = normalize(Tensor.fromArray(tensor.ctx, gy, [h, w, 1]));
  const xData = xTensor.read();
  const yData = yTensor.read();
  const mag = new Float32Array(h * w);
  for (let i = 0; i < h * w; i++) {
    mag[i] = Math.sqrt(xData[i] * xData[i] + yData[i] * yData[i]);
  }
  const zNorm = normalize(Tensor.fromArray(tensor.ctx, mag, [h, w, 1])).read();
  const out = new Float32Array(h * w * 3);
  for (let i = 0; i < h * w; i++) {
    const z = 1 - Math.abs(zNorm[i] * 2 - 1) * 0.5 + 0.5;
    out[i * 3] = xData[i];
    out[i * 3 + 1] = yData[i];
    out[i * 3 + 2] = z;
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, 3]);
}
register('normalMap', normalMap, {});

export function densityMap(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const bins = Math.max(h, w);
  const vals = normalize(tensor).read();
  const countIdx = new Int32Array(h * w);
  const counts = new Int32Array(bins);
  for (let i = 0; i < h * w; i++) {
    const v = vals[i * c];
    const b = Math.min(bins - 1, Math.floor(v * (bins - 1)));
    countIdx[i] = b;
    counts[b]++;
  }
  const out = new Float32Array(h * w);
  for (let i = 0; i < h * w; i++) out[i] = counts[countIdx[i]];
  const norm = normalize(Tensor.fromArray(tensor.ctx, out, [h, w, 1])).read();
  const full = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    for (let k = 0; k < c; k++) full[i * c + k] = norm[i];
  }
  return Tensor.fromArray(tensor.ctx, full, shape);
}
register('densityMap', densityMap, {});

export function jpegDecimate(
  tensor,
  shape,
  time,
  speed,
  iterations = 25
) {
  let out = tensor;
  for (let i = 0; i < iterations; i++) {
    const src = out.read();
    const q = randomInt(5, 50);
    const shift = Math.floor((100 - q) / 10) + 1;
    const tmp = new Uint8Array(src.length);
    for (let j = 0; j < src.length; j++) {
      let v = Math.min(255, Math.max(0, Math.round(src[j] * 255)));
      v = (v >> shift) << shift;
      tmp[j] = v;
    }
    const f32 = new Float32Array(src.length);
    for (let j = 0; j < src.length; j++) f32[j] = tmp[j] / 255;
    out = Tensor.fromArray(tensor.ctx, f32, shape);
  }
  return out;
}
register('jpegDecimate', jpegDecimate, { iterations: 25 });

const BLUR_KERNEL = maskValues(ValueMask.conv2d_blur)[0].read();
const SHARPEN_KERNEL = maskValues(ValueMask.conv2d_sharpen)[0].read();

function convolveKernel(tensor, kernel, size, normalizeKernel = true) {
  const [h, w, c] = tensor.shape;
  const src = tensor.read();
  const out = new Float32Array(h * w * c);
  const r = Math.floor(size / 2);
  let norm = 0;
  if (normalizeKernel) {
    for (let i = 0; i < kernel.length; i++) norm += kernel[i];
  } else {
    norm = 1;
  }
  if (norm === 0) norm = 1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let k = 0; k < c; k++) {
        let sum = 0;
        let idx = 0;
        for (let yy = -r; yy <= r; yy++) {
          const ycl = Math.max(0, Math.min(h - 1, y + yy));
          for (let xx = -r; xx <= r; xx++) {
            const xcl = Math.max(0, Math.min(w - 1, x + xx));
            sum += kernel[idx++] * src[(ycl * w + xcl) * c + k];
          }
        }
        out[(y * w + x) * c + k] = sum / norm;
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, tensor.shape);
}

export function convFeedback(
  tensor,
  shape,
  time,
  speed,
  iterations = 50,
  alpha = 0.5
) {
  let convolved = downsample(tensor, 2);
  for (let i = 0; i < iterations; i++) {
    convolved = convolveKernel(convolved, BLUR_KERNEL, 5, true);
    convolved = convolveKernel(convolved, SHARPEN_KERNEL, 3, false);
  }
  convolved = normalize(convolved);
  const data = convolved.read();
  const up = new Float32Array(data.length);
  const downArr = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    up[i] = Math.max((data[i] - 0.5) * 2, 0);
    downArr[i] = Math.min(data[i] * 2, 1);
  }
  const combined = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) combined[i] = up[i] + (1 - downArr[i]);
  const combinedTensor = Tensor.fromArray(convolved.ctx, combined, convolved.shape);
  const resampled = upsample(combinedTensor, 2);
  return blend(tensor, resampled, alpha);
}
register('convFeedback', convFeedback, { iterations: 50, alpha: 0.5 });

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
  const ctx = tensor.ctx;
  if (ctx && !ctx.isCPU) {
    const gl = ctx.gl;
    const fs = `#version 300 es\nprecision highp float;\nuniform sampler2D u_tex;\nuniform float u_disp;\nuniform float u_mod;\nuniform float u_channels;\nout vec4 outColor;\nvoid main(){\n vec2 res = vec2(${w}.0, ${h}.0);\n vec2 uv = gl_FragCoord.xy / res;\n vec4 col = texture(u_tex, uv);\n float lum = col.r;\n if(u_channels > 1.5){ lum = dot(col.rgb, vec3(0.2126,0.7152,0.0722)); }\n float off = lum * u_disp * u_mod + lum;\n float xo = floor(mod(off, res.x));\n float yo = floor(mod(off, res.y));\n vec2 suv = (vec2(xo, yo) + 0.5) / res;\n outColor = texture(u_tex, suv);\n}`;
    const prog = ctx.createProgram(FULLSCREEN_VS, fs);
    const pp = ctx.pingPong(w, h);
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tensor.handle);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_tex'), 0);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_disp'), displacement);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_mod'), Math.min(h, w));
    gl.uniform1f(gl.getUniformLocation(prog, 'u_channels'), c);
    ctx.bindFramebuffer(pp.writeFbo, w, h);
    ctx.drawQuad();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteProgram(prog);
    return new Tensor(ctx, pp.writeTex, shape);
  }
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

export function ripple(
  tensor,
  shape,
  time,
  speed,
  freq = 2,
  displacement = 1,
  kink = 1,
  reference = null,
  splineOrder = InterpolationType.bicubic,
) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  if (ctx && !ctx.isCPU) {
    const refTensor = reference || values(freq, [h, w, 1], { ctx, time, speed, splineOrder });
    const refTex = refTensor.ctx === ctx ? refTensor : Tensor.fromArray(ctx, refTensor.read(), refTensor.shape);
    const gl = ctx.gl;
    const rand = simplexRandom(time, undefined, speed);
    const fs = `#version 300 es\nprecision highp float;\nuniform sampler2D u_tex;\nuniform sampler2D u_ref;\nuniform float u_disp;\nuniform float u_kink;\nuniform float u_rand;\nout vec4 outColor;\nvoid main(){\n vec2 res = vec2(${w}.0, ${h}.0);\n vec2 uv = gl_FragCoord.xy / res;\n float ref = texture(u_ref, uv).r;\n float ang = ref * ${TAU} * u_kink * u_rand;\n vec2 offset = vec2(cos(ang), sin(ang)) * u_disp;\n vec2 uv2 = fract(uv + offset);\n outColor = texture(u_tex, uv2);\n}`;
    const prog = ctx.createProgram(FULLSCREEN_VS, fs);
    const pp = ctx.pingPong(w, h);
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tensor.handle);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_tex'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, refTex.handle);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_ref'), 1);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_disp'), displacement);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_kink'), kink);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_rand'), rand);
    ctx.bindFramebuffer(pp.writeFbo, w, h);
    ctx.drawQuad();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteProgram(prog);
    return new Tensor(ctx, pp.writeTex, shape);
  }
  let ref = reference;
  if (!ref) {
    ref = values(freq, [h, w, 1], { time, speed, splineOrder });
  }
  const refData = ref.read();
  const rand = simplexRandom(time, undefined, speed);
  const src = tensor.read();
  const out = new Float32Array(h * w * c);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const angle = refData[idx] * TAU * kink * rand;
      const fx = x + Math.cos(angle) * displacement * w;
      const fy = y + Math.sin(angle) * displacement * h;
      const x0 = Math.floor(fx);
      const y0 = Math.floor(fy);
      const x1 = x0 + 1;
      const y1 = y0 + 1;
      const sx = fx - x0;
      const sy = fy - y0;
      const x0m = ((x0 % w) + w) % w;
      const x1m = ((x1 % w) + w) % w;
      const y0m = ((y0 % h) + h) % h;
      const y1m = ((y1 % h) + h) % h;
      for (let k = 0; k < c; k++) {
        const c00 = src[(y0m * w + x0m) * c + k];
        const c10 = src[(y0m * w + x1m) * c + k];
        const c01 = src[(y1m * w + x0m) * c + k];
        const c11 = src[(y1m * w + x1m) * c + k];
        const c0 = c00 * (1 - sx) + c10 * sx;
        const c1 = c01 * (1 - sx) + c11 * sx;
        out[(idx) * c + k] = c0 * (1 - sy) + c1 * sy;
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register('ripple', ripple, {
  freq: 2,
  displacement: 1,
  kink: 1,
  reference: null,
  splineOrder: InterpolationType.bicubic,
});

export function colorMap(
  tensor,
  shape,
  time,
  speed,
  clut = null,
  horizontal = false,
  displacement = 0.5,
) {
  if (!clut) return tensor;
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  if (ctx && !ctx.isCPU && clut.ctx === ctx) {
    const gl = ctx.gl;
    const fs = `#version 300 es\nprecision highp float;\nuniform sampler2D u_tex;\nuniform sampler2D u_clut;\nuniform float u_disp;\nuniform float u_horizontal;\nuniform float u_channels;\nout vec4 outColor;\nvoid main(){\n vec2 res = vec2(${w}.0, ${h}.0);\n vec2 uv = gl_FragCoord.xy / res;\n vec4 col = texture(u_tex, uv);\n float lum = col.r;\n if(u_channels > 1.5){ lum = dot(col.rgb, vec3(0.2126,0.7152,0.0722)); }\n float ref = lum * u_disp;\n float xo = floor(ref * float(${w - 1})) / float(${w});\n float yo = u_horizontal > 0.5 ? 0.0 : floor(ref * float(${h - 1})) / float(${h});\n vec2 uv2 = fract(uv + vec2(xo, yo));\n outColor = texture(u_clut, uv2);\n}`;
    const prog = ctx.createProgram(FULLSCREEN_VS, fs);
    const pp = ctx.pingPong(w, h);
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tensor.handle);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_tex'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, clut.handle);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_clut'), 1);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_disp'), displacement);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_horizontal'), horizontal ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_channels'), c);
    ctx.bindFramebuffer(pp.writeFbo, w, h);
    ctx.drawQuad();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteProgram(prog);
    return new Tensor(ctx, pp.writeTex, [h, w, clut.shape[2]]);
  }
  const [ch, cw, cc] = clut.shape;
  const clutData = clut.read();
  const src = tensor.read();
  const out = new Float32Array(h * w * cc);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      let lum;
      if (c === 1) {
        lum = src[idx];
      } else {
        const base = idx * c;
        const r = src[base];
        const g = src[base + 1] || 0;
        const b = src[base + 2] || 0;
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      }
      const ref = lum * displacement;
      const xi = (x + Math.floor(ref * (w - 1))) % w;
      const yi = horizontal ? y : (y + Math.floor(ref * (h - 1))) % h;
      const sx = Math.floor(xi * cw / w);
      const sy = Math.floor(yi * ch / h);
      const srcIdx = (sy * cw + sx) * cc;
      const outIdx = (y * w + x) * cc;
      for (let k = 0; k < cc; k++) {
        out[outIdx + k] = clutData[srcIdx + k];
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, cc]);
}
register('colorMap', colorMap, { clut: null, horizontal: false, displacement: 0.5 });

function randomNormal(mean = 0, std = 1) {
  const u1 = random() || 1e-9;
  const u2 = random();
  const mag = Math.sqrt(-2 * Math.log(u1));
  const z0 = mag * Math.cos(TAU * u2);
  return z0 * std + mean;
}

function periodicValue(t, v) {
  return (Math.sin((t - v) * TAU) + 1) * 0.5;
}

function offsetIndex(yArr, height, xArr, width) {
  const yOff = Math.floor(height * 0.5 + random() * height * 0.5);
  const xOff = Math.floor(random() * width * 0.5);
  const n = yArr.length;
  const oy = new Int32Array(n);
  const ox = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    oy[i] = (yArr[i] + yOff) % height;
    ox[i] = (xArr[i] + xOff) % width;
  }
  return { y: oy, x: ox };
}

export function erosionWorms(
  tensor,
  shape,
  time,
  speed,
  density = 50,
  iterations = 50,
  contraction = 1.0,
  quantize = false,
  alpha = 0.25,
  inverse = false,
  xyBlend = 0
) {
  const [h, w, c] = shape;
  const count = Math.floor(Math.sqrt(h * w) * density);
  const x = new Float32Array(count);
  const y = new Float32Array(count);
  const xDir = new Float32Array(count);
  const yDir = new Float32Array(count);
  const inertia = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    x[i] = random() * (w - 1);
    y[i] = random() * (h - 1);
    const ang = random() * TAU;
    xDir[i] = Math.cos(ang);
    yDir[i] = Math.sin(ang);
    inertia[i] = randomNormal(0.75, 0.25);
  }
  const src = tensor.read();
  const startColors = new Float32Array(count * c);
  for (let i = 0; i < count; i++) {
    const xi = Math.floor(x[i]);
    const yi = Math.floor(y[i]);
    const base = (yi * w + xi) * c;
    for (let k = 0; k < c; k++) {
      startColors[i * c + k] = src[base + k];
    }
  }
  // grayscale values
  const valuesArr = new Float32Array(h * w);
  for (let yi = 0; yi < h; yi++) {
    for (let xi = 0; xi < w; xi++) {
      const idx = yi * w + xi;
      if (c === 1) {
        valuesArr[idx] = src[idx];
      } else {
        const base = idx * c;
        const r = src[base];
        const g = src[base + 1] || 0;
        const b = src[base + 2] || 0;
        valuesArr[idx] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      }
    }
  }
  const out = new Float32Array(h * w * c);
  for (let iter = 0; iter < iterations; iter++) {
    const exposure = iterations > 1 ? 1 - Math.abs(1 - (iter / (iterations - 1)) * 2) : 1;
    for (let j = 0; j < count; j++) {
      const xi = Math.floor(x[j]) % w;
      const yi = Math.floor(y[j]) % h;
      const idx = yi * w + xi;
      const base = idx * c;
      for (let k = 0; k < c; k++) {
        out[base + k] += startColors[j * c + k] * exposure;
      }
      const x1 = (xi + 1) % w;
      const y1 = (yi + 1) % h;
      const sv = valuesArr[idx];
      const x1v = valuesArr[yi * w + x1];
      const y1v = valuesArr[y1 * w + xi];
      const x1y1v = valuesArr[y1 * w + x1];
      const u = x[j] - Math.floor(x[j]);
      const v = y[j] - Math.floor(y[j]);
      const gX = (y1v - sv) * (1 - u) + (x1y1v - x1v) * u;
      const gY = (x1v - sv) * (1 - v) + (x1y1v - y1v) * v;
      const gx = quantize ? Math.floor(gX) : gX;
      const gy = quantize ? Math.floor(gY) : gY;
      const len = distance(gx, gy) * contraction || 1;
      xDir[j] = xDir[j] * (1 - inertia[j]) + (gx / len) * inertia[j];
      yDir[j] = yDir[j] * (1 - inertia[j]) + (gy / len) * inertia[j];
      x[j] = (x[j] + xDir[j]) % w;
      y[j] = (y[j] + yDir[j]) % h;
    }
  }
  let outTensor = Tensor.fromArray(tensor.ctx, out, shape);
  outTensor = clamp01(outTensor);
  if (inverse) {
    const d = outTensor.read();
    for (let i = 0; i < d.length; i++) d[i] = 1 - d[i];
    outTensor = Tensor.fromArray(outTensor.ctx, d, shape);
  }
  if (xyBlend) {
    const valMask = new Float32Array(h * w);
    for (let i = 0; i < h * w; i++) valMask[i] = valuesArr[i] * xyBlend;
    const mask = Tensor.fromArray(tensor.ctx, valMask, [h, w, 1]);
    tensor = blend(shadow(tensor, shape, time, speed), reindex(tensor, shape, time, speed, 1), mask);
  }
  return blend(tensor, outTensor, alpha);
}
register('erosionWorms', erosionWorms, {
  density: 50,
  iterations: 50,
  contraction: 1.0,
  quantize: false,
  alpha: 0.25,
  inverse: false,
  xyBlend: 0,
});

export function worms(
  tensor,
  shape,
  time,
  speed,
  behavior = 1,
  density = 4.0,
  duration = 4.0,
  stride = 1.0,
  strideDeviation = 0.05,
  alpha = 0.5,
  kink = 1.0,
  drunkenness = 0.0,
  quantize = false,
  colors = null
) {
  const [h, w, c] = shape;
  const count = Math.floor(Math.max(w, h) * density);
  const wormsY = new Float32Array(count);
  const wormsX = new Float32Array(count);
  const wormsStride = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    wormsY[i] = random() * (h - 1);
    wormsX[i] = random() * (w - 1);
    wormsStride[i] = randomNormal(stride, strideDeviation) * (Math.max(w, h) / 1024.0);
  }
  const colorSrc = colors ? colors : tensor;
  const src = colorSrc.read();
  const wormColors = new Float32Array(count * c);
  for (let i = 0; i < count; i++) {
    const xi = Math.floor(wormsX[i]);
    const yi = Math.floor(wormsY[i]);
    const base = (yi * w + xi) * c;
    for (let k = 0; k < c; k++) {
      wormColors[i * c + k] = src[base + k];
    }
  }
  function makeRots(beh, n) {
    const rot = new Float32Array(n);
    const base = random() * TAU;
    if (beh === 1) {
      rot.fill(base);
    } else if (beh === 2) {
      for (let i = 0; i < n; i++) {
        rot[i] = base + (Math.floor(random() * 100) % 4) * (Math.PI / 2);
      }
    } else if (beh === 3) {
      for (let i = 0; i < n; i++) {
        rot[i] = base + random() * 0.25 - 0.125;
      }
    } else if (beh === 4) {
      for (let i = 0; i < n; i++) rot[i] = random() * TAU;
    } else if (beh === 5) {
      const q = Math.floor(n * 0.25);
      rot.set(makeRots(1, q), 0);
      rot.set(makeRots(2, q), q);
      rot.set(makeRots(3, q), q * 2);
      rot.set(makeRots(4, n - q * 3), q * 3);
    } else if (beh === 10) {
      for (let i = 0; i < n; i++) rot[i] = periodicValue(time * speed, random());
    } else {
      rot.fill(base);
    }
    return rot;
  }
  const wormsRot = makeRots(behavior, count);
  const valuesArr = new Float32Array(h * w);
  const tensorData = tensor.read();
  for (let i = 0; i < h * w; i++) {
    if (c === 1) valuesArr[i] = tensorData[i];
    else {
      const base = i * c;
      const r = tensorData[base];
      const g = tensorData[base + 1] || 0;
      const b = tensorData[base + 2] || 0;
      valuesArr[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }
  }
  const indexArr = new Float32Array(h * w);
  for (let i = 0; i < h * w; i++) indexArr[i] = valuesArr[i] * TAU * kink;
  const iterations = Math.floor(Math.sqrt(Math.min(w, h)) * duration);
  const out = new Float32Array(h * w * c);
  for (let iter = 0; iter < iterations; iter++) {
    if (drunkenness) {
      const start = Math.floor(Math.min(h, w) * time * speed + iter * speed * 10);
      for (let i = 0; i < count; i++) {
        wormsRot[i] += (periodicValue(start, random()) * 2 - 1) * drunkenness * Math.PI;
      }
    }
    const exposure = iterations > 1 ? 1 - Math.abs(1 - (iter / (iterations - 1)) * 2) : 1;
    for (let i = 0; i < count; i++) {
      const yi = Math.floor(wormsY[i]) % h;
      const xi = Math.floor(wormsX[i]) % w;
      const idx = yi * w + xi;
      const base = idx * c;
      for (let k = 0; k < c; k++) {
        out[base + k] += wormColors[i * c + k] * exposure;
      }
      let next = indexArr[idx] + wormsRot[i];
      if (quantize) next = Math.round(next);
      wormsY[i] = (wormsY[i] + Math.cos(next) * wormsStride[i]) % h;
      wormsX[i] = (wormsX[i] + Math.sin(next) * wormsStride[i]) % w;
    }
  }
  let outTensor = Tensor.fromArray(tensor.ctx, out, shape);
  outTensor = normalize(outTensor);
  const d = outTensor.read();
  for (let i = 0; i < d.length; i++) d[i] = Math.sqrt(d[i]);
  outTensor = Tensor.fromArray(outTensor.ctx, d, shape);
  return blend(tensor, outTensor, alpha);
}
register('worms', worms, {
  behavior: 1,
  density: 4.0,
  duration: 4.0,
  stride: 1.0,
  strideDeviation: 0.05,
  alpha: 0.5,
  kink: 1.0,
  drunkenness: 0.0,
  quantize: false,
  colors: null,
});

export function wormhole(
  tensor,
  shape,
  time,
  speed,
  kink = 1.0,
  inputStride = 1.0,
  alpha = 1.0
) {
  const [h, w, c] = shape;
  const src = tensor.read();
  const valuesArr = new Float32Array(h * w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (c === 1) valuesArr[idx] = src[idx];
      else {
        const base = idx * c;
        const r = src[base];
        const g = src[base + 1] || 0;
        const b = src[base + 2] || 0;
        valuesArr[idx] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      }
    }
  }
  const stride = 1024 * inputStride;
  const xArr = new Int32Array(h * w);
  const yArr = new Int32Array(h * w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const deg = valuesArr[idx] * TAU * kink;
      const xo = (Math.cos(deg) + 1) * stride;
      const yo = (Math.sin(deg) + 1) * stride;
      xArr[idx] = Math.floor((x + xo)) % w;
      yArr[idx] = Math.floor((y + yo)) % h;
    }
  }
  const offs = offsetIndex(yArr, h, xArr, w);
  const out = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    const dest = (offs.y[i] * w + offs.x[i]) * c;
    const lum = valuesArr[i];
    const l2 = lum * lum;
    const base = i * c;
    for (let k = 0; k < c; k++) {
      out[dest + k] += src[base + k] * l2;
    }
  }
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < out.length; i++) {
    if (out[i] < min) min = out[i];
    if (out[i] > max) max = out[i];
  }
  if (max > min) {
    const range = max - min;
    for (let i = 0; i < out.length; i++) {
      out[i] = Math.sqrt((out[i] - min) / range);
    }
  } else {
    for (let i = 0; i < out.length; i++) {
      out[i] = Math.sqrt(out[i]);
    }
  }
  const outTensor = Tensor.fromArray(tensor.ctx, out, shape);
  return blend(tensor, outTensor, alpha);
}
register('wormhole', wormhole, { kink: 1.0, inputStride: 1.0, alpha: 1.0 });

export function vignette(tensor, shape, time, speed, brightness = 0.0, alpha = 1.0) {
  const [h, w, c] = shape;
  const norm = normalize(tensor);
  const ctx = tensor.ctx;
  if (ctx && !ctx.isCPU) {
    const gl = ctx.gl;
    const fs = `#version 300 es\nprecision highp float;\nuniform sampler2D u_tex;\nuniform float u_brightness;\nuniform float u_alpha;\nout vec4 outColor;\nvoid main(){\n vec2 res = vec2(${w}.0, ${h}.0);\n vec2 uv = gl_FragCoord.xy / res;\n vec4 color = texture(u_tex, uv);\n float dist = distance(uv, vec2(0.5,0.5)) / length(vec2(0.5,0.5));\n vec4 vignetted = mix(color, vec4(u_brightness), dist*dist);\n outColor = mix(color, vignetted, u_alpha);\n}`;
    const prog = ctx.createProgram(FULLSCREEN_VS, fs);
    const pp = ctx.pingPong(w, h);
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, norm.handle);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_tex'), 0);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_brightness'), brightness);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_alpha'), alpha);
    ctx.bindFramebuffer(pp.writeFbo, w, h);
    ctx.drawQuad();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteProgram(prog);
    return new Tensor(ctx, pp.writeTex, [h, w, c]);
  }
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
  const ctx = tensor.ctx;
  if (ctx && !ctx.isCPU) {
    const noise = values(Math.max(h, w), [h, w, 1], {
      ctx,
      time,
      seed: 0,
      speed: speed * 1000,
    });
    const gl = ctx.gl;
    const fs = `#version 300 es\nprecision highp float;\nuniform sampler2D u_tex;\nuniform sampler2D u_noise;\nuniform float u_levels;\nout vec4 outColor;\nvoid main(){\n vec2 uv = gl_FragCoord.xy / vec2(${w}.0, ${h}.0);\n vec4 c = texture(u_tex, uv);\n float n = texture(u_noise, uv).r - 0.5;\n vec4 v = c + n / u_levels;\n v = floor(clamp(v,0.0,1.0)*u_levels)/u_levels;\n outColor = v;\n}`;
    const prog = ctx.createProgram(FULLSCREEN_VS, fs);
    const pp = ctx.pingPong(w, h);
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tensor.handle);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_tex'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, noise.handle);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_noise'), 1);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_levels'), levels);
    ctx.bindFramebuffer(pp.writeFbo, w, h);
    ctx.drawQuad();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteProgram(prog);
    return new Tensor(ctx, pp.writeTex, shape);
  }
  const noise = values(Math.max(h, w), [h, w, 1], {
    ctx: tensor.ctx,
    time,
    seed: 0,
    speed: speed * 1000,
  });
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
  const ctx = tensor.ctx;
  if (ctx && !ctx.isCPU) {
    const noise = values(Math.max(h, w), [h, w, c], {
      ctx,
      time,
      speed: speed * 100,
    });
    return blend(tensor, noise, alpha);
  }
  const noise = values(Math.max(h, w), [h, w, c], {
    ctx: tensor.ctx,
    time,
    speed: speed * 100,
  });
  return blend(tensor, noise, alpha);
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
  const [h, w] = shape;
  const ctx = tensor.ctx;
  if (ctx && !ctx.isCPU) {
    const gl = ctx.gl;
    const fs = `#version 300 es\nprecision highp float;\nuniform sampler2D u_tex;\nuniform float u_amount;\nout vec4 outColor;\nvoid main(){\n vec2 uv = gl_FragCoord.xy / vec2(${w}.0, ${h}.0);\n vec4 color = texture(u_tex, uv) + u_amount;\n outColor = clamp(color, 0.0, 1.0);\n}`;
    const prog = ctx.createProgram(FULLSCREEN_VS, fs);
    const pp = ctx.pingPong(w, h);
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tensor.handle);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_tex'), 0);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_amount'), amount);
    ctx.bindFramebuffer(pp.writeFbo, w, h);
    ctx.drawQuad();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteProgram(prog);
    return new Tensor(ctx, pp.writeTex, shape);
  }
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
  const [h, w] = shape;
  const ctx = tensor.ctx;
  if (ctx && !ctx.isCPU) {
    const gl = ctx.gl;
    const fs = `#version 300 es\nprecision highp float;\nuniform sampler2D u_tex;\nuniform float u_amount;\nout vec4 outColor;\nvoid main(){\n vec2 uv = gl_FragCoord.xy / vec2(${w}.0, ${h}.0);\n vec4 color = texture(u_tex, uv);\n vec4 v = (color - 0.5) * u_amount + 0.5;\n outColor = clamp(v, 0.0, 1.0);\n}`;
    const prog = ctx.createProgram(FULLSCREEN_VS, fs);
    const pp = ctx.pingPong(w, h);
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tensor.handle);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_tex'), 0);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_amount'), amount);
    ctx.bindFramebuffer(pp.writeFbo, w, h);
    ctx.drawQuad();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteProgram(prog);
    return new Tensor(ctx, pp.writeTex, shape);
  }
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
