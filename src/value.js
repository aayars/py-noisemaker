import { Tensor } from './tensor.js';
import { ValueDistribution } from './constants.js';
import { maskValues } from './masks.js';

function fract(x) {
  return x - Math.floor(x);
}

function rand2D(x, y, seed = 0, time = 0, speed = 1) {
  const s =
    x * 374761393 + y * 668265263 + seed * 69069 + time * speed * 43758.5453;
  return fract(Math.sin(s) * 43758.5453);
}

/**
 * Generate value noise or other simple distributions.
 *
 * @param {number} freq Frequency of the grid.
 * @param {[number, number, number]} shape Output tensor shape `[height,width,channels]`.
 * @param {Object} opts Options object.
 * @param {ValueDistribution} [opts.distrib=ValueDistribution.uniform] Distribution type.
 * @param {boolean} [opts.corners=false] Wrap noise coordinates for seamless tiling.
 * @param {ValueMask} [opts.mask] Optional mask to multiply the result with.
 * @param {boolean} [opts.maskInverse=false] Invert the mask values.
 * @param {boolean} [opts.maskStatic=false] Ignore time/speed when generating masks.
 * @param {number} [opts.splineOrder=3] Interpolation order (1=linear,3=cubic,5=quintic).
 * @param {number} [opts.time=0] Animation time value.
 * @param {number} [opts.seed=0] Random seed.
 * @param {number} [opts.speed=1] Time multiplier for animation.
 * @returns {Tensor} Generated tensor.
 */
export function values(freq, shape, opts = {}) {
  const [height, width, channels = 1] = shape;
  const {
    distrib = ValueDistribution.uniform,
    corners = false,
    mask,
    maskInverse = false,
    maskStatic = false,
    splineOrder = 3,
    time = 0,
    seed = 0,
    speed = 1,
  } = opts;

  let maskData = null;
  if (mask !== undefined && mask !== null) {
    const [maskTensor] = maskValues(mask, [height, width, 1], {
      inverse: maskInverse,
      time: maskStatic ? 0 : time,
      speed,
    });
    maskData = maskTensor.read();
  }

  const data = new Float32Array(height * width * channels);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let val = 0;
      switch (distrib) {
        case ValueDistribution.ones:
          val = 1;
          break;
        case ValueDistribution.mids:
          val = 0.5;
          break;
        case ValueDistribution.zeros:
          val = 0;
          break;
        case ValueDistribution.column_index:
          val = width === 1 ? 0 : x / (width - 1);
          break;
        case ValueDistribution.row_index:
          val = height === 1 ? 0 : y / (height - 1);
          break;
        case ValueDistribution.center_circle: {
          const dx = (x + 0.5) / width - 0.5;
          const dy = (y + 0.5) / height - 0.5;
          const d = Math.sqrt(dx * dx + dy * dy);
          val = Math.max(0, 1 - d * 2);
          break;
        }
        case ValueDistribution.scan_up:
          val = height === 1 ? 0 : y / (height - 1);
          break;
        case ValueDistribution.scan_down:
          val = height === 1 ? 0 : 1 - y / (height - 1);
          break;
        case ValueDistribution.scan_left:
          val = width === 1 ? 0 : x / (width - 1);
          break;
        case ValueDistribution.scan_right:
          val = width === 1 ? 0 : 1 - x / (width - 1);
          break;
        case ValueDistribution.exp: {
          const r = rand2D(x, y, seed, time, speed);
          val = Math.pow(r, 3);
          break;
        }
        case ValueDistribution.uniform:
        default: {
          const u = (x / width) * freq;
          const v = (y / height) * freq;
          const x0 = Math.floor(u);
          const y0 = Math.floor(v);
          const xf = u - x0;
          const yf = v - y0;
          const f = Math.max(1, Math.floor(freq));
          const xb = corners ? x0 % f : x0;
          const yb = corners ? y0 % f : y0;
          const x1 = corners ? (xb + 1) % f : xb + 1;
          const y1 = corners ? (yb + 1) % f : yb + 1;
          const r00 = rand2D(xb, yb, seed, time, speed);
          const r10 = rand2D(x1, yb, seed, time, speed);
          const r01 = rand2D(xb, y1, seed, time, speed);
          const r11 = rand2D(x1, y1, seed, time, speed);
          function interp(t) {
            if (splineOrder === 1) return t;
            if (splineOrder === 5)
              return t * t * t * (t * (t * 6 - 15) + 10);
            return t * t * (3 - 2 * t);
          }
          const sx = interp(xf);
          const sy = interp(yf);
          const nx0 = r00 * (1 - sx) + r10 * sx;
          const nx1 = r01 * (1 - sx) + r11 * sx;
          val = nx0 * (1 - sy) + nx1 * sy;
          break;
        }
      }
      if (maskData) {
        const m = maskData[y * width + x];
        val *= m;
      }
      for (let c = 0; c < channels; c++) {
        data[(y * width + x) * channels + c] = val;
      }
    }
  }
  return Tensor.fromArray(null, data, [height, width, channels]);
}

export function downsample(tensor, factor) {
  const [h, w, c] = tensor.shape;
  const nh = Math.floor(h / factor);
  const nw = Math.floor(w / factor);
  const src = tensor.read();
  const out = new Float32Array(nh * nw * c);
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      for (let k = 0; k < c; k++) {
        let sum = 0;
        for (let yy = 0; yy < factor; yy++) {
          for (let xx = 0; xx < factor; xx++) {
            const idx = ((y * factor + yy) * w + (x * factor + xx)) * c + k;
            sum += src[idx];
          }
        }
        out[(y * nw + x) * c + k] = sum / (factor * factor);
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [nh, nw, c]);
}

export function upsample(tensor, factor) {
  const [h, w, c] = tensor.shape;
  const nh = h * factor;
  const nw = w * factor;
  const src = tensor.read();
  const out = new Float32Array(nh * nw * c);
  for (let y = 0; y < nh; y++) {
    const y0 = Math.floor(y / factor);
    for (let x = 0; x < nw; x++) {
      const x0 = Math.floor(x / factor);
      for (let k = 0; k < c; k++) {
        out[(y * nw + x) * c + k] = src[(y0 * w + x0) * c + k];
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [nh, nw, c]);
}

export function warp(tensor, flow, amount = 1) {
  const [h, w, c] = tensor.shape;
  const src = tensor.read();
  const flowData = flow.read();
  const out = new Float32Array(h * w * c);
  function sample(x, y, k) {
    x = Math.max(0, Math.min(w - 1, x));
    y = Math.max(0, Math.min(h - 1, y));
    const ix = Math.floor(x);
    const iy = Math.floor(y);
    return src[(iy * w + ix) * c + k];
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = flowData[(y * w + x) * 2] * amount;
      const dy = flowData[(y * w + x) * 2 + 1] * amount;
      for (let k = 0; k < c; k++) {
        out[(y * w + x) * c + k] = sample(x + dx, y + dy, k);
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
}

export function blend(a, b, t) {
  const [h, w, c] = a.shape;
  const da = a.read();
  const db = b.read();
  const dt = typeof t === 'number' ? null : t.read();
  const out = new Float32Array(h * w * c);
  for (let i = 0; i < out.length; i++) {
    const tv = dt ? dt[i] : t;
    out[i] = da[i] * (1 - tv) + db[i] * tv;
  }
  return Tensor.fromArray(a.ctx, out, [h, w, c]);
}

export function normalize(tensor) {
  const [h, w, c] = tensor.shape;
  const src = tensor.read();
  const out = new Float32Array(src.length);
  for (let k = 0; k < c; k++) {
    let min = Infinity;
    let max = -Infinity;
    for (let i = k; i < src.length; i += c) {
      const v = src[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min || 1;
    for (let i = k; i < src.length; i += c) {
      out[i] = (src[i] - min) / range;
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
}

export function clamp01(tensor) {
  const src = tensor.read();
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    out[i] = Math.min(1, Math.max(0, src[i]));
  }
  return Tensor.fromArray(tensor.ctx, out, tensor.shape);
}

export function distance(dx, dy, metric = DistanceMetric.euclidean) {
  switch (metric) {
    case DistanceMetric.manhattan:
      return Math.abs(dx) + Math.abs(dy);
    case DistanceMetric.chebyshev:
      return Math.max(Math.abs(dx), Math.abs(dy));
    case DistanceMetric.octagram:
      return Math.abs(dx) + Math.abs(dy) + Math.abs(dx - dy);
    case DistanceMetric.triangular:
      return Math.abs(dx) + Math.abs(dy) + Math.abs(dx + dy);
    case DistanceMetric.euclidean:
    default:
      return Math.sqrt(dx * dx + dy * dy);
  }
}

export function sobel(tensor) {
  const [h, w, c] = tensor.shape;
  const src = tensor.read();
  const out = new Float32Array(src.length);
  const gxKernel = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const gyKernel = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  function get(x, y, k) {
    x = Math.max(0, Math.min(w - 1, x));
    y = Math.max(0, Math.min(h - 1, y));
    return src[(y * w + x) * c + k];
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let k = 0; k < c; k++) {
        let gx = 0, gy = 0, idx = 0;
        for (let yy = -1; yy <= 1; yy++) {
          for (let xx = -1; xx <= 1; xx++) {
            const v = get(x + xx, y + yy, k);
            gx += gxKernel[idx] * v;
            gy += gyKernel[idx] * v;
            idx++;
          }
        }
        out[(y * w + x) * c + k] = Math.sqrt(gx * gx + gy * gy);
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
}

export function hsvToRgb(tensor) {
  const [h, w, c] = tensor.shape;
  const src = tensor.read();
  const out = new Float32Array(h * w * 3);
  for (let i = 0; i < h * w; i++) {
    const H = src[i * c];
    const S = src[i * c + 1];
    const V = src[i * c + 2];
    const C = V * S;
    const hPrime = (H * 6) % 6;
    const X = C * (1 - Math.abs((hPrime % 2) - 1));
    let r1, g1, b1;
    switch (Math.floor(hPrime)) {
      case 0: r1 = C; g1 = X; b1 = 0; break;
      case 1: r1 = X; g1 = C; b1 = 0; break;
      case 2: r1 = 0; g1 = C; b1 = X; break;
      case 3: r1 = 0; g1 = X; b1 = C; break;
      case 4: r1 = X; g1 = 0; b1 = C; break;
      case 5: r1 = C; g1 = 0; b1 = X; break;
      default: r1 = 0; g1 = 0; b1 = 0; break;
    }
    const m = V - C;
    out[i * 3] = r1 + m;
    out[i * 3 + 1] = g1 + m;
    out[i * 3 + 2] = b1 + m;
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, 3]);
}

export function rgbToHsv(tensor) {
  const [h, w, c] = tensor.shape;
  const src = tensor.read();
  const out = new Float32Array(h * w * 3);
  for (let i = 0; i < h * w; i++) {
    const r = src[i * c];
    const g = src[i * c + 1];
    const b = src[i * c + 2];
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const d = max - min;
    let hVal;
    if (d === 0) {
      hVal = 0;
    } else if (max === r) {
      hVal = ((g - b) / d) % 6;
    } else if (max === g) {
      hVal = (b - r) / d + 2;
    } else {
      hVal = (r - g) / d + 4;
    }
    hVal /= 6;
    if (hVal < 0) hVal += 1;
    const sVal = max === 0 ? 0 : d / max;
    out[i * 3] = hVal;
    out[i * 3 + 1] = sVal;
    out[i * 3 + 2] = max;
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, 3]);
}

export function adjustHue(tensor, amount) {
  const hsv = rgbToHsv(tensor);
  const data = hsv.read();
  for (let i = 0; i < data.length; i += 3) {
    data[i] = (data[i] + amount) % 1;
    if (data[i] < 0) data[i] += 1;
  }
  return hsvToRgb(Tensor.fromArray(hsv.ctx, data, hsv.shape));
}

export function randomHue(tensor, range = 0.05) {
  const shift = random() * range * 2 - range;
  return adjustHue(tensor, shift);
}

export function valueMap(tensor, palette) {
  const [h, w, c] = tensor.shape;
  if (c !== 1) throw new Error('valueMap expects single-channel tensor');
  const src = tensor.read();
  const out = new Float32Array(h * w * 3);
  const n = palette.length;
  for (let i = 0; i < h * w; i++) {
    const idx = Math.min(n - 1, Math.max(0, Math.floor(src[i] * (n - 1))));
    const [r, g, b] = palette[idx];
    out[i * 3] = r;
    out[i * 3 + 1] = g;
    out[i * 3 + 2] = b;
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, 3]);
}

export function ridge(tensor) {
  const data = tensor.read();
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    out[i] = 1 - Math.abs(data[i] * 2 - 1);
  }
  return Tensor.fromArray(tensor.ctx, out, tensor.shape);
}

export function convolution(tensor, kernel, opts = {}) {
  const { normalize: doNormalize = true, alpha = 1 } = opts;
  const [h, w, c] = tensor.shape;
  const kh = kernel.length;
  const kw = kernel[0].length;
  let maxVal = -Infinity;
  let minVal = Infinity;
  for (const row of kernel) {
    for (const v of row) {
      if (v > maxVal) maxVal = v;
      if (v < minVal) minVal = v;
    }
  }
  const scale = Math.max(Math.abs(maxVal), Math.abs(minVal)) || 1;
  const src = tensor.read();
  const out = new Float32Array(h * w * c);
  const halfH = Math.floor(kh / 2);
  const halfW = Math.floor(kw / 2);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let k = 0; k < c; k++) {
        let sum = 0;
        for (let j = 0; j < kh; j++) {
          for (let i = 0; i < kw; i++) {
            const yy = (y + j - halfH + h) % h;
            const xx = (x + i - halfW + w) % w;
            const val = src[(yy * w + xx) * c + k];
            sum += (kernel[j][i] / scale) * val;
          }
        }
        out[(y * w + x) * c + k] = sum;
      }
    }
  }
  let result = Tensor.fromArray(tensor.ctx, out, [h, w, c]);
  if (doNormalize) result = normalize(result);
  if (alpha !== 1) result = blend(tensor, result, alpha);
  return result;
}

export function refract(tensor, referenceX = null, referenceY = null, displacement = 0.5) {
  const [h, w, c] = tensor.shape;
  const rx = (referenceX || tensor).read();
  const ry = (referenceY || tensor).read();
  const flowData = new Float32Array(h * w * 2);
  for (let i = 0; i < h * w; i++) {
    flowData[i * 2] = rx[i] * 2 - 1;
    flowData[i * 2 + 1] = ry[i] * 2 - 1;
  }
  const flow = Tensor.fromArray(tensor.ctx, flowData, [h, w, 2]);
  return warp(tensor, flow, displacement);
}

export function fft(tensor) {
  const [h, w, c] = tensor.shape;
  const src = tensor.read();
  const out = new Float32Array(h * w * c * 2);
  for (let k = 0; k < c; k++) {
    for (let u = 0; u < h; u++) {
      for (let v = 0; v < w; v++) {
        let re = 0, im = 0;
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const angle = -2 * Math.PI * ((u * y) / h + (v * x) / w);
            const val = src[(y * w + x) * c + k];
            re += val * Math.cos(angle);
            im += val * Math.sin(angle);
          }
        }
        const idx = (u * w + v) * c * 2 + k * 2;
        out[idx] = re;
        out[idx + 1] = im;
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, c * 2]);
}

export function ifft(tensor) {
  const [h, w, c2] = tensor.shape;
  const c = c2 / 2;
  const src = tensor.read();
  const out = new Float32Array(h * w * c);
  for (let k = 0; k < c; k++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let re = 0;
        for (let u = 0; u < h; u++) {
          for (let v = 0; v < w; v++) {
            const idx = (u * w + v) * c * 2 + k * 2;
            const real = src[idx];
            const imag = src[idx + 1];
            const angle = 2 * Math.PI * ((u * y) / h + (v * x) / w);
            re += real * Math.cos(angle) - imag * Math.sin(angle);
          }
        }
        out[(y * w + x) * c + k] = re / (h * w);
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
}

export function rotate(tensor, angle) {
  const [h, w, c] = tensor.shape;
  const src = tensor.read();
  const out = new Float32Array(h * w * c);
  const cx = (w - 1) / 2;
  const cy = (h - 1) / 2;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const rx = (x - cx) * cos + (y - cy) * sin + cx;
      const ry = -(x - cx) * sin + (y - cy) * cos + cy;
      const ix = Math.round(rx);
      const iy = Math.round(ry);
      for (let k = 0; k < c; k++) {
        let val = 0;
        if (ix >= 0 && ix < w && iy >= 0 && iy < h) {
          val = src[(iy * w + ix) * c + k];
        }
        out[(y * w + x) * c + k] = val;
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
}

export function zoom(tensor, factor) {
  const [h, w, c] = tensor.shape;
  const src = tensor.read();
  const out = new Float32Array(h * w * c);
  const cx = (w - 1) / 2;
  const cy = (h - 1) / 2;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const rx = (x - cx) / factor + cx;
      const ry = (y - cy) / factor + cy;
      const ix = Math.round(rx);
      const iy = Math.round(ry);
      for (let k = 0; k < c; k++) {
        let val = 0;
        if (ix >= 0 && ix < w && iy >= 0 && iy < h) {
          val = src[(iy * w + ix) * c + k];
        }
        out[(y * w + x) * c + k] = val;
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
}

export function fxaa(tensor) {
  const [h, w, c] = tensor.shape;
  const src = tensor.read();
  const out = new Float32Array(h * w * c);
  const lumWeights = [0.299, 0.587, 0.114];
  function idx(x, y, k) {
    x = Math.max(0, Math.min(w - 1, x));
    y = Math.max(0, Math.min(h - 1, y));
    return (y * w + x) * c + k;
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (c === 1) {
        const lC = src[idx(x, y, 0)];
        const lN = src[idx(x, y - 1, 0)];
        const lS = src[idx(x, y + 1, 0)];
        const lW = src[idx(x - 1, y, 0)];
        const lE = src[idx(x + 1, y, 0)];
        const wC = 1.0;
        const wN = Math.exp(-Math.abs(lC - lN));
        const wS = Math.exp(-Math.abs(lC - lS));
        const wW = Math.exp(-Math.abs(lC - lW));
        const wE = Math.exp(-Math.abs(lC - lE));
        const sum = wC + wN + wS + wW + wE + 1e-10;
        out[idx(x, y, 0)] =
          (lC * wC + lN * wN + lS * wS + lW * wW + lE * wE) / sum;
      } else if (c === 2) {
        const lum = src[idx(x, y, 0)];
        const alpha = src[idx(x, y, 1)];
        const lN = src[idx(x, y - 1, 0)];
        const lS = src[idx(x, y + 1, 0)];
        const lW = src[idx(x - 1, y, 0)];
        const lE = src[idx(x + 1, y, 0)];
        const wC = 1.0;
        const wN = Math.exp(-Math.abs(lum - lN));
        const wS = Math.exp(-Math.abs(lum - lS));
        const wW = Math.exp(-Math.abs(lum - lW));
        const wE = Math.exp(-Math.abs(lum - lE));
        const sum = wC + wN + wS + wW + wE + 1e-10;
        out[idx(x, y, 0)] =
          (lum * wC + lN * wN + lS * wS + lW * wW + lE * wE) / sum;
        out[idx(x, y, 1)] = alpha;
      } else if (c === 3 || c === 4) {
        const rgbC = [src[idx(x, y, 0)], src[idx(x, y, 1)], src[idx(x, y, 2)]];
        const rgbN = [src[idx(x, y - 1, 0)], src[idx(x, y - 1, 1)], src[idx(x, y - 1, 2)]];
        const rgbS = [src[idx(x, y + 1, 0)], src[idx(x, y + 1, 1)], src[idx(x, y + 1, 2)]];
        const rgbW = [src[idx(x - 1, y, 0)], src[idx(x - 1, y, 1)], src[idx(x - 1, y, 2)]];
        const rgbE = [src[idx(x + 1, y, 0)], src[idx(x + 1, y, 1)], src[idx(x + 1, y, 2)]];
        const lC = rgbC[0] * lumWeights[0] + rgbC[1] * lumWeights[1] + rgbC[2] * lumWeights[2];
        const lN = rgbN[0] * lumWeights[0] + rgbN[1] * lumWeights[1] + rgbN[2] * lumWeights[2];
        const lS = rgbS[0] * lumWeights[0] + rgbS[1] * lumWeights[1] + rgbS[2] * lumWeights[2];
        const lW = rgbW[0] * lumWeights[0] + rgbW[1] * lumWeights[1] + rgbW[2] * lumWeights[2];
        const lE = rgbE[0] * lumWeights[0] + rgbE[1] * lumWeights[1] + rgbE[2] * lumWeights[2];
        const wC = 1.0;
        const wN = Math.exp(-Math.abs(lC - lN));
        const wS = Math.exp(-Math.abs(lC - lS));
        const wW = Math.exp(-Math.abs(lC - lW));
        const wE = Math.exp(-Math.abs(lC - lE));
        const sum = wC + wN + wS + wW + wE + 1e-10;
        for (let k = 0; k < 3; k++) {
          out[idx(x, y, k)] =
            (rgbC[k] * wC + rgbN[k] * wN + rgbS[k] * wS + rgbW[k] * wW + rgbE[k] * wE) / sum;
        }
        if (c === 4) out[idx(x, y, 3)] = src[idx(x, y, 3)];
      } else {
        for (let k = 0; k < c; k++) out[idx(x, y, k)] = src[idx(x, y, k)];
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
}

export function gaussianBlur(tensor, radius = 1) {
  const size = radius * 2 + 1;
  const sigma = radius / 2 || 1;
  const kernel = [];
  for (let y = -radius; y <= radius; y++) {
    const row = [];
    for (let x = -radius; x <= radius; x++) {
      row.push(Math.exp(-(x * x + y * y) / (2 * sigma * sigma)));
    }
    kernel.push(row);
  }
  return convolution(tensor, kernel, { normalize: false });
}
