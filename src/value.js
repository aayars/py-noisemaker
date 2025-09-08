import { Tensor } from './tensor.js';
import { ValueDistribution, ValueMask, DistanceMetric } from './constants.js';

function fract(x) {
  return x - Math.floor(x);
}

function rand2D(x, y, seed = 0, time = 0) {
  const s = x * 374761393 + y * 668265263 + seed * 69069 + time * 43758.5453;
  return fract(Math.sin(s) * 43758.5453);
}

function maskValue(type, x, y, w, h, freq = 1) {
  switch (type) {
    case ValueMask.grid:
      return (x % freq === 0 || y % freq === 0) ? 0 : 1;
    case ValueMask.square:
    default:
      return 1;
  }
}

export function values(freq, shape, opts = {}) {
  const [height, width, channels = 1] = shape;
  const {
    distrib = ValueDistribution.uniform,
    mask,
    maskInverse = false,
    time = 0,
    seed = 0,
  } = opts;

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
          const r = rand2D(x, y, seed, time);
          val = Math.pow(r, 3);
          break;
        }
        case ValueDistribution.uniform:
        default: {
          const u = x / width * freq;
          const v = y / height * freq;
          const x0 = Math.floor(u);
          const y0 = Math.floor(v);
          const xf = u - x0;
          const yf = v - y0;
          const r00 = rand2D(x0, y0, seed, time);
          const r10 = rand2D(x0 + 1, y0, seed, time);
          const r01 = rand2D(x0, y0 + 1, seed, time);
          const r11 = rand2D(x0 + 1, y0 + 1, seed, time);
          const sx = xf * xf * (3 - 2 * xf);
          const sy = yf * yf * (3 - 2 * yf);
          const nx0 = r00 * (1 - sx) + r10 * sx;
          const nx1 = r01 * (1 - sx) + r11 * sx;
          val = nx0 * (1 - sy) + nx1 * sy;
          break;
        }
      }
      if (mask !== undefined && mask !== null) {
        const m = maskValue(mask, x, y, width, height, freq);
        val = maskInverse ? (1 - m) * val : m * val;
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
