import { Tensor } from './tensor.js';
import {
  ValueDistribution,
  DistanceMetric,
  InterpolationType,
  isNativeSize,
} from './constants.js';
import { maskValues } from './masks.js';
import {
  random,
  setSeed as setRNGSeed,
  withTensorData,
  withTensorDatas,
} from './util.js';
import { simplex as simplexNoise, setSeed as setSimplexSeed } from './simplex.js';
import {
  VALUE_WGSL,
  RESAMPLE_WGSL,
  DOWNSAMPLE_WGSL,
  BLEND_WGSL,
  SOBEL_WGSL,
  REFRACT_WGSL,
  CONVOLUTION_WGSL,
} from './webgpu/shaders.js';

let _seed = 0x12345678;
let _opCounter = 0;

export function setSeed(s) {
  _seed = s >>> 0;
  setRNGSeed(s);
  setSimplexSeed(s);
}

export const FULLSCREEN_VS = `#version 300 es
precision highp float;
in vec2 position;
void main() { gl_Position = vec4(position, 0.0, 1.0); }`;

function fract(x) {
  return x - Math.floor(x);
}

// GPU fragment shader implementations operate in 32bit float precision. The
// original hash used very large constants which, when combined with the limited
// precision of GLSL `sin`, produced visible banding artifacts.  Using smaller
// constants keeps the argument to `sin` within a sane range while still
// providing a deterministic pseudo-random distribution.
function rand2D(x, y, seed = 0, time = 0, speed = 1) {
  // Keep the original high‑precision hash for CPU paths to preserve existing
  // test fixtures. The GPU shader uses a reduced version to avoid float
  // precision issues.  Floor coordinates to integers to match the GPU shader
  // and avoid fractional artifacts when called with non‑integer inputs.
  const sx = Math.floor(x);
  const sy = Math.floor(y);
  const s =
    sx * 12.9898 +
    sy * 78.233 +
    seed * 37.719 +
    time * speed * 0.1;
  return fract(Math.sin(s) * 43758.5453);
}

function offsetTensor(tensor, shape, x = 0, y = 0) {
  const [h, w, c] = shape;
  if (x === 0 && y === 0) return tensor;
  return withTensorData(tensor, (src) => {
    const out = new Float32Array(h * w * c);
    for (let yy = 0; yy < h; yy++) {
      const sy = (yy + y + h) % h;
      for (let xx = 0; xx < w; xx++) {
        const sx = (xx + x + w) % w;
        const dst = (yy * w + xx) * c;
        const sIdx = (sy * w + sx) * c;
        for (let k = 0; k < c; k++) {
          out[dst + k] = src[sIdx + k];
        }
      }
    }
    return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
  });
}

function pinCorners(tensor, shape, freq, corners) {
  if (
    (!corners && freq[0] % 2 === 0) ||
    (corners && freq[0] % 2 === 1)
  ) {
    const xOff = Math.floor((shape[1] / freq[1]) * 0.5);
    const yOff = Math.floor((shape[0] / freq[0]) * 0.5);
    return offsetTensor(tensor, shape, xOff, yOff);
  }
  return tensor;
}

// All ValueDistribution members are supported on the GPU path.  Keep the set
// in sync with the enumeration so any new distributions automatically opt in.
const GPU_DISTRIBS = new Set(Object.values(ValueDistribution));

export function valueNoise(count, freq = 8) {
  // RNG: freq+1 calls to random for lattice points 0..freq
  const lattice = Array.from({ length: freq + 1 }, () => random());
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const x = (i / count) * freq;
    const xi = Math.floor(x);
    const xf = x - xi;
    const t = xf * xf * (3 - 2 * xf);
    out[i] = lattice[xi] * (1 - t) + lattice[xi + 1] * t;
  }
  return out;
}

export function freqForShape(freq, shape) {
  const [height, width] = shape;
  if (height === width) {
    return [freq, freq];
  }
  if (height < width) {
    return [freq, Math.floor((freq * width) / height)];
  }
  return [Math.floor((freq * height) / width), freq];
}

/**
 * Generate value noise or other simple distributions.
 *
 * @param {number|[number, number]} freq Frequency of the grid (scalar or `[fx, fy]`).
 * @param {[number, number, number]} shape Output tensor shape `[height,width,channels]`.
 * @param {Object} opts Options object.
 * @param {ValueDistribution} [opts.distrib=ValueDistribution.simplex] Distribution type.
 * @param {boolean} [opts.corners=false] Wrap noise coordinates for seamless tiling.
 * @param {ValueMask} [opts.mask] Optional mask to multiply the result with.
 * @param {boolean} [opts.maskInverse=false] Invert the mask values.
 * @param {boolean} [opts.maskStatic=false] Ignore time/speed when generating masks.
 * @param {InterpolationType} [opts.splineOrder=InterpolationType.bicubic]
 *   Interpolation type.
 * @param {number} [opts.time=0] Animation time value.
 * @param {number} [opts.seed=0] Random seed.
 * @param {number} [opts.speed=1] Time multiplier for animation.
 * @returns {Tensor} Generated tensor.
 */
export function values(freq, shape, opts = {}) {
  const [height, width, channels = 1] = shape;
  let freqX, freqY;
  if (Array.isArray(freq)) {
    // Python's freq argument is [height, width]; mirror that here
    [freqY, freqX] = freq;
  } else {
    // Match Python's freq_for_shape behavior for scalar input
    [freqY, freqX] = freqForShape(freq, [height, width]);
  }
  const {
    ctx = null,
    distrib = ValueDistribution.simplex,
    corners = false,
    mask,
    maskInverse = false,
    maskStatic = false,
    splineOrder = InterpolationType.bicubic,
    time = 0,
    seed,
    speed = 1,
  } = opts;
  const gpuDistrib = GPU_DISTRIBS.has(distrib);
  let maskData = null;
  let maskWidth = 0;
  let maskHeight = 0;
  let maskChannels = 0;
  const interp = (t) => {
    switch (splineOrder) {
      case InterpolationType.linear:
        return t;
      case InterpolationType.cosine:
        return 0.5 - Math.cos(t * Math.PI) * 0.5;
      default:
        return t * t * (3 - 2 * t);
    }
  };
  if (mask !== undefined && mask !== null) {
    const maskShape = [
      Math.max(1, Math.floor(freqY)),
      Math.max(1, Math.floor(freqX)),
      1,
    ];
    const [maskTensor] = maskValues(mask, maskShape, {
      inverse: maskInverse,
      time: maskStatic ? 0 : time,
      speed,
    });
    [maskHeight, maskWidth, maskChannels] = maskTensor.shape;
    maskData = maskTensor.read();
  }

  if (ctx && ctx.device && gpuDistrib && channels === 1) {
    return (async () => {
      try {
        const size = width * height * channels;
        const outBuf = ctx.device.createBuffer({
          size: size * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        const maskBuf = maskData
          ? ctx.createGPUBuffer(maskData, GPUBufferUsage.STORAGE)
          : ctx.createGPUBuffer(new Float32Array([1]), GPUBufferUsage.STORAGE);
        const paramsArr = new Float32Array([
          width,
          height,
          freqX,
          freqY,
          seed ?? _seed,
          time,
          speed,
          corners ? 1 : 0,
          splineOrder,
          distrib,
          maskData ? 1 : 0,
          maskWidth,
          maskHeight,
          0,
          0,
          0,
        ]);
        const paramsBuf = ctx.createGPUBuffer(paramsArr, GPUBufferUsage.UNIFORM);
        await ctx.runCompute(
          VALUE_WGSL,
          [
            { binding: 0, resource: { buffer: outBuf } },
            { binding: 1, resource: { buffer: paramsBuf } },
            { binding: 2, resource: { buffer: maskBuf } },
          ],
          Math.ceil(size / 64),
        );
        const out = await ctx.readGPUBuffer(outBuf, size * 4);
        return Tensor.fromArray(ctx, out, [height, width, channels]);
      } catch (e) {
        console.warn('WebGPU value fallback to CPU', e);
        return values(freq, shape, { ...opts, ctx: null });
      }
    })();
  }

  const fx = Math.max(1, Math.floor(freqX));
  const fy = Math.max(1, Math.floor(freqY));
  const needsFullSize = isNativeSize(distrib);
  const initWidth = needsFullSize ? width : fx;
  const initHeight = needsFullSize ? height : fy;
  const size = initHeight * initWidth * channels;
  let tensor;
  if (distrib === ValueDistribution.simplex || distrib === ValueDistribution.exp) {
    tensor = simplexNoise([initHeight, initWidth, channels], { time, seed, speed });
    if (distrib === ValueDistribution.exp) {
      tensor = withTensorData(tensor, (data) => {
        for (let i = 0; i < data.length; i++) {
          data[i] = Math.pow(data[i], 4);
        }
        return Tensor.fromArray(null, data, [initHeight, initWidth, channels]);
      });
    }
  } else {
    const data = new Float32Array(size);
    for (let y = 0; y < initHeight; y++) {
      for (let x = 0; x < initWidth; x++) {
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
            val = initHeight === 1 ? 0 : y / (initHeight - 1);
            break;
          case ValueDistribution.row_index:
            val = initWidth === 1 ? 0 : x / (initWidth - 1);
            break;
          case ValueDistribution.center_circle:
          case ValueDistribution.center_triangle:
          case ValueDistribution.center_diamond:
          case ValueDistribution.center_square:
          case ValueDistribution.center_pentagon:
          case ValueDistribution.center_hexagon:
          case ValueDistribution.center_heptagon:
          case ValueDistribution.center_octagon:
          case ValueDistribution.center_nonagon:
          case ValueDistribution.center_decagon:
          case ValueDistribution.center_hendecagon:
          case ValueDistribution.center_dodecagon: {
            const dx = (x + 0.5) / initWidth - 0.5;
            const dy = (y + 0.5) / initHeight - 0.5;
            let metric = DistanceMetric.euclidean;
            let sdfSides = 5;
            switch (distrib) {
              case ValueDistribution.center_triangle:
                metric = DistanceMetric.triangular;
                break;
              case ValueDistribution.center_diamond:
                metric = DistanceMetric.manhattan;
                break;
              case ValueDistribution.center_square:
                metric = DistanceMetric.chebyshev;
                break;
              case ValueDistribution.center_pentagon:
                metric = DistanceMetric.sdf;
                sdfSides = 5;
                break;
              case ValueDistribution.center_hexagon:
                metric = DistanceMetric.hexagram;
                break;
              case ValueDistribution.center_heptagon:
                metric = DistanceMetric.sdf;
                sdfSides = 7;
                break;
              case ValueDistribution.center_octagon:
                metric = DistanceMetric.octagram;
                break;
              case ValueDistribution.center_nonagon:
                metric = DistanceMetric.sdf;
                sdfSides = 9;
                break;
              case ValueDistribution.center_decagon:
                metric = DistanceMetric.sdf;
                sdfSides = 10;
                break;
              case ValueDistribution.center_hendecagon:
                metric = DistanceMetric.sdf;
                sdfSides = 11;
                break;
              case ValueDistribution.center_dodecagon:
                metric = DistanceMetric.sdf;
                sdfSides = 12;
                break;
              default:
                metric = DistanceMetric.euclidean;
            }
            const d = distance(dx, dy, metric, sdfSides);
            val = Math.max(0, 1 - d * 2);
            break;
          }
          default:
            val = rand2D(x, y, seed, time, speed);
            break;
        }
        const idx = (y * initWidth + x) * channels;
        for (let c = 0; c < channels; c++) {
          data[idx + c] = val;
        }
      }
    }
    tensor = Tensor.fromArray(null, data, [initHeight, initWidth, channels]);
  }

  if (maskData) {
    let mTensor = Tensor.fromArray(null, maskData, [maskHeight, maskWidth, maskChannels]);
    if (needsFullSize) {
      mTensor = resample(mTensor, [height, width, maskChannels], splineOrder);
      mTensor = pinCorners(
        mTensor,
        [height, width, maskChannels],
        [freqY, freqX],
        corners
      );
    }
    const combine = (t) =>
      withTensorDatas([mTensor, t], (mArr, tArr) => {
        const mh = mTensor.shape[0];
        const mw = mTensor.shape[1];
        const total = mh * mw;
        if (channels === 2) {
          const out = new Float32Array(total * 2);
          for (let i = 0; i < total; i++) {
            out[i * 2] = tArr[i * channels];
            out[i * 2 + 1] = mArr[i];
          }
          return Tensor.fromArray(null, out, [mh, mw, 2]);
        } else if (channels === 4) {
          const out = new Float32Array(total * 4);
          for (let i = 0; i < total; i++) {
            out[i * 4] = tArr[i * channels];
            out[i * 4 + 1] = tArr[i * channels + 1];
            out[i * 4 + 2] = tArr[i * channels + 2];
            out[i * 4 + 3] = mArr[i];
          }
          return Tensor.fromArray(null, out, [mh, mw, 4]);
        }
        for (let i = 0; i < total; i++) {
          for (let c = 0; c < channels; c++) {
            tArr[i * channels + c] *= mArr[i];
          }
        }
        return Tensor.fromArray(null, tArr, [mh, mw, channels]);
      });
    tensor = tensor && typeof tensor.then === 'function' ? tensor.then(combine) : combine(tensor);
  }

  if (!needsFullSize) {
    tensor = resample(tensor, [height, width, channels], splineOrder);
    tensor = pinCorners(
      tensor,
      [height, width, channels],
      [freqY, freqX],
      corners
    );
  }

  if (
    distrib !== ValueDistribution.ones &&
    distrib !== ValueDistribution.mids &&
    distrib !== ValueDistribution.zeros
  ) {
    tensor = normalize(tensor);
  }

  return tensor;
}



/**
 * Resample a tensor to a new shape.
 *
 * Returns a Promise when GPU acceleration (WebGPU/WebGL) is used so callers
 * can `await` the result. CPU paths still resolve synchronously.
 *
 * @param {Tensor} tensor
 * @param {number[]} shape
 * @param {number} [splineOrder=InterpolationType.bicubic]
 * @returns {Tensor|Promise<Tensor>}
 */
export function resample(tensor, shape, splineOrder = InterpolationType.bicubic) {
  const [h, w, c] = tensor.shape;
  const [nh, nw, nc = c] = shape;
  const ctx = tensor.ctx;
  const interp = (t) => {
    switch (splineOrder) {
      case InterpolationType.linear:
        return t;
      case InterpolationType.cosine:
        return 0.5 - Math.cos(t * Math.PI) * 0.5;
      default:
        return t * t * (3 - 2 * t);
    }
  };

  const cpuResample = (src) => {
    const out = new Float32Array(nh * nw * nc);

    function sampleWrapped(ix, iy, k) {
      ix = ((ix % w) + w) % w;
      iy = ((iy % h) + h) % h;
      return src[(iy * w + ix) * c + Math.min(k, c - 1)];
    }

    function cubic(a, b, c1, d, t) {
      const t2 = t * t;
      const t3 = t2 * t;
      const a0 = d - c1 - a + b;
      const a1 = a - b - a0;
      const a2 = c1 - a;
      const a3 = b;
      return a0 * t3 + a1 * t2 + a2 * t + a3;
    }

    for (let y = 0; y < nh; y++) {
      const gy = (y * h) / nh;
      const y0 = Math.floor(gy);
      const yf = gy - y0;
      for (let x = 0; x < nw; x++) {
        const gx = (x * w) / nw;
        const x0 = Math.floor(gx);
        const xf = gx - x0;
        for (let k = 0; k < nc; k++) {
          let val;
          if (splineOrder === InterpolationType.constant) {
            val = sampleWrapped(Math.round(gx), Math.round(gy), k);
          } else if (
            splineOrder === InterpolationType.linear ||
            splineOrder === InterpolationType.cosine
          ) {
            const v00 = sampleWrapped(x0, y0, k);
            const v10 = sampleWrapped(x0 + 1, y0, k);
            const v01 = sampleWrapped(x0, y0 + 1, k);
            const v11 = sampleWrapped(x0 + 1, y0 + 1, k);
            const sx =
              splineOrder === InterpolationType.cosine
                ? 0.5 - Math.cos(xf * Math.PI) * 0.5
                : xf;
            const sy =
              splineOrder === InterpolationType.cosine
                ? 0.5 - Math.cos(yf * Math.PI) * 0.5
                : yf;
            const mx0 = v00 * (1 - sx) + v10 * sx;
            const mx1 = v01 * (1 - sx) + v11 * sx;
            val = mx0 * (1 - sy) + mx1 * sy;
          } else {
            const rows = [];
            for (let m = -1; m < 3; m++) {
              rows[m + 1] = cubic(
                sampleWrapped(x0 - 1, y0 + m, k),
                sampleWrapped(x0, y0 + m, k),
                sampleWrapped(x0 + 1, y0 + m, k),
                sampleWrapped(x0 + 2, y0 + m, k),
                xf
              );
            }
            val = cubic(rows[0], rows[1], rows[2], rows[3], yf);
          }
          out[(y * nw + x) * nc + k] = val;
        }
      }
    }

    return Tensor.fromArray(tensor.ctx, out, [nh, nw, nc]);
  };

  if (
    ctx &&
    ctx.device &&
    typeof GPUBuffer !== 'undefined' &&
    tensor.handle instanceof GPUBuffer
  ) {
    return Promise.resolve(tensor.read()).then(cpuResample);
  }

  if (ctx && ctx.device && tensor.handle instanceof GPUTexture) {
    return (async () => {
      try {
        const device = ctx.device;
        const outSize = nh * nw * nc;
        const outBuf = device.createBuffer({ size: outSize * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const paramsArr = new Float32Array([ w, h, nw, nh, nc, splineOrder, 0, 0 ]);
        const paramsBuf = ctx.createGPUBuffer(paramsArr, GPUBufferUsage.UNIFORM);
        await ctx.runCompute(
          RESAMPLE_WGSL,
          [
            { binding: 0, resource: tensor.handle.createView() },
            { binding: 1, resource: { buffer: outBuf } },
            { binding: 2, resource: { buffer: paramsBuf } },
          ],
          Math.ceil(nw / 8),
          Math.ceil(nh / 8),
        );
        const out = await ctx.readGPUBuffer(outBuf, outSize * 4);
        return Tensor.fromArray(ctx, out, [nh, nw, nc]);
      } catch (e) {
        console.warn('WebGPU resample fallback to CPU', e);
        const data = await tensor.read();
        return cpuResample(data);
      }
    })();
  }

  const srcMaybe = tensor.read();
  if (srcMaybe && typeof srcMaybe.then === 'function') {
    return srcMaybe.then(cpuResample);
  }
  return cpuResample(srcMaybe);
}
export function downsample(tensor, factor) {
  const [h, w, c] = tensor.shape;
  const ctx = tensor.ctx;
  const nh = Math.floor(h / factor);
  const nw = Math.floor(w / factor);
  const cpuDownsample = (src) => {
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
    return Tensor.fromArray(ctx, out, [nh, nw, c]);
  };
  if (ctx && ctx.device && tensor.handle instanceof GPUTexture) {
    return (async () => {
      try {
        const device = ctx.device;
        const outSize = nh * nw * c;
        const outBuf = device.createBuffer({ size: outSize * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const paramsArr = new Float32Array([w, h, nw, nh, factor, c, 0, 0]);
        const paramsBuf = ctx.createGPUBuffer(paramsArr, GPUBufferUsage.UNIFORM);
        await ctx.runCompute(
          DOWNSAMPLE_WGSL,
          [
            { binding: 0, resource: tensor.handle.createView() },
            { binding: 1, resource: { buffer: outBuf } },
            { binding: 2, resource: { buffer: paramsBuf } },
          ],
          Math.ceil(nw / 8),
          Math.ceil(nh / 8),
        );
        const out = await ctx.readGPUBuffer(outBuf, outSize * 4);
        return Tensor.fromArray(ctx, out, [nh, nw, c]);
      } catch (e) {
        console.warn('WebGPU downsample fallback to CPU', e);
        const data = await tensor.read();
        return cpuDownsample(data);
      }
    })();
  }
  return withTensorData(tensor, cpuDownsample);
}
export function proportionalDownsample(tensor, shape, newShape) {
  const [h, w, c] = shape;
  const [nh, nw] = newShape;
  const kH = Math.max(1, Math.floor(h / nh));
  const kW = Math.max(1, Math.floor(w / nw));
  const outH = Math.floor((h - kH) / kH + 1);
  const outW = Math.floor((w - kW) / kW + 1);
  const ctx = tensor.ctx;
  return withTensorData(tensor, (src) => {
    const out = new Float32Array(outH * outW * c);
    for (let y = 0; y < outH; y++) {
      for (let x = 0; x < outW; x++) {
        for (let k = 0; k < c; k++) {
          let sum = 0;
          for (let yy = 0; yy < kH; yy++) {
            for (let xx = 0; xx < kW; xx++) {
              const iy = y * kH + yy;
              const ix = x * kW + xx;
              const idx = (iy * w + ix) * c + k;
              sum += src[idx];
            }
          }
          out[(y * outW + x) * c + k] = sum / (kH * kW);
        }
      }
    }
    const down = Tensor.fromArray(ctx, out, [outH, outW, c]);
    return resample(down, [nh, nw, c]);
  });
}

export function upsample(tensor, factor, splineOrder = InterpolationType.bicubic) {
  const [h, w, c] = tensor.shape;
  return resample(tensor, [h * factor, w * factor, c], splineOrder);
}

function cubicInterpolate(a, b, c, d, t) {
  const t2 = t * t;
  const a0 = d - c - a + b;
  const a1 = a - b - a0;
  const a2 = c - a;
  const a3 = b;
  return a0 * t * t2 + a1 * t2 + a2 * t + a3;
}

export function warp(tensor, flow, amount = 1, splineOrder = InterpolationType.bicubic) {
  const [h, w, c] = tensor.shape;
  return withTensorDatas([tensor, flow], (src, flowData) => {
    const out = new Float32Array(h * w * c);

    const wrap = (v, max) => {
      v %= max;
      return v < 0 ? v + max : v;
    };

    function sample(x, y, k) {
      x = wrap(x, w);
      y = wrap(y, h);
      if (splineOrder === InterpolationType.constant) {
        const ix = wrap(Math.round(x), w);
        const iy = wrap(Math.round(y), h);
        return src[(iy * w + ix) * c + k];
      }

      const x0 = Math.floor(x);
      const y0 = Math.floor(y);
      const fx = x - x0;
      const fy = y - y0;

      if (splineOrder === InterpolationType.bicubic) {
        const get = (ix, iy) => {
          ix = wrap(ix, w);
          iy = wrap(iy, h);
          return src[(iy * w + ix) * c + k];
        };
        const col = new Array(4);
        for (let m = -1; m < 3; m++) {
          const row = new Array(4);
          for (let n = -1; n < 3; n++) {
            row[n + 1] = get(x0 + n, y0 + m);
          }
          col[m + 1] = cubicInterpolate(row[0], row[1], row[2], row[3], fx);
        }
        return cubicInterpolate(col[0], col[1], col[2], col[3], fy);
      }

      const x1 = wrap(x0 + 1, w);
      const y1 = wrap(y0 + 1, h);
      const interp = splineOrder === InterpolationType.cosine
        ? (t) => 0.5 - Math.cos(t * Math.PI) * 0.5
        : (t) => t;
      const tx = interp(fx);
      const ty = interp(fy);
      const s00 = src[(y0 * w + x0) * c + k];
      const s10 = src[(y0 * w + x1) * c + k];
      const s01 = src[(y1 * w + x0) * c + k];
      const s11 = src[(y1 * w + x1) * c + k];
      const x_y0 = s00 * (1 - tx) + s10 * tx;
      const x_y1 = s01 * (1 - tx) + s11 * tx;
      return x_y0 * (1 - ty) + x_y1 * ty;
    }

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const dx = flowData[(y * w + x) * 2] * amount * w;
        const dy = flowData[(y * w + x) * 2 + 1] * amount * h;
        for (let k = 0; k < c; k++) {
          out[(y * w + x) * c + k] = sample(x + dx, y + dy, k);
        }
      }
    }
    return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
  });
}

export function blend(a, b, t) {
  if (
    (a && typeof a.then === 'function') ||
    (b && typeof b.then === 'function') ||
    (typeof t !== 'number' && t && typeof t.then === 'function')
  ) {
    const promises = [a, b];
    if (typeof t !== 'number' && t) promises.push(t);
    return Promise.all(promises).then((arr) =>
      blend(arr[0], arr[1], typeof t === 'number' ? t : arr[2])
    );
  }

  const [h, w, c] = a.shape;
  const ctx = a.ctx;
  const bChannels = b.shape[2];

  if (typeof t === 'number') {
    if (t <= 0) return a;
    if (t >= 1) return b;
  }

  const cpuBlend = (da, db, dt) => {
    const [bh, bw, bc] = b.shape;
    const [th, tw, tc] = dt ? t.shape : [0, 0, 0];
    const out = new Float32Array(h * w * c);
    for (let y = 0; y < h; y++) {
      const by = y % bh;
      const ty = dt ? y % th : 0;
      for (let x = 0; x < w; x++) {
        const bx = x % bw;
        const tx = dt ? x % tw : 0;
        const baseA = (y * w + x) * c;
        const baseB = (by * bw + bx) * bc;
        const baseT = dt ? (ty * tw + tx) * tc : 0;
        for (let k = 0; k < c; k++) {
          const aVal = da[baseA + k];
          const bVal = db[baseB + (k < bc ? k : 0)];
          const tVal = dt ? dt[baseT + (k < tc ? k : 0)] : t;
          out[baseA + k] = Math.fround(aVal * (1 - tVal) + bVal * tVal);
        }
      }
    }
    return Tensor.fromArray(ctx, out, [h, w, c]);
  };

  if (
    ctx &&
    ctx.device &&
    typeof GPUBuffer !== 'undefined' &&
    a.handle instanceof GPUBuffer &&
    b.ctx === ctx &&
    b.handle instanceof GPUBuffer &&
    (typeof t === 'number' || (t.ctx === ctx && t.handle instanceof GPUBuffer))
  ) {
    if (typeof t === 'number') {
      return Promise.all([a.read(), b.read()]).then(([da, db]) =>
        cpuBlend(da, db, null)
      );
    }
    return Promise.all([a.read(), b.read(), t.read()]).then(([da, db, dt]) =>
      cpuBlend(da, db, dt)
    );
  }

  if (
    ctx &&
    ctx.device &&
    b.ctx === ctx &&
    typeof t === 'number' &&
    bChannels === c &&
    a.handle instanceof GPUTexture &&
    b.handle instanceof GPUTexture
  ) {
    return (async () => {
      try {
        const device = ctx.device;
        const outSize = h * w * c;
        const outBuf = device.createBuffer({
          size: outSize * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const paramsArr = new Float32Array([w, h, c, t, 0, 0, 0, 0]);
        const paramsBuf = ctx.createGPUBuffer(
          paramsArr,
          GPUBufferUsage.UNIFORM,
        );
        await ctx.runCompute(
          BLEND_WGSL,
          [
            { binding: 0, resource: a.handle.createView() },
            { binding: 1, resource: b.handle.createView() },
            { binding: 2, resource: { buffer: outBuf } },
            { binding: 3, resource: { buffer: paramsBuf } },
          ],
          Math.ceil(w / 8),
          Math.ceil(h / 8),
        );
        const out = await ctx.readGPUBuffer(outBuf, outSize * 4);
        return Tensor.fromArray(ctx, out, [h, w, c]);
      } catch (e) {
        console.warn('WebGPU blend fallback to CPU', e);
        const [da, db] = await Promise.all([a.read(), b.read()]);
        return cpuBlend(da, db, null);
      }
    })();
  }

  const daMaybe = a.read();
  const dbMaybe = b.read();
  const dtMaybe = typeof t === 'number' ? null : t.read();
  if (
    (daMaybe && typeof daMaybe.then === 'function') ||
    (dbMaybe && typeof dbMaybe.then === 'function') ||
    (dtMaybe && typeof dtMaybe.then === 'function')
  ) {
    const promises = [daMaybe, dbMaybe];
    if (dtMaybe) promises.push(dtMaybe);
    return Promise.all(promises).then((arr) => {
      const da = arr[0];
      const db = arr[1];
      const dt = dtMaybe ? arr[2] : null;
      return cpuBlend(da, db, dt);
    });
  }
  return cpuBlend(daMaybe, dbMaybe, dtMaybe);
}

export function normalize(tensor) {
  if (tensor && typeof tensor.then === 'function') {
    return tensor.then(normalize);
  }
  const [h, w, c] = tensor.shape;
  const compute = (src) => {
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < src.length; i++) {
      const v = src[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    min = Math.fround(min);
    max = Math.fround(max);
    const range = Math.fround(max - min) || Math.fround(1);
    const out = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) {
      const diff = Math.fround(src[i] - min);
      out[i] = Math.fround(diff / range);
    }
    return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
  };
  const srcMaybe = tensor.read();
  if (srcMaybe && typeof srcMaybe.then === 'function') {
    return srcMaybe.then(compute);
  }
  return compute(srcMaybe);
}

export function clamp01(tensor) {
  const compute = (src) => {
    const out = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) {
      out[i] = Math.min(1, Math.max(0, src[i]));
    }
    return Tensor.fromArray(tensor.ctx, out, tensor.shape);
  };
  const srcMaybe = tensor.read();
  if (srcMaybe && typeof srcMaybe.then === 'function') {
    return srcMaybe.then(compute);
  }
  return compute(srcMaybe);
}

export function distance(
  dx,
  dy,
  metric = DistanceMetric.euclidean,
  sdfSides = 5
) {
  switch (metric) {
    case DistanceMetric.manhattan:
      return Math.abs(dx) + Math.abs(dy);
    case DistanceMetric.chebyshev:
      return Math.max(Math.abs(dx), Math.abs(dy));
    case DistanceMetric.octagram:
      return Math.max(
        (Math.abs(dx) + Math.abs(dy)) / Math.SQRT2,
        Math.max(Math.abs(dx), Math.abs(dy))
      );
    case DistanceMetric.triangular:
      return Math.max(Math.abs(dx) - dy * 0.5, dy);
    case DistanceMetric.hexagram:
      return Math.max(
        Math.max(Math.abs(dx) - dy * 0.5, dy),
        Math.max(Math.abs(dx) + dy * 0.5, -dy)
      );
    case DistanceMetric.sdf: {
      const arctan = Math.atan2(dx, -dy) + Math.PI;
      const r = (Math.PI * 2) / sdfSides;
      return (
        Math.cos(Math.floor(0.5 + arctan / r) * r - arctan) *
        Math.sqrt(dx * dx + dy * dy)
      );
    }
    case DistanceMetric.euclidean:
    default: {
      const sum = Math.fround(Math.fround(dx * dx) + Math.fround(dy * dy));
      return Math.fround(Math.sqrt(sum));
    }
  }
}

export function sobel(tensor) {
  if (tensor && typeof tensor.then === 'function') {
    return tensor.then(sobel);
  }
  const [h, w, c] = tensor.shape;
  const ctx = tensor.ctx;
  const cpuSobel = (src) => {
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
          let gx = 0,
            gy = 0,
            idx = 0;
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
    return Tensor.fromArray(ctx, out, [h, w, c]);
  };
  if (ctx && ctx.device && tensor.handle instanceof GPUTexture) {
    return (async () => {
      try {
        const device = ctx.device;
        const outSize = h * w * c;
        const outBuf = device.createBuffer({
          size: outSize * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const paramsArr = new Float32Array([w, h, c, 0, 0, 0, 0, 0]);
        const paramsBuf = ctx.createGPUBuffer(
          paramsArr,
          GPUBufferUsage.UNIFORM,
        );
        await ctx.runCompute(
          SOBEL_WGSL,
          [
            { binding: 0, resource: tensor.handle.createView() },
            { binding: 1, resource: { buffer: outBuf } },
            { binding: 2, resource: { buffer: paramsBuf } },
          ],
          Math.ceil(w / 8),
          Math.ceil(h / 8),
        );
        const out = await ctx.readGPUBuffer(outBuf, outSize * 4);
        return Tensor.fromArray(ctx, out, [h, w, c]);
      } catch (e) {
        console.warn('WebGPU sobel fallback to CPU', e);
        const data = await tensor.read();
        return cpuSobel(data);
      }
    })();
  }
  return withTensorData(tensor, cpuSobel);
}

export function hsvToRgb(tensor) {
  const [h, w, c] = tensor.shape;
  const compute = (src) => {
    const out = new Float32Array(h * w * 3);
    for (let i = 0; i < h * w; i++) {
      const H = Math.fround(src[i * c]);
      const S = Math.fround(src[i * c + 1]);
      const V = Math.fround(src[i * c + 2]);
      const C = Math.fround(V * S);
      const hPrime = Math.fround(Math.fround(H * 6) % 6);
      const X = Math.fround(
        C * Math.fround(1 - Math.abs(Math.fround(hPrime % 2) - 1)),
      );
      let r1, g1, b1;
      switch (Math.floor(hPrime)) {
        case 0:
          r1 = C;
          g1 = X;
          b1 = 0;
          break;
        case 1:
          r1 = X;
          g1 = C;
          b1 = 0;
          break;
        case 2:
          r1 = 0;
          g1 = C;
          b1 = X;
          break;
        case 3:
          r1 = 0;
          g1 = X;
          b1 = C;
          break;
        case 4:
          r1 = X;
          g1 = 0;
          b1 = C;
          break;
        case 5:
          r1 = C;
          g1 = 0;
          b1 = X;
          break;
        default:
          r1 = 0;
          g1 = 0;
          b1 = 0;
          break;
      }
      const m = Math.fround(V - C);
      out[i * 3] = Math.fround(r1 + m);
      out[i * 3 + 1] = Math.fround(g1 + m);
      out[i * 3 + 2] = Math.fround(b1 + m);
    }
    return Tensor.fromArray(tensor.ctx, out, [h, w, 3]);
  };
  const srcMaybe = tensor.read();
  if (srcMaybe && typeof srcMaybe.then === 'function') {
    return srcMaybe.then(compute);
  }
  return compute(srcMaybe);
}

export function rgbToHsv(tensor) {
  const [h, w, c] = tensor.shape;
  const compute = (src) => {
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
        hVal = Math.fround(((g - b) / d) % 6);
      } else if (max === g) {
        hVal = Math.fround((b - r) / d + 2);
      } else {
        hVal = Math.fround((r - g) / d + 4);
      }
      hVal = Math.fround(hVal / 6);
      if (hVal < 0) hVal += 1;
      const sVal = max === 0 ? 0 : Math.fround(d / max);
      out[i * 3] = Math.fround(hVal);
      out[i * 3 + 1] = Math.fround(sVal);
      out[i * 3 + 2] = Math.fround(max);
    }
    return Tensor.fromArray(tensor.ctx, out, [h, w, 3]);
  };
  const srcMaybe = tensor.read();
  if (srcMaybe && typeof srcMaybe.then === 'function') {
    return srcMaybe.then(compute);
  }
  return compute(srcMaybe);
}

export function adjustHue(tensor, amount) {
  const hsvMaybe = rgbToHsv(tensor);
  const process = (hsv) => {
    const dataMaybe = hsv.read();
    const shift = (data) => {
      for (let i = 0; i < data.length; i += 3) {
        data[i] = (data[i] + amount) % 1;
        if (data[i] < 0) data[i] += 1;
      }
      return hsvToRgb(Tensor.fromArray(hsv.ctx, data, hsv.shape));
    };
    if (dataMaybe && typeof dataMaybe.then === 'function') {
      return dataMaybe.then(shift);
    }
    return shift(dataMaybe);
  };
  if (hsvMaybe && typeof hsvMaybe.then === 'function') {
    return hsvMaybe.then(process);
  }
  return process(hsvMaybe);
}

export function randomHue(tensor, range = 0.05) {
  const shift = random() * range * 2 - range;
  return adjustHue(tensor, shift);
}

export function valueMap(tensor, palette) {
  const [h, w, c] = tensor.shape;
  if (c !== 1) throw new Error('valueMap expects single-channel tensor');
  const compute = (src) => {
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
  };
  const srcMaybe = tensor.read();
  if (srcMaybe && typeof srcMaybe.then === 'function') {
    return srcMaybe.then(compute);
  }
  return compute(srcMaybe);
}

export function ridge(tensor) {
  return withTensorData(tensor, (data) => {
    const out = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
      out[i] = 1 - Math.abs(data[i] * 2 - 1);
    }
    return Tensor.fromArray(tensor.ctx, out, tensor.shape);
  });
}

export function convolution(tensor, kernel, opts = {}) {
  const { normalize: doNormalize = true, alpha = 1 } = opts;

  const handle = (t) => {
    const [h, w, c] = t.shape;
    const ctx = t.ctx;
    const kh = kernel.length;
    const kw = kernel[0].length;
    const halfH = Math.floor(kh / 2);
    const halfW = Math.floor(kw / 2);

    const cpuCompute = (src) => {
      const out = new Float32Array(h * w * c);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          for (let k = 0; k < c; k++) {
            let sum = 0;
            for (let j = 0; j < kh; j++) {
              for (let i = 0; i < kw; i++) {
                const yy = (y + j - halfH + h) % h;
                const xx = (x + i - halfW + w) % w;
                const val = src[(yy * w + xx) * c + k];
                const contrib = Math.fround(kernel[j][i] * val);
                sum = Math.fround(sum + contrib);
              }
            }
            out[(y * w + x) * c + k] = Math.fround(sum);
          }
        }
      }
      let result = Tensor.fromArray(t.ctx, out, [h, w, c]);
      if (doNormalize) {
        const norm = normalize(result);
        if (norm && typeof norm.then === 'function') {
          return norm.then((r) => {
            if (alpha !== 1) {
              const blended = blend(t, r, alpha);
              return blended && typeof blended.then === 'function'
                ? blended
                : blended;
            }
            return r;
          });
        }
        result = norm;
      }
      if (alpha !== 1) {
        const blended = blend(t, result, alpha);
        if (blended && typeof blended.then === 'function') {
          return blended;
        }
        return blended;
      }
      return result;
    };

    if (
      ctx &&
      ctx.device &&
      typeof GPUTexture !== 'undefined' &&
      t.handle instanceof GPUTexture
    ) {
      return (async () => {
        try {
          const device = ctx.device;
          const outSize = h * w * c;
          const outBuf = device.createBuffer({
            size: outSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
          });
          const kernelArr = new Float32Array(kh * kw);
          let idx = 0;
          let kSum = 0;
          for (let j = 0; j < kh; j++) {
            for (let i = 0; i < kw; i++) {
              const v = kernel[j][i];
              kernelArr[idx++] = v;
              kSum += v;
            }
          }
          const kernelBuf = ctx.createGPUBuffer(
            kernelArr,
            GPUBufferUsage.STORAGE,
          );
          const paramsArr = new Float32Array([
            w,
            h,
            c,
            kw,
            kh,
            doNormalize ? 1 : 0,
            alpha,
            kSum,
          ]);
          const paramsBuf = ctx.createGPUBuffer(
            paramsArr,
            GPUBufferUsage.UNIFORM,
          );
          await ctx.runCompute(
            CONVOLUTION_WGSL,
            [
              { binding: 0, resource: t.handle.createView() },
              { binding: 1, resource: { buffer: outBuf } },
              { binding: 2, resource: { buffer: kernelBuf } },
              { binding: 3, resource: { buffer: paramsBuf } },
            ],
            Math.ceil(w / 8),
            Math.ceil(h / 8),
          );
          const out = await ctx.readGPUBuffer(outBuf, outSize * 4);
          return Tensor.fromArray(ctx, out, [h, w, c]);
        } catch (e) {
          console.warn('WebGPU convolution fallback to CPU', e);
          const src = await t.read();
          return cpuCompute(src);
        }
      })();
    }

    return withTensorData(t, cpuCompute);
  };

  if (tensor && typeof tensor.then === 'function') {
    return tensor.then(handle);
  }
  return handle(tensor);
}

export function refract(
  tensor,
  referenceX = null,
  referenceY = null,
  displacement = 0.5,
  splineOrder = InterpolationType.bicubic,
  signedRange = true,
) {
  const run = (t, rx = t, ry = t) => {
    const [h, w, c] = t.shape;
    const ctx = t.ctx;
    const cpuRefract = (src, refx, refy) => {
      const out = new Float32Array(h * w * c);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          let vx = refx[y * w + x];
          let vy = refy[y * w + x];
          if (signedRange) {
            vx = vx * 2 - 1;
            vy = vy * 2 - 1;
          } else {
            vx *= 2;
            vy *= 2;
          }
          const dx = Math.fround(vx * displacement * w);
          const dy = Math.fround(vy * displacement * h);
          let x0 = x + Math.trunc(dx);
          let y0 = y + Math.trunc(dy);
          let x1 = x0 + 1;
          let y1 = y0 + 1;
          x0 = ((x0 % w) + w) % w;
          x1 = ((x1 % w) + w) % w;
          y0 = ((y0 % h) + h) % h;
          y1 = ((y1 % h) + h) % h;
          const fx = Math.fround(dx - Math.floor(dx));
          const fy = Math.fround(dy - Math.floor(dy));
          const tx = splineOrder === InterpolationType.cosine
            ? Math.fround(0.5 - Math.cos(fx * Math.PI) * 0.5)
            : fx;
          const ty = splineOrder === InterpolationType.cosine
            ? Math.fround(0.5 - Math.cos(fy * Math.PI) * 0.5)
            : fy;
          for (let k = 0; k < c; k++) {
            const s00 = src[(y0 * w + x0) * c + k];
            const s10 = src[(y0 * w + x1) * c + k];
            const s01 = src[(y1 * w + x0) * c + k];
            const s11 = src[(y1 * w + x1) * c + k];
            const x_y0 = s00 * (1 - tx) + s10 * tx;
            const x_y1 = s01 * (1 - tx) + s11 * tx;
            out[(y * w + x) * c + k] = x_y0 * (1 - ty) + x_y1 * ty;
          }
        }
      }
      return Tensor.fromArray(t.ctx, out, [h, w, c]);
    };

    if (ctx && ctx.device) {
      let rxTex = rx;
      let ryTex = ry;
      const prep = [];
      if (rxTex.ctx !== ctx) {
        const rxData = rx.read();
        if (rxData && typeof rxData.then === 'function') {
          prep.push(
            rxData.then((d) => {
              rxTex = Tensor.fromArray(ctx, d, rx.shape);
            }),
          );
        } else {
          rxTex = Tensor.fromArray(ctx, rxData, rx.shape);
        }
      }
      if (ryTex.ctx !== ctx) {
        const ryData = ry.read();
        if (ryData && typeof ryData.then === 'function') {
          prep.push(
            ryData.then((d) => {
              ryTex = Tensor.fromArray(ctx, d, ry.shape);
            }),
          );
        } else {
          ryTex = Tensor.fromArray(ctx, ryData, ry.shape);
        }
      }
      const runGPU = () => {
        if (
          t.handle instanceof GPUTexture &&
          rxTex.handle instanceof GPUTexture &&
          ryTex.handle instanceof GPUTexture
        ) {
          return (async () => {
            try {
              const device = ctx.device;
              const outSize = h * w * c;
              const outBuf = device.createBuffer({
                size: outSize * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
              });
              const paramsArr = new Float32Array([
                w,
                h,
                c,
                displacement,
                signedRange ? 1 : 0,
                splineOrder,
                0,
                0,
              ]);
              const paramsBuf = ctx.createGPUBuffer(
                paramsArr,
                GPUBufferUsage.UNIFORM,
              );
              await ctx.runCompute(
                REFRACT_WGSL,
                [
                  { binding: 0, resource: t.handle.createView() },
                  { binding: 1, resource: rxTex.handle.createView() },
                  { binding: 2, resource: ryTex.handle.createView() },
                  { binding: 3, resource: { buffer: outBuf } },
                  { binding: 4, resource: { buffer: paramsBuf } },
                ],
                Math.ceil(w / 8),
                Math.ceil(h / 8),
              );
              const out = await ctx.readGPUBuffer(outBuf, outSize * 4);
              return Tensor.fromArray(ctx, out, [h, w, c]);
            } catch (e) {
              console.warn('WebGPU refract fallback to CPU', e);
              const [s, rxData, ryData] = await Promise.all([
                t.read(),
                rxTex.read(),
                ryTex.read(),
              ]);
              return cpuRefract(s, rxData, ryData);
            }
          })();
        }
        return withTensorDatas([t, rx, ry], cpuRefract);
      };
      return prep.length ? Promise.all(prep).then(runGPU) : runGPU();
    }
    return withTensorDatas([t, rx, ry], cpuRefract);
  };

  if (
    (tensor && typeof tensor.then === 'function') ||
    (referenceX && typeof referenceX.then === 'function') ||
    (referenceY && typeof referenceY.then === 'function')
  ) {
    return Promise.all([tensor, referenceX, referenceY]).then(([t, rx, ry]) =>
      run(t, rx || t, ry || t),
    );
  }
  return run(tensor, referenceX || tensor, referenceY || tensor);
}

export function fft(tensor) {
  const [h, w, c] = tensor.shape;
  return withTensorData(tensor, (src) => {
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
  });
}

export function ifft(tensor) {
  const [h, w, c2] = tensor.shape;
  const c = c2 / 2;
  return withTensorData(tensor, (src) => {
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
  });
}

export function rotate(tensor, angle) {
  const [h, w, c] = tensor.shape;
  return withTensorData(tensor, (src) => {
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
  });
}

export function zoom(tensor, factor) {
  const [h, w, c] = tensor.shape;
  return withTensorData(tensor, (src) => {
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
  });
}

export function fxaa(tensor) {
  const [h, w, c] = tensor.shape;
  return withTensorData(tensor, (src) => {
    const out = new Float32Array(h * w * c);
    const lumWeights = [0.299, 0.587, 0.114];
    function reflect(i, n) {
      if (n === 1) return 0;
      const m = 2 * n - 2;
      i = ((i % m) + m) % m;
      return i < n ? i : m - i;
    }
    function idx(x, y, k) {
      x = reflect(x, w);
      y = reflect(y, h);
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
  });
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
