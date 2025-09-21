import { Tensor, markPresentationNormalized } from "./tensor.js";
import {
  warp as warpOp,
  sobel as sobelValue,
  normalize,
  blend,
  values,
  freqForShape,
  adjustHue,
  rgbToHsv,
  hsvToRgb,
  clamp01,
  distance,
  ridge,
  downsample,
  upsample,
  resample,
  proportionalDownsample,
  refract as refractOp,
  toValueMap,
  convolution,
  fxaa,
} from "./value.js";
import { PALETTES } from "./palettes.js";
import { register } from "./effectsRegistry.js";
import { random as simplexRandom } from "./simplex.js";
import { getAtlas, maskValues, maskShape } from "./masks.js";
import { loadGlyphs } from "./glyphs.js";
import {
  random,
  randomInt,
  uniform as randomUniform,
  normal as randomNormalArray,
  fromSRGB,
  toSRGB,
  withTensorData,
  withTensorDatas,
} from "./util.js";
import { pointCloud } from "./points.js";
import { rgbToOklab } from "./oklab.js";
import { getBufferPool } from "./bufferPool.js";

import {
  InterpolationType,
  DistanceMetric,
  ValueMask,
  ValueDistribution,
  PointDistribution,
  VoronoiDiagramType,
  WormBehavior,
} from "./constants.js";
import {
  VORONOI_WGSL,
  EROSION_WORMS_WGSL,
  WORMS_WGSL,
  REINDEX_WGSL,
  RIPPLE_WGSL,
  COLOR_MAP_WGSL,
  VIGNETTE_WGSL,
  DITHER_WGSL,
  ADJUST_BRIGHTNESS_WGSL,
  ADJUST_CONTRAST_WGSL,
  ROTATE_WGSL,
  GLYPH_MAP_WGSL,
  WARP_WGSL,
  SPATTER_MASK_WGSL,
  SCRATCHES_MASK_WGSL,
  SCRATCHES_BLEND_WGSL,
  GRIME_MASK_WGSL,
  GRIME_BLEND_WGSL,
  DERIVATIVE_WGSL,
  PIXEL_SORT_WGSL,
  KALEIDO_WGSL,
  NORMAL_MAP_WGSL,
  CRT_WGSL,
  WOBBLE_WGSL,
  VORTEX_WGSL,
  WORMHOLE_WGSL,
  DLA_WGSL,
  REVERB_WGSL,
  VASELINE_BLUR_WGSL,
  VASELINE_MASK_WGSL,
  LENS_DISTORTION_WGSL,
  DEGAUSS_WGSL,
  TINT_WGSL,
  VHS_WGSL,
  UNARY_OP_WGSL,
  BINARY_OP_WGSL,
  GRAYSCALE_WGSL,
  EXPAND_CHANNELS_WGSL,
} from "./webgpu/shaders.js";

const UNARY_OP_INVERT = 0;
const UNARY_OP_SQUARE = 1;
const UNARY_OP_CLAMP01 = 2;
const BINARY_OP_MIN_ONE_MINUS = 0;

function ensureTextureTensor(tensor) {
  const ctx = tensor?.ctx;
  if (!ctx || !ctx.device) return tensor;
  if (
    typeof GPUTexture !== "undefined" &&
    tensor.handle instanceof GPUTexture
  ) {
    return tensor;
  }
  if (typeof GPUBuffer !== "undefined" && tensor.handle instanceof GPUBuffer) {
    return Tensor.fromGPUBuffer(ctx, tensor.handle, tensor.shape);
  }
  return tensor;
}

async function runUnaryShader(tensor, op, target = null) {
  const ctx = tensor?.ctx;
  if (!ctx || !ctx.device) {
    throw new Error("WebGPU context required for unary shader");
  }
  const tex = ensureTextureTensor(tensor);
  const [h, w, c] = tex.shape;
  const pool = getBufferPool(ctx);
  const outBuf = pool.acquire(
    h * w * c * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  );
  const paramsArr = new Uint32Array([w, h, c, op]);
  const paramsBuf = pool.acquire(
    paramsArr.byteLength,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  );
  ctx.queue.writeBuffer(paramsBuf, 0, paramsArr);
  let result;
  try {
    await ctx.runCompute(
      UNARY_OP_WGSL,
      [
        { binding: 0, resource: tex.handle.createView() },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: paramsBuf } },
      ],
      Math.ceil(w / 8),
      Math.ceil(h / 8),
    );
    result = Tensor.fromGPUBuffer(ctx, outBuf, [h, w, c], target || undefined);
  } finally {
    pool.release(outBuf);
    pool.release(paramsBuf);
  }
  return target || result;
}

async function runBinaryShader(a, b, op, target = null) {
  const ctx = a?.ctx;
  if (!ctx || !ctx.device || !b || b.ctx !== ctx) {
    throw new Error("WebGPU context required for binary shader");
  }
  const texA = ensureTextureTensor(a);
  const texB = ensureTextureTensor(b);
  const [h, w, c] = texA.shape;
  const pool = getBufferPool(ctx);
  const outBuf = pool.acquire(
    h * w * c * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  );
  const paramsArr = new Uint32Array([w, h, c, op]);
  const paramsBuf = pool.acquire(
    paramsArr.byteLength,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  );
  ctx.queue.writeBuffer(paramsBuf, 0, paramsArr);
  let result;
  try {
    await ctx.runCompute(
      BINARY_OP_WGSL,
      [
        { binding: 0, resource: texA.handle.createView() },
        { binding: 1, resource: texB.handle.createView() },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: paramsBuf } },
      ],
      Math.ceil(w / 8),
      Math.ceil(h / 8),
    );
    result = Tensor.fromGPUBuffer(ctx, outBuf, [h, w, c], target || undefined);
  } finally {
    pool.release(outBuf);
    pool.release(paramsBuf);
  }
  return target || result;
}

async function runGrayscaleShader(tensor, srcChannels) {
  const ctx = tensor?.ctx;
  if (!ctx || !ctx.device) {
    throw new Error("WebGPU context required for grayscale shader");
  }
  const tex = ensureTextureTensor(tensor);
  const [h, w] = tex.shape;
  const pool = getBufferPool(ctx);
  const outBuf = pool.acquire(
    h * w * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  );
  const paramsArr = new Uint32Array([w, h, srcChannels, 0]);
  const paramsBuf = pool.acquire(
    paramsArr.byteLength,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  );
  ctx.queue.writeBuffer(paramsBuf, 0, paramsArr);
  let result;
  try {
    await ctx.runCompute(
      GRAYSCALE_WGSL,
      [
        { binding: 0, resource: tex.handle.createView() },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: paramsBuf } },
      ],
      Math.ceil(w / 8),
      Math.ceil(h / 8),
    );
    result = Tensor.fromGPUBuffer(ctx, outBuf, [h, w, 1]);
  } finally {
    pool.release(outBuf);
    pool.release(paramsBuf);
  }
  return result;
}

async function expandChannelsShader(tensor, channels) {
  const ctx = tensor?.ctx;
  if (!ctx || !ctx.device) {
    throw new Error("WebGPU context required for expand shader");
  }
  const tex = ensureTextureTensor(tensor);
  const [h, w] = tex.shape;
  const pool = getBufferPool(ctx);
  const outBuf = pool.acquire(
    h * w * channels * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  );
  const paramsArr = new Uint32Array([w, h, channels, 0]);
  const paramsBuf = pool.acquire(
    paramsArr.byteLength,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  );
  ctx.queue.writeBuffer(paramsBuf, 0, paramsArr);
  let result;
  try {
    await ctx.runCompute(
      EXPAND_CHANNELS_WGSL,
      [
        { binding: 0, resource: tex.handle.createView() },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: paramsBuf } },
      ],
      Math.ceil(w / 8),
      Math.ceil(h / 8),
    );
    result = Tensor.fromGPUBuffer(ctx, outBuf, [h, w, channels]);
  } finally {
    pool.release(outBuf);
    pool.release(paramsBuf);
  }
  return result;
}


export async function warpWebGPU(
  tensor,
  shape,
  time,
  speed,
  freq = 2,
  octaves = 5,
  displacement = 1,
  splineOrder = InterpolationType.bicubic,
  signedRange = true,
) {
  const ctx = tensor.ctx;
  const ensureGPU = async (t) => {
    if (!t.handle || typeof t.handle.createView !== "function") {
      const data = await t.read();
      return Tensor.fromArray(ctx, data, t.shape);
    }
    return t;
  };
  const baseFreq = Array.isArray(freq) ? freq : freqForShape(freq, shape);
  const [h, w, c] = shape;
  const flows = [];
  for (let octave = 1; octave <= octaves; octave++) {
    const mult = 2 ** octave;
    const f = [
      Math.floor(baseFreq[0] * 0.5 * mult),
      Math.floor(baseFreq[1] * 0.5 * mult),
    ];
    if (f[0] >= h || f[1] >= w) break;
    const opts = { ctx, time, speed, splineOrder };
    let rx = await values(f, [h, w, 1], opts);
    let ry = await values(f, [h, w, 1], opts);
    rx = await ensureGPU(rx);
    ry = await ensureGPU(ry);
    flows.push({ rx, ry, mult });
  }
  let out = await ensureGPU(tensor);
  const outSize = h * w * c * 4;
  const pool = getBufferPool(ctx);
  const outBuf = pool.acquire(outSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const paramsArr = new Float32Array(8);
  const paramsBuf = pool.acquire(
    paramsArr.byteLength,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  );
  if (flows.length) {
    await ctx.withEncoder(async () => {
      for (const { rx, ry, mult } of flows) {
        const disp = displacement / mult;
        paramsArr.set([
          w,
          h,
          c,
          disp,
          signedRange ? 1 : 0,
          0,
          0,
          0,
        ]);
        ctx.queue.writeBuffer(paramsBuf, 0, paramsArr);
        await ctx.runCompute(
          WARP_WGSL,
          [
            { binding: 0, resource: out.handle.createView() },
            { binding: 1, resource: rx.handle.createView() },
            { binding: 2, resource: ry.handle.createView() },
            { binding: 3, resource: { buffer: outBuf } },
            { binding: 4, resource: { buffer: paramsBuf } },
          ],
          Math.ceil(w / 8),
          Math.ceil(h / 8),
        );
        out = Tensor.fromGPUBuffer(ctx, outBuf, [h, w, c], out);
      }
    });
  }
  pool.release(outBuf);
  pool.release(paramsBuf);
  return out;
}

export async function warp(
  tensor,
  shape,
  time,
  speed,
  freq = 2,
  octaves = 5,
  displacement = 1,
  splineOrder = InterpolationType.bicubic,
  signedRange = true,
) {
  if (tensor.ctx && tensor.ctx.device) {
    return warpWebGPU(
      tensor,
      shape,
      time,
      speed,
      freq,
      octaves,
      displacement,
      splineOrder,
      signedRange,
    );
  }
  let out = tensor;
  const baseFreq = Array.isArray(freq) ? freq : freqForShape(freq, shape);
  for (let octave = 1; octave <= octaves; octave++) {
    const mult = 2 ** octave;
    const f = [
      Math.floor(baseFreq[0] * 0.5 * mult),
      Math.floor(baseFreq[1] * 0.5 * mult),
    ];
    if (f[0] >= shape[0] || f[1] >= shape[1]) break;
    const opts = { ctx: tensor.ctx, time, speed, splineOrder };
    const flowX = await values(f, [shape[0], shape[1], 1], opts);
    const flowY = await values(f, [shape[0], shape[1], 1], opts);
    out = await refractOp(
      out,
      flowX,
      flowY,
      displacement / mult,
      InterpolationType.linear,
      signedRange,
    );
  }
  return out;
}
register("warp", warp, {
  freq: 2,
  octaves: 5,
  displacement: 1,
  splineOrder: InterpolationType.bicubic,
  signedRange: true,
});

export async function shadow(
  tensor,
  shape,
  time,
  speed,
  alpha = 1,
  reference = null,
) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  let ref = reference || tensor;
  ref = await toValueMap(ref);
  const valueShape = [h, w, 1];
  if (ref.shape[0] !== h || ref.shape[1] !== w) {
    ref = await resample(ref, valueShape);
  }
  ref = await normalize(ref);
  const [sobelXTensor, sobelYTensor] = await Promise.all([
    convolve(
      ref,
      valueShape,
      time,
      speed,
      ValueMask.conv2d_sobel_x,
      true,
      1,
    ),
    convolve(
      ref,
      valueShape,
      time,
      speed,
      ValueMask.conv2d_sobel_y,
      true,
      1,
    ),
  ]);
  let shade = await withTensorDatas(
    [sobelXTensor, sobelYTensor],
    (sobelXData, sobelYData) => {
      const distData = new Float32Array(h * w);
      for (let i = 0; i < h * w; i++) {
        distData[i] = distance(
          sobelXData[i],
          sobelYData[i],
          DistanceMetric.euclidean,
        );
      }
      return Tensor.fromArray(ctx, distData, valueShape);
    },
  );
  shade = await normalize(shade);
  shade = await convolve(
    shade,
    valueShape,
    time,
    speed,
    ValueMask.conv2d_sharpen,
    true,
    0.5,
  );
  const shadeDataMaybe = shade.read();
  const shadeData =
    shadeDataMaybe && typeof shadeDataMaybe.then === "function"
      ? await shadeDataMaybe
      : shadeDataMaybe;
  const highlight = new Float32Array(h * w);
  for (let i = 0; i < h * w; i++) {
    highlight[i] = Math.fround(shadeData[i] * shadeData[i]);
  }
  const srcMaybe = tensor.read();
  const srcData =
    srcMaybe && typeof srcMaybe.then === "function" ? await srcMaybe : srcMaybe;
  const shadedData = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    const sh = shadeData[i];
    const hi = highlight[i];
    for (let k = 0; k < c; k++) {
      const idx = i * c + k;
      const val = srcData[idx];
      const dark = Math.fround((1 - val) * (1 - hi));
      const lit = Math.fround(1 - dark);
      const shaded = Math.fround(lit * sh);
      shadedData[idx] = Math.min(1, Math.max(0, shaded));
    }
  }
  const shadedTensor = Tensor.fromArray(ctx, shadedData, shape);
  if (c === 1) {
    return blend(tensor, shadedTensor, alpha);
  }
  if (c === 2) {
    const out = srcData.slice();
    let alphaVals = null;
    let alphaChannels = 0;
    if (alpha && typeof alpha === "object") {
      let alphaTensor = alpha;
      if (alphaTensor && typeof alphaTensor.then === "function") {
        alphaTensor = await alphaTensor;
      }
      if (alphaTensor && typeof alphaTensor.read === "function") {
        const alphaMaybe = alphaTensor.read();
        alphaVals =
          alphaMaybe && typeof alphaMaybe.then === "function"
            ? await alphaMaybe
            : alphaMaybe;
        alphaChannels = alphaTensor.shape[2] || 1;
      }
    }
    for (let i = 0; i < h * w; i++) {
      const tVal =
        alphaVals && alphaChannels
          ? alphaVals[i * alphaChannels]
          : alpha;
      const shadeVal = shadedData[i * c];
      out[i * 2] = Math.fround(
        (1 - tVal) * srcData[i * 2] + tVal * shadeVal,
      );
    }
    return Tensor.fromArray(ctx, out, shape);
  }
  let rgbTensor;
  let alphaChannel = null;
  if (c === 4) {
    const rgbData = new Float32Array(h * w * 3);
    alphaChannel = new Float32Array(h * w);
    for (let i = 0; i < h * w; i++) {
      const base = i * 4;
      const base3 = i * 3;
      rgbData[base3] = srcData[base];
      rgbData[base3 + 1] = srcData[base + 1];
      rgbData[base3 + 2] = srcData[base + 2];
      alphaChannel[i] = srcData[base + 3];
    }
    rgbTensor = Tensor.fromArray(ctx, rgbData, [h, w, 3]);
  } else {
    rgbTensor = tensor;
  }
  const shadeRgb = new Float32Array(h * w * 3);
  for (let i = 0; i < h * w; i++) {
    const base = i * c;
    const base3 = i * 3;
    shadeRgb[base3] = shadedData[base];
    shadeRgb[base3 + 1] = shadedData[base + Math.min(1, c - 1)];
    shadeRgb[base3 + 2] = shadedData[base + Math.min(2, c - 1)];
  }
  let hsvTensor = rgbToHsv(rgbTensor);
  if (hsvTensor && typeof hsvTensor.then === "function") {
    hsvTensor = await hsvTensor;
  }
  let shadeHsvTensor = rgbToHsv(
    Tensor.fromArray(ctx, shadeRgb, [h, w, 3]),
  );
  if (shadeHsvTensor && typeof shadeHsvTensor.then === "function") {
    shadeHsvTensor = await shadeHsvTensor;
  }
  const hsvDataMaybe = hsvTensor.read();
  const hsvData =
    hsvDataMaybe && typeof hsvDataMaybe.then === "function"
      ? await hsvDataMaybe
      : hsvDataMaybe;
  const shadeHsvDataMaybe = shadeHsvTensor.read();
  const shadeHsvData =
    shadeHsvDataMaybe && typeof shadeHsvDataMaybe.then === "function"
      ? await shadeHsvDataMaybe
      : shadeHsvDataMaybe;
  let alphaVals = null;
  let alphaChannels = 0;
  if (alpha && typeof alpha === "object") {
    let alphaTensor = alpha;
    if (alphaTensor && typeof alphaTensor.then === "function") {
      alphaTensor = await alphaTensor;
    }
    if (alphaTensor && typeof alphaTensor.read === "function") {
      const alphaMaybe = alphaTensor.read();
      alphaVals =
        alphaMaybe && typeof alphaMaybe.then === "function"
          ? await alphaMaybe
          : alphaMaybe;
      alphaChannels = alphaTensor.shape[2] || 1;
    }
  }
  for (let i = 0; i < h * w; i++) {
    const tVal =
      alphaVals && alphaChannels ? alphaVals[i * alphaChannels] : alpha;
    hsvData[i * 3 + 2] = Math.fround(
      (1 - tVal) * hsvData[i * 3 + 2] + tVal * shadeHsvData[i * 3 + 2],
    );
  }
  let result = hsvToRgb(Tensor.fromArray(ctx, hsvData, [h, w, 3]));
  if (result && typeof result.then === "function") {
    result = await result;
  }
  if (c === 4) {
    const resDataMaybe = result.read();
    const resData =
      resDataMaybe && typeof resDataMaybe.then === "function"
        ? await resDataMaybe
        : resDataMaybe;
    const out = new Float32Array(h * w * 4);
    for (let i = 0; i < h * w; i++) {
      const base = i * 4;
      const base3 = i * 3;
      out[base] = resData[base3];
      out[base + 1] = resData[base3 + 1];
      out[base + 2] = resData[base3 + 2];
      out[base + 3] = alphaChannel[i];
    }
    return Tensor.fromArray(ctx, out, shape);
  }
  return result;
}
register("shadow", shadow, { alpha: 1, reference: null });

export async function bloom(tensor, shape, time, speed, alpha = 0.5) {
  const [h, w, c] = shape;
  const src = await tensor.read();
  const bright = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    let v = src[i] * 2 - 1;
    if (v < 0) v = 0;
    if (v > 1) v = 1;
    bright[i] = v;
  }

  let blurred = Tensor.fromArray(tensor.ctx, bright, shape);
  const targetShape = [
    Math.max(1, Math.floor(h / 100)),
    Math.max(1, Math.floor(w / 100)),
    c,
  ];
  blurred = await proportionalDownsample(blurred, shape, targetShape);
  const blurredData = await blurred.read();
  for (let i = 0; i < blurredData.length; i++) {
    blurredData[i] *= 4;
  }
  blurred = Tensor.fromArray(tensor.ctx, blurredData, targetShape);
  blurred = await resample(blurred, shape);
  const xOffset = Math.trunc(w * -0.05);
  const yOffset = Math.trunc(h * -0.05);
  blurred = await offsetTensor(blurred, xOffset, yOffset);
  blurred = await adjustBrightness(blurred, shape, time, speed, 0.25);
  blurred = await adjustContrast(blurred, shape, time, speed, 1.5);

  const blurredFull = await blurred.read();
  const mixData = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    mixData[i] = (src[i] + blurredFull[i]) * 0.5;
  }
  const mixedTensor = Tensor.fromArray(tensor.ctx, mixData, shape);
  const [baseClamped, mixedClamped] = await Promise.all([
    clamp01(tensor),
    clamp01(mixedTensor),
  ]);
  return blend(baseClamped, mixedClamped, alpha);
}
register("bloom", bloom, { alpha: 0.5 });

export function derivative(
  tensor,
  shape,
  time,
  speed,
  distMetric = DistanceMetric.euclidean,
  withNormalize = true,
  alpha = 1,
) {
  const [h, w, c] = shape;
  const kx = [
    [0, 0, 0],
    [0, 1, -1],
    [0, 0, 0],
  ];
  const ky = [
    [0, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
  ];
  const dx = convolution(tensor, kx, { normalize: false });
  const dy = convolution(tensor, ky, { normalize: false });
  const compute = (dxData, dyData) => {
    const out = new Float32Array(h * w * c);
    for (let i = 0; i < out.length; i++) {
      out[i] = distance(dxData[i], dyData[i], distMetric);
    }
    let result = Tensor.fromArray(tensor.ctx, out, shape);
    if (withNormalize) result = normalize(result);
    if (alpha !== 1) result = blend(tensor, result, alpha);
    return result;
  };
  const handle = (dxT, dyT) => {
    const ctx = tensor.ctx;
    if (
      ctx &&
      ctx.device &&
      typeof GPUTexture !== "undefined" &&
      dxT.handle instanceof GPUTexture &&
      dyT.handle instanceof GPUTexture
    ) {
      return (async () => {
        try {
          const device = ctx.device;
          const outSize = h * w * c;
          const outBuf = device.createBuffer({
            size: outSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
          });
          const paramsArr = new Float32Array([w, h, c, distMetric, 0, 0, 0, 0]);
          const paramsBuf = ctx.createGPUBuffer(
            paramsArr,
            GPUBufferUsage.UNIFORM,
          );
          await ctx.runCompute(
            DERIVATIVE_WGSL,
            [
              { binding: 0, resource: dxT.handle.createView() },
              { binding: 1, resource: dyT.handle.createView() },
              { binding: 2, resource: { buffer: outBuf } },
              { binding: 3, resource: { buffer: paramsBuf } },
            ],
            Math.ceil(w / 8),
            Math.ceil(h / 8),
          );
          const out = await ctx.readGPUBuffer(outBuf, outSize * 4);
          let result = Tensor.fromArray(ctx, out, shape);
          if (withNormalize) result = await normalize(result);
          if (alpha !== 1) result = await blend(tensor, result, alpha);
          return result;
        } catch (e) {
          console.warn("WebGPU derivative fallback to CPU", e);
          return withTensorDatas([dxT, dyT], compute);
        }
      })();
    }
    return withTensorDatas([dxT, dyT], compute);
  };
  if (
    (dx && typeof dx.then === "function") ||
    (dy && typeof dy.then === "function")
  ) {
    return Promise.all([dx, dy]).then(([dxx, dyy]) => handle(dxx, dyy));
  }
  return handle(dx, dy);
}
register("derivative", derivative, {
  distMetric: DistanceMetric.euclidean,
  withNormalize: true,
  alpha: 1,
});

export function sobelOperator(
  tensor,
  shape,
  time,
  speed,
  distMetric = DistanceMetric.euclidean,
) {
  const [h, w, c] = shape;
  const blurKernel = [
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1],
  ].map((row) => row.map((v) => Math.fround(v / 36)));
  const sobelXKernel = [
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1],
  ].map((row) => row.map((v) => Math.fround(v / 2)));
  const sobelYKernel = [
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1],
  ].map((row) => row.map((v) => Math.fround(v / 2)));
  const convolveAndProcess = (blurred) => {
    const gx = convolution(blurred, sobelXKernel, { normalize: false });
    const gy = convolution(blurred, sobelYKernel, { normalize: false });

    const gradient = withTensorDatas([gx, gy], (gxData, gyData) => {
      const grad = new Float32Array(gxData.length);
      for (let i = 0; i < grad.length; i++) {
        grad[i] = Math.fround(distance(gxData[i], gyData[i], distMetric));
      }
      return Tensor.fromArray(tensor.ctx, grad, shape);
    });

    const handleGradient = (gradTensor) => {
      const normalized = normalize(gradTensor);
      const process = (normTensor) =>
        withTensorData(normTensor, (data) => {
          const out = new Float32Array(data.length);
          for (let i = 0; i < data.length; i++) {
            const doubled = Math.fround(data[i] * 2);
            const centered = Math.fround(doubled - 1);
            out[i] = Math.fround(Math.abs(centered));
          }
          const ctx = (normTensor && normTensor.ctx) || tensor.ctx;
          return Tensor.fromArray(ctx, out, shape);
        });

      if (normalized && typeof normalized.then === "function") {
        return normalized.then(process);
      }
      return process(normalized);
    };

    const processed =
      gradient && typeof gradient.then === "function"
        ? gradient.then(handleGradient)
        : handleGradient(gradient);

    const shift = (result) => offsetTensor(result, -1, -1);

    if (processed && typeof processed.then === "function") {
      return processed.then(shift);
    }
    return shift(processed);
  };
  const blurred = convolution(tensor, blurKernel);
  if (blurred && typeof blurred.then === "function") {
    return blurred.then(convolveAndProcess);
  }
  return convolveAndProcess(blurred);
}
register("sobel_operator", sobelOperator, {
  distMetric: DistanceMetric.euclidean,
});

export function sobel(
  tensor,
  shape,
  time,
  speed,
  distMetric = DistanceMetric.euclidean,
  rgb = false,
  alpha = 1,
) {
  let out;
  if (rgb) {
    out = sobelOperator(tensor, shape, time, speed, distMetric);
  } else {
    out = outline(tensor, shape, time, speed, distMetric, true);
  }
  if (alpha !== 1) {
    out = blend(tensor, out, alpha);
  }
  return out;
}
register("sobel", sobel, {
  distMetric: DistanceMetric.euclidean,
  rgb: false,
  alpha: 1,
});

export async function outline(
  tensor,
  shape,
  time,
  speed,
  sobelMetric = DistanceMetric.euclidean,
  invert = false,
) {
  const [h, w, c] = shape;
  const valueShape = [h, w, 1];
  let values = await toValueMap(tensor);
  if (
    values.shape[0] !== h ||
    values.shape[1] !== w ||
    values.shape[2] !== 1
  ) {
    values = await resample(values, valueShape);
    if (values.shape[2] !== 1) {
      values = await toValueMap(values);
    }
  }
  values = await normalize(values);
  const edgesTensor = await sobelOperator(
    values,
    valueShape,
    time,
    speed,
    sobelMetric,
  );
  const [edgeDataMaybe, srcDataMaybe] = await Promise.all([
    edgesTensor.read(),
    tensor.read(),
  ]);
  const edgeData =
    edgeDataMaybe instanceof Float32Array
      ? edgeDataMaybe
      : Float32Array.from(edgeDataMaybe || []);
  const srcData =
    srcDataMaybe instanceof Float32Array
      ? srcDataMaybe
      : Float32Array.from(srcDataMaybe || []);
  const out = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    const eVal = invert ? 1 - edgeData[i] : edgeData[i];
    for (let k = 0; k < c; k++) {
      out[i * c + k] = srcData[i * c + k] * eVal;
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
}
register("outline", outline, {
  sobelMetric: DistanceMetric.euclidean,
  invert: false,
});

export async function glowingEdges(
  tensor,
  shape,
  time,
  speed,
  sobelMetric = DistanceMetric.manhattan,
  alpha = 1,
) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const valueShape = [h, w, 1];
  const srcMaybe = tensor.read();
  const src =
    srcMaybe && typeof srcMaybe.then === "function"
      ? await srcMaybe
      : srcMaybe;

  let edges = await toValueMap(tensor);
  if (edges.shape[0] !== h || edges.shape[1] !== w) {
    edges = await resample(edges, valueShape);
  }
  edges = await normalize(edges);
  const levels = randomInt(3, 5);
  edges = await posterize(edges, valueShape, time, speed, levels);
  edges = await sobelOperator(edges, valueShape, time, speed, sobelMetric);
  const sobelDataMaybe = edges.read();
  const sobelData =
    sobelDataMaybe && typeof sobelDataMaybe.then === "function"
      ? await sobelDataMaybe
      : sobelDataMaybe;
  const inverted = new Float32Array(h * w);
  for (let i = 0; i < h * w; i++) {
    inverted[i] = Math.fround(1 - sobelData[i]);
  }
  let edgeTensor = Tensor.fromArray(ctx, inverted, valueShape);

  const srcData =
    src instanceof Float32Array ? src : Float32Array.from(src || []);
  const multiplied = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    const eVal = Math.min(inverted[i] * 8, 1);
    const base = i * c;
    for (let k = 0; k < c; k++) {
      const tVal = Math.min(srcData[base + k] * 1.25, 1);
      multiplied[base + k] = Math.fround(eVal * tVal);
    }
  }
  edgeTensor = Tensor.fromArray(ctx, multiplied, shape);
  edgeTensor = await bloom(edgeTensor, shape, time, speed, 0.5);

  const blurTensor = maskValues(ValueMask.conv2d_blur)[0];
  const [bh, bw] = maskShape(ValueMask.conv2d_blur);
  const blurFlatMaybe = blurTensor.read();
  const blurFlat =
    blurFlatMaybe && typeof blurFlatMaybe.then === "function"
      ? await blurFlatMaybe
      : blurFlatMaybe;
  const blurKernel = [];
  for (let y = 0; y < bh; y++) {
    const row = [];
    for (let x = 0; x < bw; x++) {
      row.push(blurFlat[y * bw + x]);
    }
    blurKernel.push(row);
  }
  const blurred = await convolution(edgeTensor, blurKernel);
  const edgeDataMaybe = edgeTensor.read();
  const blurredDataMaybe = blurred.read();
  const edgeData =
    edgeDataMaybe && typeof edgeDataMaybe.then === "function"
      ? await edgeDataMaybe
      : edgeDataMaybe;
  const blurredData =
    blurredDataMaybe && typeof blurredDataMaybe.then === "function"
      ? await blurredDataMaybe
      : blurredDataMaybe;
  const sum = new Float32Array(edgeData.length);
  for (let i = 0; i < sum.length; i++) {
    sum[i] = Math.fround(edgeData[i] + blurredData[i]);
  }
  edgeTensor = Tensor.fromArray(ctx, sum, shape);
  edgeTensor = await normalize(edgeTensor);
  const normalizedMaybe = edgeTensor.read();
  const normalized =
    normalizedMaybe && typeof normalizedMaybe.then === "function"
      ? await normalizedMaybe
      : normalizedMaybe;
  const final = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    const base = i * c;
    for (let k = 0; k < c; k++) {
      const edgeVal = normalized[base + k];
      const srcVal = srcData[base + k];
      final[base + k] = Math.fround(1 - (1 - edgeVal) * (1 - srcVal));
    }
  }
  let result = Tensor.fromArray(ctx, final, shape);
  if (alpha !== 1) {
    result = await blend(tensor, result, alpha);
  }
  return result;
}
register("glowing_edges", glowingEdges, {
  sobelMetric: DistanceMetric.manhattan,
  alpha: 1,
});

export async function normalMap(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  if (
    ctx &&
    ctx.device &&
    typeof GPUTexture !== "undefined" &&
    tensor.handle instanceof GPUTexture
  ) {
    try {
      const outSize = h * w * 3;
      const outBuf = ctx.device.createBuffer({
        size: outSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const paramsArr = new Float32Array([w, h, c, 0, 0, 0, 0, 0]);
      const paramsBuf = ctx.createGPUBuffer(
        paramsArr,
        GPUBufferUsage.UNIFORM,
      );
      await ctx.runCompute(
        NORMAL_MAP_WGSL,
        [
          { binding: 0, resource: tensor.handle.createView() },
          { binding: 1, resource: { buffer: outBuf } },
          { binding: 2, resource: { buffer: paramsBuf } },
        ],
        Math.ceil(w / 8),
        Math.ceil(h / 8),
      );
      const out = await ctx.readGPUBuffer(outBuf, outSize * 4);
      return Tensor.fromArray(ctx, out, [h, w, 3]);
    } catch (e) {
      console.warn("WebGPU normalMap fallback to CPU", e);
      const data = await tensor.read();
      return normalMap(
        Tensor.fromArray(null, data, [h, w, c]),
        shape,
        time,
        speed,
      );
    }
  }

  const valueShape = [h, w, 1];
  let reference = await toValueMap(tensor);
  reference = await normalize(reference);
  const sobelX = await convolve(
    reference,
    valueShape,
    time,
    speed,
    ValueMask.conv2d_sobel_x,
    true,
    1,
  );
  const sobelY = await convolve(
    reference,
    valueShape,
    time,
    speed,
    ValueMask.conv2d_sobel_y,
    true,
    1,
  );
  const oneMinusX = await withTensorData(sobelX, (data) => {
    const out = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
      out[i] = 1 - data[i];
    }
    return Tensor.fromArray(sobelX.ctx, out, sobelX.shape);
  });
  const [xTensor, yTensor] = await Promise.all([
    normalize(oneMinusX),
    normalize(sobelY),
  ]);
  const [xData, yData] = await Promise.all([xTensor.read(), yTensor.read()]);
  const mag = new Float32Array(h * w);
  for (let i = 0; i < h * w; i++) {
    mag[i] = Math.sqrt(xData[i] * xData[i] + yData[i] * yData[i]);
  }
  const zTensor = await normalize(
    Tensor.fromArray(tensor.ctx, mag, [h, w, 1]),
  );
  const zData = await zTensor.read();
  const out = new Float32Array(h * w * 3);
  for (let i = 0; i < h * w; i++) {
    const z = 1 - Math.abs(zData[i] * 2 - 1) * 0.5 + 0.5;
    out[i * 3] = xData[i];
    out[i * 3 + 1] = yData[i];
    out[i * 3 + 2] = z;
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, 3]);
}
register("normal_map", normalMap, {});

async function voronoiCPU(
  tensor,
  shape,
  time,
  speed,
  diagramType = VoronoiDiagramType.range,
  nth = 0,
  distMetric = DistanceMetric.euclidean,
  sdfSides = 3,
  alpha = 1,
  withRefract = 0,
  inverse = false,
  refractYFromOffset = true,
  pointFreq = 3,
  pointGenerations = 1,
  pointDistrib = PointDistribution.random,
  pointDrift = 0,
  pointCorners = false,
  xy = null,
  downsample = true,
) {
  const originalShape = shape;
  if (downsample) {
    const dh = Math.max(1, Math.floor(shape[0] * 0.5));
    const dw = Math.max(1, Math.floor(shape[1] * 0.5));
    shape = [dh, dw, shape[2]];
  }
  const [h, w, c] = shape;
  let xPts, yPts, count;
  if (!xy) {
    [xPts, yPts] = pointCloud(pointFreq, {
      distrib: pointDistrib,
      shape,
      corners: pointCorners,
      generations: pointGenerations,
      drift: pointDrift,
      time,
      speed,
    });
    count = xPts.length;
  } else {
    [xPts, yPts, count] = xy;
    if (downsample) {
      xPts = xPts.map((v) => v / 2);
      yPts = yPts.map((v) => v / 2);
    }
  }
  if (count === 0) return tensor;
  const needFlow =
    diagramType === VoronoiDiagramType.flow ||
    diagramType === VoronoiDiagramType.color_flow;
  const isTriangular =
    distMetric === DistanceMetric.triangular ||
    distMetric === DistanceMetric.hexagram ||
    distMetric === DistanceMetric.sdf;
  if (needFlow) {
    const jitterX = randomNormalArray(count, 0, 0.0001);
    const jitterY = randomNormalArray(count, 0, 0.0001);
    for (let i = 0; i < count; i++) {
      xPts[i] += jitterX[i];
      yPts[i] += jitterY[i];
    }
  }
  const distMap = new Float32Array(h * w);
  const indexMap = new Int32Array(h * w);
  const flowMap = needFlow ? new Float32Array(h * w) : null;
  const colorFlowMap =
    needFlow && tensor && diagramType === VoronoiDiagramType.color_flow
      ? new Float32Array(h * w * c)
      : null;
  let maxDist = 0;
  let minDist = Infinity;
  const regionColorsNeeded =
    tensor &&
    (diagramType === VoronoiDiagramType.color_regions ||
      diagramType === VoronoiDiagramType.range_regions ||
      diagramType === VoronoiDiagramType.color_flow);
  const src = tensor ? await tensor.read() : null;
  const srcH = tensor ? tensor.shape[0] : 0;
  const srcW = tensor ? tensor.shape[1] : 0;
  const pointColors = regionColorsNeeded
    ? new Float32Array(count * c)
    : null;
  if (regionColorsNeeded) {
    const scaleY = srcH / h;
    const scaleX = srcW / w;
    for (let i = 0; i < count; i++) {
      const px = ((Math.trunc(yPts[i] * scaleY) % srcH) + srcH) % srcH;
      const py = ((Math.trunc(xPts[i] * scaleX) % srcW) + srcW) % srcW;
      const base = (px * srcW + py) * c;
      for (let k = 0; k < c; k++) {
        pointColors[i * c + k] = Math.fround(src[base + k]);
      }
    }
  }
  const halfW = Math.floor(w / 2);
  const halfH = Math.floor(h / 2);
  let n = needFlow ? 0 : nth >= 0 ? nth : count + nth;
  if (n < 0) n = 0;
  if (n >= count) n = count - 1;
  const bestDists = new Float32Array(n + 1);
  const bestIdx = new Int32Array(n + 1);
  const colorSum = colorFlowMap ? new Float32Array(c) : null;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let j = 0; j <= n; j++) {
        bestDists[j] = Infinity;
        bestIdx[j] = 0;
      }
      let flowSum = 0;
      if (colorSum) {
        for (let k = 0; k < c; k++) colorSum[k] = 0;
      }
      const ySign = inverse ? -1 : 1;
      for (let i = 0; i < count; i++) {
        let dx, dy;
        if (isTriangular) {
          dx = Math.fround((x - xPts[i]) / w);
          dy = Math.fround(((y - yPts[i]) * ySign) / h);
        } else {
          const x0 = Math.fround(x - xPts[i] - halfW);
          const x1 = Math.fround(x - xPts[i] + halfW);
          const y0 = Math.fround(y - yPts[i] - halfH);
          const y1 = Math.fround(y - yPts[i] + halfH);
          dx = Math.fround(Math.min(Math.abs(x0), Math.abs(x1)) / w);
          dy = Math.fround(Math.min(Math.abs(y0), Math.abs(y1)) / h);
        }
        const d = Math.fround(distance(dx, dy, distMetric, sdfSides));
        if (needFlow) {
          const ld = Math.fround(
            Math.max(-10, Math.min(10, Math.log(d))),
          );
          flowSum = Math.fround(flowSum + ld);
          if (colorSum) {
            for (let k = 0; k < c; k++) {
              colorSum[k] = Math.fround(
                colorSum[k] + Math.fround(ld * pointColors[i * c + k]),
              );
            }
          }
        }
        if (
          d < bestDists[n] - 1e-6 ||
          (Math.abs(d - bestDists[n]) <= 1e-6 && i < bestIdx[n])
        ) {
          let j = n;
          while (j > 0 && d < bestDists[j - 1]) {
            bestDists[j] = bestDists[j - 1];
            bestIdx[j] = bestIdx[j - 1];
            j--;
          }
          bestDists[j] = d;
          bestIdx[j] = i;
        }
      }
      const idx = y * w + x;
      if (needFlow) {
        flowMap[idx] = Math.fround(flowSum);
        if (colorFlowMap) {
          for (let k = 0; k < c; k++) {
            colorFlowMap[idx * c + k] = Math.fround(colorSum[k]);
          }
        }
        distMap[idx] = bestDists[0];
        indexMap[idx] = bestIdx[0];
      } else {
        const selDist = bestDists[n];
        const selIdx = bestIdx[n];
        distMap[idx] = selDist;
        indexMap[idx] = selIdx;
        if (selDist > maxDist) maxDist = selDist;
        if (selDist < minDist) minDist = selDist;
      }
    }
  }
  let rangeTensor = null;
  let indexTensor = null;
  let regionsTensor = null;
  let colorRegionsTensor = null;
  let rangeData = null;
  if (!needFlow) {
    rangeData = new Float32Array(h * w);
    const delta = maxDist - minDist;
    for (let i = 0; i < h * w; i++) {
      let v = distMap[i];
      if (delta > 0) {
        v = (v - minDist) / delta;
      }
      rangeData[i] = Math.sqrt(v);
    }
    if (inverse) {
      for (let i = 0; i < h * w; i++) {
        rangeData[i] = 1 - rangeData[i];
      }
    }
    rangeTensor = Tensor.fromArray(tensor ? tensor.ctx : null, rangeData, [h, w, 1]);
    indexTensor = Tensor.fromArray(
      tensor ? tensor.ctx : null,
      new Float32Array(indexMap),
      [h, w, 1],
    );
  }
  const xOff = isTriangular ? 0 : Math.floor(w / 2);
  const yOff = isTriangular ? 0 : Math.floor(h / 2);
  // offsetTensor expects a resolved Tensor; await any pending promises
  if (rangeTensor) {
    rangeTensor = await offsetTensor(await rangeTensor, xOff, yOff);
    if (downsample) {
      rangeTensor = await resample(rangeTensor, [originalShape[0], originalShape[1], 1]);
    }
  }
  if (indexTensor) indexTensor = await offsetTensor(await indexTensor, xOff, yOff);
  if (indexTensor) {
    const idxData = await indexTensor.read();
    regionsTensor = Tensor.fromArray(
      tensor ? tensor.ctx : null,
      Float32Array.from(idxData, (v) => v / count),
      [h, w, 1],
    );
    if (regionColorsNeeded) {
      const out = new Float32Array(h * w * c);
      for (let i = 0; i < h * w; i++) {
        const region = idxData[i];
        for (let k = 0; k < c; k++) {
          out[i * c + k] = pointColors[region * c + k];
        }
      }
      colorRegionsTensor = Tensor.fromArray(tensor.ctx, out, shape);
    }
  }
  let outTensor;
  if (diagramType === VoronoiDiagramType.range) {
    outTensor = rangeTensor;
  } else if (diagramType === VoronoiDiagramType.color_range && tensor) {
    let rTensor = rangeTensor;
    if (downsample)
      rTensor = await resample(rangeTensor, [originalShape[0], originalShape[1], 1]);
    rTensor = await rTensor;
    const rData = await rTensor.read();
    const out = new Float32Array(originalShape[0] * originalShape[1] * c);
    for (let i = 0; i < rData.length; i++) {
      const r = rData[i];
      for (let k = 0; k < c; k++) {
        const base = i * c + k;
        const aVal = src[base] * r;
        out[base] = aVal * (1 - r) + r * r;
      }
    }
    outTensor = Tensor.fromArray(tensor.ctx, out, [originalShape[0], originalShape[1], c]);
  } else if (diagramType === VoronoiDiagramType.regions) {
    outTensor = regionsTensor;
  } else if (diagramType === VoronoiDiagramType.color_regions && tensor) {
    outTensor = colorRegionsTensor;
  } else if (diagramType === VoronoiDiagramType.range_regions && tensor) {
    const rangeSq = new Float32Array(h * w);
    for (let i = 0; i < h * w; i++) rangeSq[i] = rangeData[i] * rangeData[i];
    const rangeSqTensor = Tensor.fromArray(tensor.ctx, rangeSq, [h, w, 1]);
    outTensor = blend(colorRegionsTensor, rangeTensor, rangeSqTensor);
  } else if (diagramType === VoronoiDiagramType.flow) {
    const out = new Float32Array(h * w);
    for (let i = 0; i < h * w; i++) {
      let v = Math.fround(flowMap[i] / count);
      out[i] = Math.fround((v + 1.75) / 1.45);
    }
    outTensor = Tensor.fromArray(tensor ? tensor.ctx : null, out, [h, w, 1]);
  } else if (diagramType === VoronoiDiagramType.color_flow && tensor) {
    const out = new Float32Array(h * w * c);
    for (let i = 0; i < h * w; i++) {
      for (let k = 0; k < c; k++) {
        let v = Math.fround(colorFlowMap[i * c + k] / count);
        out[i * c + k] = v;
      }
    }
    outTensor = Tensor.fromArray(tensor.ctx, out, shape);
  } else {
    return tensor;
  }
  if (
    diagramType === VoronoiDiagramType.flow ||
    diagramType === VoronoiDiagramType.color_flow
  ) {
    outTensor = await offsetTensor(outTensor, xOff, yOff);
  }
  if (downsample) {
    const splineOrder =
      diagramType === VoronoiDiagramType.color_regions
        ? InterpolationType.constant
        : InterpolationType.bicubic;
    outTensor = await resample(outTensor, originalShape, splineOrder);
  }
  if (withRefract) {
    // Python's voronoi effect calls value.refract() directly with the range
    // slice as the X reference.  When ``refract_y_from_offset`` is False it
    // derives the Y reference from sin/cos transformations of that slice; when
    // True it simply offsets the slice by half the image dimensions.  Mirror
    // that behaviour here instead of routing through the higher level
    // ``refractEffect`` helper.

    let rx = outTensor;
    let ry;
    if (refractYFromOffset) {
      ry = await offsetTensor(outTensor, Math.floor(w * 0.5), Math.floor(h * 0.5));
    } else {
      const data = await outTensor.read();
      const cosData = new Float32Array(h * w);
      const sinData = new Float32Array(h * w);
      for (let i = 0; i < h * w; i++) {
        const v = data[i] * TAU;
        cosData[i] = Math.cos(v);
        sinData[i] = Math.sin(v);
      }
      rx = Tensor.fromArray(outTensor.ctx, cosData, [h, w, 1]);
      ry = Tensor.fromArray(outTensor.ctx, sinData, [h, w, 1]);
    }
    outTensor = await refractOp(
      tensor,
      rx,
      ry,
      withRefract,
      InterpolationType.bicubic,
      true,
    );
  }
  if (tensor && diagramType !== VoronoiDiagramType.color_regions) {
    return blend(tensor, outTensor, alpha);
  }
  return outTensor;
}

async function voronoiWebGPU(
  tensor,
  shape,
  time,
  speed,
  diagramType = VoronoiDiagramType.range,
  nth = 0,
  distMetric = DistanceMetric.euclidean,
  sdfSides = 3,
  alpha = 1,
  withRefract = 0,
  inverse = false,
  refractYFromOffset = true,
  pointFreq = 3,
  pointGenerations = 1,
  pointDistrib = PointDistribution.random,
  pointDrift = 0,
  pointCorners = false,
  xy = null,
  downsample = true,
) {
  const ctx = tensor ? tensor.ctx : null;
  if (!ctx || !ctx.device) {
    return await voronoiCPU(
      tensor,
      shape,
      time,
      speed,
      diagramType,
      nth,
      distMetric,
      sdfSides,
      alpha,
      withRefract,
      inverse,
      refractYFromOffset,
      pointFreq,
      pointGenerations,
      pointDistrib,
      pointDrift,
      pointCorners,
      xy,
      downsample,
    );
  }

  // Currently only a subset of the full CPU voronoi implementation is
  // accelerated on the GPU. For unsupported modes fall back to the CPU
  // version to ensure feature parity.
  const supportedType =
    diagramType === VoronoiDiagramType.range ||
    (diagramType === VoronoiDiagramType.color_range && tensor) ||
    diagramType === VoronoiDiagramType.regions ||
    (diagramType === VoronoiDiagramType.color_regions && tensor) ||
    (diagramType === VoronoiDiagramType.range_regions && tensor) ||
    diagramType === VoronoiDiagramType.flow ||
    (diagramType === VoronoiDiagramType.color_flow && tensor);
  if (
    !supportedType ||
    nth >= 64 ||
    ![
      DistanceMetric.euclidean,
      DistanceMetric.manhattan,
      DistanceMetric.chebyshev,
      DistanceMetric.octagram,
      DistanceMetric.triangular,
      DistanceMetric.hexagram,
      DistanceMetric.sdf,
    ].includes(distMetric) ||
    (distMetric === DistanceMetric.sdf && sdfSides < 3)
  ) {
    return await voronoiCPU(
      tensor,
      shape,
      time,
      speed,
      diagramType,
      nth,
      distMetric,
      sdfSides,
      alpha,
      withRefract,
      inverse,
      refractYFromOffset,
      pointFreq,
      pointGenerations,
      pointDistrib,
      pointDrift,
      pointCorners,
      xy,
      downsample,
    );
  }

  const originalShape = shape;
  if (downsample) {
    const dh = Math.max(1, Math.floor(shape[0] * 0.5));
    const dw = Math.max(1, Math.floor(shape[1] * 0.5));
    shape = [dh, dw, shape[2]];
  }
  const [h, w] = shape;

  // Generate points on the CPU; the compute shader only handles the distance
  // calculation.
  let xPts, yPts, count;
  if (!xy) {
    [xPts, yPts] = pointCloud(pointFreq, {
      distrib: pointDistrib,
      shape: [h, w, 1],
      corners: pointCorners,
      generations: pointGenerations,
      drift: pointDrift,
      time,
      speed,
    });
    count = xPts.length;
  } else {
    [xPts, yPts, count] = xy;
    if (downsample) {
      xPts = xPts.map((v) => v / 2);
      yPts = yPts.map((v) => v / 2);
    }
  }

  if (
    diagramType === VoronoiDiagramType.flow ||
    diagramType === VoronoiDiagramType.color_flow
  ) {
    const jitterX = randomNormalArray(count, 0, 0.0001);
    const jitterY = randomNormalArray(count, 0, 0.0001);
    for (let i = 0; i < count; i++) {
      xPts[i] += jitterX[i];
      yPts[i] += jitterY[i];
    }
  }

  if (count === 0) {
    return tensor;
  }
  let n = nth;
  if (n < 0) n += count;
  if (n < 0) n = 0;
  if (n >= count) n = count - 1;
  if (n >= 64) {
    return await voronoiCPU(
      tensor,
      shape,
      time,
      speed,
      diagramType,
      nth,
      distMetric,
      sdfSides,
      alpha,
      withRefract,
      inverse,
      refractYFromOffset,
      pointFreq,
      pointGenerations,
      pointDistrib,
      pointDrift,
      pointCorners,
      xy,
      downsample,
    );
  }
  const c = tensor ? tensor.shape[2] : 0;
  const needFlow =
    diagramType === VoronoiDiagramType.flow ||
    diagramType === VoronoiDiagramType.color_flow;
  const regionColorsNeeded =
    tensor &&
    (diagramType === VoronoiDiagramType.color_regions ||
      diagramType === VoronoiDiagramType.range_regions ||
      diagramType === VoronoiDiagramType.color_flow);
  let pointColors = null;
  if (regionColorsNeeded) {
    const src = await tensor.read();
    const srcH = tensor.shape[0];
    const srcW = tensor.shape[1];
    pointColors = new Float32Array(count * c);
    const scaleY = srcH / h;
    const scaleX = srcW / w;
    for (let i = 0; i < count; i++) {
      const px = ((Math.trunc(yPts[i] * scaleY) % srcH) + srcH) % srcH;
      const py = ((Math.trunc(xPts[i] * scaleX) % srcW) + srcW) % srcW;
      const base = (px * srcW + py) * c;
      for (let k = 0; k < c; k++) {
        pointColors[i * c + k] = Math.fround(src[base + k]);
      }
    }
  }

  const pointsArr = new Float32Array(count * 2);
  for (let i = 0; i < count; i++) {
    pointsArr[i * 2] = xPts[i];
    pointsArr[i * 2 + 1] = yPts[i];
  }
  const pointsBuf = ctx.createGPUBuffer(pointsArr, GPUBufferUsage.STORAGE);
  const rangeBuf = ctx.device.createBuffer({
    size: h * w * 4,
    usage:
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const indexBuf = ctx.device.createBuffer({
    size: h * w * 4,
    usage:
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const flowBuf = needFlow
    ? ctx.device.createBuffer({
        size: h * w * 4,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      })
    : ctx.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE,
      });
  const pointColorBuf = regionColorsNeeded
    ? ctx.createGPUBuffer(pointColors, GPUBufferUsage.STORAGE)
    : ctx.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE,
      });
  const colorFlowBuf =
    diagramType === VoronoiDiagramType.color_flow && tensor
      ? ctx.device.createBuffer({
          size: h * w * c * 4,
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST,
        })
      : ctx.device.createBuffer({
          size: 4,
          usage: GPUBufferUsage.STORAGE,
        });

  // ``inverse`` affects both the distance output and, for certain metrics,
  // flips the Y offset.  The compute shader interprets any non-zero value as
  // ``true`` and handles the necessary transformations.
  const inv = inverse ? 1 : 0;
  const params = new Float32Array([
    w,
    h,
    count,
    0,
    0,
    inv,
    distMetric,
    n,
    sdfSides,
    needFlow ? 1 : 0,
    diagramType === VoronoiDiagramType.color_flow && tensor ? c : 0,
    0,
  ]);
  const paramsBuf = ctx.createGPUBuffer(params, GPUBufferUsage.UNIFORM);

  const module = ctx.device.createShaderModule({ code: VORONOI_WGSL });
  const pipeline = ctx.device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  const bindGroup = ctx.device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: pointsBuf } },
      { binding: 2, resource: { buffer: rangeBuf } },
      { binding: 3, resource: { buffer: indexBuf } },
      { binding: 4, resource: { buffer: paramsBuf } },
      { binding: 5, resource: { buffer: flowBuf } },
      { binding: 6, resource: { buffer: pointColorBuf } },
      { binding: 7, resource: { buffer: colorFlowBuf } },
    ],
  });
  const encoder = ctx.device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8), 1);
  pass.end();
  ctx.queue.submit([encoder.finish()]);
  ctx._pendingDispatch = true;

  let rangeTensor = new Tensor(ctx, rangeBuf, [h, w, 1]);
  let regionsTensor = new Tensor(ctx, indexBuf, [h, w, 1]);
  let flowTensor = needFlow ? new Tensor(ctx, flowBuf, [h, w, 1]) : null;
  let colorFlowTensor =
    needFlow && diagramType === VoronoiDiagramType.color_flow && tensor
      ? new Tensor(ctx, colorFlowBuf, [h, w, c])
      : null;
  const isTriangular =
    distMetric === DistanceMetric.triangular ||
    distMetric === DistanceMetric.hexagram ||
    distMetric === DistanceMetric.sdf;
  const xOff = isTriangular ? 0 : Math.floor(w * 0.5);
  const yOff = isTriangular ? 0 : Math.floor(h * 0.5);
  rangeTensor = await offsetTensor(rangeTensor, xOff, yOff);
  regionsTensor = await offsetTensor(regionsTensor, xOff, yOff);
  if (flowTensor) flowTensor = await offsetTensor(flowTensor, xOff, yOff);
  if (colorFlowTensor) colorFlowTensor = await offsetTensor(colorFlowTensor, xOff, yOff);
  let colorRegionsTensor = null;
  if (regionColorsNeeded) {
    await ctx.flush();
    const idxData = await regionsTensor.read();
    const out = new Float32Array(h * w * c);
    for (let i = 0; i < idxData.length; i++) {
      const region = Math.min(count - 1, Math.floor(idxData[i] * count));
      for (let k = 0; k < c; k++) {
        out[i * c + k] = pointColors[region * c + k];
      }
    }
    colorRegionsTensor = Tensor.fromArray(ctx, out, [h, w, c]);
  }
  if (!needFlow) {
    await ctx.flush();
    const raw = await rangeTensor.read();
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < raw.length; i++) {
      const v = raw[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const delta = max - min;
    const norm = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) {
      let v = raw[i];
      if (delta > 0) {
        v = (v - min) / delta;
      }
      v = Math.sqrt(v);
      if (inverse) v = 1 - v;
      norm[i] = v;
    }
    rangeTensor = Tensor.fromArray(ctx, norm, [h, w, 1]);
  }
  let outTensor;
  if (diagramType === VoronoiDiagramType.color_range && tensor) {
    let rTensor = rangeTensor;
    if (downsample) {
      rTensor = await resample(
        rTensor,
        [originalShape[0], originalShape[1], 1],
        InterpolationType.bicubic,
      );
    }
    await ctx.flush();
    const rData = await rTensor.read();
    const src = await tensor.read();
    const out = new Float32Array(originalShape[0] * originalShape[1] * c);
    for (let i = 0; i < originalShape[0] * originalShape[1]; i++) {
      const r = rData[i];
      for (let k = 0; k < c; k++) {
        const base = i * c + k;
        const aVal = src[base] * r;
        out[base] = Math.fround(aVal * (1 - r) + r * r);
      }
    }
    outTensor = Tensor.fromArray(ctx, out, [originalShape[0], originalShape[1], c]);
  } else if (diagramType === VoronoiDiagramType.regions) {
    outTensor = regionsTensor;
  } else if (diagramType === VoronoiDiagramType.color_regions && tensor) {
    outTensor = colorRegionsTensor;
  } else if (diagramType === VoronoiDiagramType.range_regions && tensor) {
    await ctx.flush();
    const rData = await rangeTensor.read();
    const rangeSq = new Float32Array(rData.length);
    for (let i = 0; i < rData.length; i++) rangeSq[i] = rData[i] * rData[i];
    const rangeSqTensor = Tensor.fromArray(ctx, rangeSq, [h, w, 1]);
    outTensor = await blend(colorRegionsTensor, rangeTensor, rangeSqTensor);
  } else if (diagramType === VoronoiDiagramType.flow) {
    await ctx.flush();
    const fData = await flowTensor.read();
    const out = new Float32Array(h * w);
    for (let i = 0; i < h * w; i++) {
      let v = Math.fround(fData[i] / count);
      out[i] = Math.fround((v + 1.75) / 1.45);
    }
    outTensor = Tensor.fromArray(ctx, out, [h, w, 1]);
  } else if (diagramType === VoronoiDiagramType.color_flow && tensor) {
    await ctx.flush();
    const fData = await colorFlowTensor.read();
    const out = new Float32Array(h * w * c);
    for (let i = 0; i < h * w; i++) {
      for (let k = 0; k < c; k++) {
        const v = Math.fround(fData[i * c + k] / count);
        out[i * c + k] = v;
      }
    }
    outTensor = Tensor.fromArray(ctx, out, [h, w, c]);
  } else {
    outTensor = rangeTensor;
  }
  if (downsample) {
    const splineOrder =
      diagramType === VoronoiDiagramType.color_regions
        ? InterpolationType.constant
        : InterpolationType.bicubic;
    outTensor = await resample(
      outTensor,
      [originalShape[0], originalShape[1], outTensor.shape[2]],
      splineOrder,
    );
  }

  if (withRefract && tensor) {
    let rx = outTensor;
    let ry;
    const [oh, ow, oc] = outTensor.shape;
    if (refractYFromOffset) {
      ry = await offsetTensor(outTensor, Math.floor(ow * 0.5), Math.floor(oh * 0.5));
    } else {
      const data = await outTensor.read();
      const cosData = new Float32Array(oh * ow);
      const sinData = new Float32Array(oh * ow);
      for (let i = 0; i < oh * ow; i++) {
        const v = data[i * oc] * TAU;
        cosData[i] = Math.cos(v);
        sinData[i] = Math.sin(v);
      }
      rx = Tensor.fromArray(ctx, cosData, [oh, ow, 1]);
      ry = Tensor.fromArray(ctx, sinData, [oh, ow, 1]);
    }
    outTensor = await refractOp(
      tensor,
      rx,
      ry,
      withRefract,
      InterpolationType.bicubic,
      true,
    );
  }

  if (tensor && diagramType !== VoronoiDiagramType.color_regions) {
    outTensor = await blend(tensor, outTensor, alpha);
  }
  return outTensor;
}

async function erosionWormsWebGPU(
  tensor,
  shape,
  time,
  speed,
  density,
  iterations,
  contraction,
  quantize,
  alpha,
  inverse,
  xyBlend,
) {
  const ctx = tensor.ctx;
  const [h, w, c] = shape;
  const count = Math.floor(Math.sqrt(h * w) * density);
  const x = randomUniform(count, 0, 1);
  const y = randomUniform(count, 0, 1);
  const dirX = randomNormalArray(count, 0, 1);
  const dirY = randomNormalArray(count, 0, 1);
  const inertia = randomNormalArray(count, 0.75, 0.25);
  const wormsArr = new Float32Array(count * 5);
  const widthScale = w - 1;
  const heightScale = h - 1;
  for (let i = 0; i < count; i++) {
    x[i] = Math.fround(x[i] * widthScale);
    y[i] = Math.fround(y[i] * heightScale);
    const len = Math.hypot(dirX[i], dirY[i]) || 1;
    const dx = Math.fround(dirX[i] / len);
    const dy = Math.fround(dirY[i] / len);
    const base = i * 5;
    wormsArr[base] = x[i];
    wormsArr[base + 1] = y[i];
    wormsArr[base + 2] = dx;
    wormsArr[base + 3] = dy;
    wormsArr[base + 4] = Math.fround(inertia[i]);
  }
  const src = await tensor.read();
  const startColors = new Float32Array(count * c);
  for (let i = 0; i < count; i++) {
    const xi = Math.floor(x[i]);
    const yi = Math.floor(y[i]);
    const base = (yi * w + xi) * c;
    for (let k = 0; k < c; k++) {
      startColors[i * c + k] = src[base + k];
    }
  }
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
  const wormsBuf = ctx.createGPUBuffer(
    wormsArr,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  );
  const valuesBuf = ctx.createGPUBuffer(valuesArr, GPUBufferUsage.STORAGE);
  const startBuf = ctx.createGPUBuffer(startColors, GPUBufferUsage.STORAGE);
  const outBuf = ctx.createGPUBuffer(
    new Float32Array(h * w * c),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  );
  const paramData = new Float32Array(8);
  const paramU32 = new Uint32Array(paramData.buffer);
  paramData[0] = w;
  paramData[1] = h;
  paramU32[2] = count;
  paramU32[3] = iterations;
  paramData[4] = contraction;
  paramData[5] = quantize ? 1 : 0;
  paramU32[6] = c;
  paramData[7] = 0;
  const paramsBuf = ctx.createGPUBuffer(paramData, GPUBufferUsage.UNIFORM);
  const module = ctx.device.createShaderModule({ code: EROSION_WORMS_WGSL });
  const pipeline = ctx.device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  const bindGroup = ctx.device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: wormsBuf } },
      { binding: 1, resource: { buffer: valuesBuf } },
      { binding: 2, resource: { buffer: startBuf } },
      { binding: 3, resource: { buffer: outBuf } },
      { binding: 4, resource: { buffer: paramsBuf } },
    ],
  });
  const encoder = ctx.device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  ctx.queue.submit([encoder.finish()]);
  ctx._pendingDispatch = true;
  let outTensor = new Tensor(ctx, outBuf, [h, w, c]);
  await ctx.flush();
  outTensor = await clamp01(outTensor);
  if (inverse) {
    await ctx.flush();
    const d = await outTensor.read();
    for (let i = 0; i < d.length; i++) d[i] = 1 - d[i];
    outTensor = Tensor.fromArray(ctx, d, [h, w, c]);
  }
  if (xyBlend) {
    const valMask = new Float32Array(h * w);
    for (let i = 0; i < h * w; i++) valMask[i] = valuesArr[i] * xyBlend;
    const mask = Tensor.fromArray(ctx, valMask, [h, w, 1]);
    tensor = blend(
      shadow(tensor, shape, time, speed),
      reindex(tensor, shape, time, speed, 1),
      mask,
    );
  }
  return blend(tensor, outTensor, alpha);
}

export async function voronoi(
  tensor,
  shape,
  time,
  speed,
  diagramType = VoronoiDiagramType.range,
  nth = 0,
  distMetric = DistanceMetric.euclidean,
  sdfSides = 3,
  alpha = 1,
  withRefract = 0,
  inverse = false,
  refractYFromOffset = true,
  pointFreq = 3,
  pointGenerations = 1,
  pointDistrib = PointDistribution.random,
  pointDrift = 0,
  pointCorners = false,
  xy = null,
  downsample = true,
) {
  if (tensor && tensor.ctx && tensor.ctx.device) {
    return await voronoiWebGPU(
      tensor,
      shape,
      time,
      speed,
      diagramType,
      nth,
      distMetric,
      sdfSides,
      alpha,
      withRefract,
      inverse,
      refractYFromOffset,
      pointFreq,
      pointGenerations,
      pointDistrib,
      pointDrift,
      pointCorners,
      xy,
      downsample,
    );
  }
  return await voronoiCPU(
    tensor,
    shape,
    time,
    speed,
    diagramType,
    nth,
    distMetric,
    sdfSides,
    alpha,
    withRefract,
    inverse,
    refractYFromOffset,
    pointFreq,
    pointGenerations,
    pointDistrib,
    pointDrift,
    pointCorners,
    xy,
    downsample,
  );
}
register("voronoi", voronoi, {
  diagramType: VoronoiDiagramType.range,
  nth: 0,
  distMetric: DistanceMetric.euclidean,
  sdfSides: 3,
  alpha: 1,
  withRefract: 0,
  inverse: false,
  refractYFromOffset: true,
  pointFreq: 3,
  pointGenerations: 1,
  pointDistrib: PointDistribution.random,
  pointDrift: 0,
  pointCorners: false,
  xy: null,
  downsample: true,
});

export async function singularity(
  tensor,
  shape,
  time,
  speed,
  diagramType = VoronoiDiagramType.range,
  distMetric = DistanceMetric.euclidean,
) {
  const [x, y] = pointCloud(1, {
    distrib: PointDistribution.square,
    shape,
  });
  return await voronoi(
    tensor,
    shape,
    time,
    speed,
    diagramType,
    0,
    distMetric,
    1,
    1,
    0,
    false,
    true,
    1,
    1,
    PointDistribution.square,
    0,
    false,
    [x, y, 1],
    true,
  );
}
register("singularity", singularity, {
  diagramType: VoronoiDiagramType.range,
  distMetric: DistanceMetric.euclidean,
});

export async function lowpoly(
  tensor,
  shape,
  time,
  speed,
  distrib = PointDistribution.random,
  freq = 10,
  distMetric = DistanceMetric.euclidean,
) {
  const [xPts, yPts] = pointCloud(freq, {
    distrib,
    shape,
    drift: 1.0,
    time,
    speed,
  });
  const count = xPts.length;
  if (count === 0) return tensor;
  const xy = [xPts, yPts, count];
  const base = tensor;
  // Match the Python implementation by passing the existing tensor into the
  // distance-field computation. Supplying `null` forces the CPU code path and
  // causes the demo to stall on large canvases. Providing the tensor enables
  // GPU acceleration when available and prevents the render-time hang observed
  // in the "basic-low-poly" preset.
  const distance = await voronoi(
    base,
    shape,
    time,
    speed,
    VoronoiDiagramType.range,
    1,
    distMetric,
    3,
    1,
    0,
    false,
    true,
    1,
    1,
    PointDistribution.square,
    0,
    false,
    xy,
    true,
  );
  const color = await voronoi(
    base,
    shape,
    time,
    speed,
    VoronoiDiagramType.color_regions,
    0,
    distMetric,
    3,
    1,
    0,
    false,
    true,
    1,
    1,
    PointDistribution.square,
    0,
    false,
    xy,
    true,
  );
  const blended = await blend(distance, color, 0.5);
  return await normalize(blended);
}
register("lowpoly", lowpoly, {
  distrib: PointDistribution.random,
  freq: 10,
  distMetric: DistanceMetric.euclidean,
});

export async function kaleido(
  tensor,
  shape,
  time,
  speed,
  sides = 6,
  sdfSides = 5,
  xy = null,
  blendEdges = true,
  pointFreq = 1,
  pointGenerations = 1,
  pointDistrib = PointDistribution.random,
  pointDrift = 0,
  pointCorners = false,
) {
  const [h, w, c] = shape;
  const ctx = tensor?.ctx ?? null;
  const valueShape = [h, w, 1];
  const xyArg = xy ? [xy[0], xy[1], xy[0].length] : null;
  const distMetric =
    sdfSides < 3 ? DistanceMetric.euclidean : DistanceMetric.sdf;
  const rTensor = await voronoi(
    null,
    valueShape,
    time,
    speed,
    VoronoiDiagramType.range,
    0,
    distMetric,
    sdfSides,
    1,
    0,
    false,
    true,
    pointFreq,
    pointGenerations,
    pointDistrib,
    pointDrift,
    pointCorners,
    xyArg,
  );
  const r = await rTensor.read();
  let fader = null;
  if (blendEdges) {
    const sTensor = await singularity(
      null,
      valueShape,
      time,
      speed,
      VoronoiDiagramType.range,
      DistanceMetric.chebyshev,
    );
    const fTensor = await normalize(sTensor);
    fader = await fTensor.read();
    for (let i = 0; i < fader.length; i++) fader[i] = Math.pow(fader[i], 5);
  }
  if (
    ctx &&
    ctx.device &&
    typeof GPUTexture !== "undefined" &&
    tensor.handle instanceof GPUTexture
  ) {
    const outBuf = ctx.createGPUBuffer(
      new Float32Array(h * w * c),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    const rBuf = ctx.createGPUBuffer(r, GPUBufferUsage.STORAGE);
    const fadeBuf = ctx.createGPUBuffer(
      fader || new Float32Array(4),
      GPUBufferUsage.STORAGE,
    );
    const paramsBuf = ctx.createGPUBuffer(
      new Float32Array([w, h, c, sides, blendEdges ? 1 : 0, 0, 0, 0]),
      GPUBufferUsage.UNIFORM,
    );
    await ctx.runCompute(
      KALEIDO_WGSL,
      [
        { binding: 0, resource: tensor.handle.createView() },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: rBuf } },
        { binding: 3, resource: { buffer: fadeBuf } },
        { binding: 4, resource: { buffer: paramsBuf } },
      ],
      Math.ceil(w / 8),
      Math.ceil(h / 8),
    );
    return new Tensor(ctx, outBuf, [h, w, c]);
  }
  const src = await tensor.read();
  const out = new Float32Array(h * w * c);
  const step = (Math.PI * 2) / sides;
  const denomX = w > 1 ? w - 1 : 1;
  const denomY = h > 1 ? h - 1 : 1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const xi = x / denomX - 0.5;
      const yi = y / denomY - 0.5;
      const radius = r[idx];
      let a = Math.atan2(yi, xi) + Math.PI / 2;
      a = ((a % step) + step) % step;
      a = Math.abs(a - step / 2);
      let nx = radius * w * Math.sin(a);
      let ny = radius * h * Math.cos(a);
      if (blendEdges) {
        const fade = fader[idx];
        nx = nx * (1 - fade) + x * fade;
        ny = ny * (1 - fade) + y * fade;
      }
      nx = ((Math.trunc(nx) % w) + w) % w;
      ny = ((Math.trunc(ny) % h) + h) % h;
      const srcBase = (ny * w + nx) * c;
      const dstBase = idx * c;
      for (let k = 0; k < c; k++) out[dstBase + k] = src[srcBase + k];
    }
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register("kaleido", kaleido, {
  sides: 6,
  sdfSides: 5,
  xy: null,
  blendEdges: true,
  pointFreq: 1,
  pointGenerations: 1,
  pointDistrib: PointDistribution.random,
  pointDrift: 0,
  pointCorners: false,
});

export async function texture(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const valueShape = [h, w, 1];

  const noiseMaybe = await simpleMultiresTensor(
    64,
    valueShape,
    time,
    speed,
    8,
    ctx,
    true,
  );
  const noise =
    noiseMaybe && typeof noiseMaybe.then === "function"
      ? await noiseMaybe
      : noiseMaybe;

  const shadeTensor = await shadow(noise, valueShape, time, speed, 1);
  const shadeMaybe = shadeTensor.read();
  const shade =
    shadeMaybe && typeof shadeMaybe.then === "function"
      ? await shadeMaybe
      : shadeMaybe;

  const srcMaybe = tensor.read();
  const src =
    srcMaybe && typeof srcMaybe.then === "function" ? await srcMaybe : srcMaybe;
  const out = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    const m = 0.9 + shade[i] * 0.1;
    for (let k = 0; k < c; k++) out[i * c + k] = src[i * c + k] * m;
  }
  return Tensor.fromArray(ctx, out, shape);
}
register("texture", texture, {});

export async function densityMap(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const bins = Math.max(h, w);
  const normalizedTensor = await normalize(tensor);
  const vals = await normalizedTensor.read();
  const total = vals.length;
  const binIndices = new Int32Array(total);
  const counts = new Int32Array(bins);
  for (let i = 0; i < total; i++) {
    let bin = Math.floor(vals[i] * (bins - 1));
    if (bin < 0) bin = 0;
    if (bin >= bins) bin = bins - 1;
    binIndices[i] = bin;
    counts[bin]++;
  }
  const gathered = new Float32Array(total);
  for (let i = 0; i < total; i++) {
    gathered[i] = counts[binIndices[i]];
  }
  const gatheredTensor = Tensor.fromArray(tensor.ctx, gathered, shape);
  return normalize(gatheredTensor);
}
register("density_map", densityMap, {});

export async function jpegDecimate(
  tensor,
  shape,
  time,
  speed,
  iterations = 25,
) {
  const [h, w, c] = shape;
  if (iterations <= 0 || Math.min(h, w) < 2) {
    return tensor;
  }
  let current = tensor;
  for (let i = 0; i < iterations; i++) {
    const dataMaybe = current.read();
    const data =
      dataMaybe && typeof dataMaybe.then === "function"
        ? await dataMaybe
        : dataMaybe;
    const length = data.length;
    const quality = randomInt(5, 50);
    const step = Math.max(1, Math.round((60 - quality) / 5));
    const quantized = new Float32Array(length);
    for (let idx = 0; idx < length; idx++) {
      let value = data[idx];
      if (!Number.isFinite(value)) value = 0;
      if (value < 0) value = 0;
      else if (value > 1) value = 1;
      let byte = Math.round(value * 255);
      byte = Math.round(byte / step) * step;
      if (byte < 0) byte = 0;
      else if (byte > 255) byte = 255;
      quantized[idx] = byte / 255;
    }
    let degraded = Tensor.fromArray(tensor.ctx, quantized, shape);
    const maxFactor = Math.max(2, Math.min(h, w));
    const factor = Math.min(randomInt(2, 8), maxFactor);
    if (factor > 1) {
      const down = await downsample(degraded, factor);
      degraded = await upsample(down, factor, InterpolationType.linear);
    }
    current = degraded;
  }
  return current;
}
register("jpeg_decimate", jpegDecimate, { iterations: 25 });

const kernelCache = new Map();

async function getKernel(mask) {
  if (kernelCache.has(mask)) return kernelCache.get(mask);
  const [tensor] = maskValues(mask);
  const size = tensor.shape[0];
  const buildKernel = (data) => {
    const kernel = [];
    for (let y = 0; y < size; y++) {
      kernel.push(Array.from(data.slice(y * size, y * size + size)));
    }
    return kernel;
  };
  const dataMaybe = tensor.read();
  let result;
  if (dataMaybe && typeof dataMaybe.then === "function") {
    result = dataMaybe.then(buildKernel);
  } else {
    result = buildKernel(dataMaybe);
  }
  kernelCache.set(mask, result);
  return result;
}

export async function convFeedback(
  tensor,
  shape,
  time,
  speed,
  iterations = 50,
  alpha = 0.5,
) {
  const halfShape = [
    Math.max(1, Math.floor(shape[0] * 0.5)),
    Math.max(1, Math.floor(shape[1] * 0.5)),
    shape[2],
  ];
  let convolved = await proportionalDownsample(tensor, shape, halfShape);
  const blurKernel = await getKernel(ValueMask.conv2d_blur);
  const sharpenKernel = await getKernel(ValueMask.conv2d_sharpen);
  const iterCount = 100;
  for (let i = 0; i < iterCount; i++) {
    convolved = await convolution(convolved, blurKernel);
    convolved = await convolution(convolved, sharpenKernel);
  }
  convolved = await normalize(convolved);
  const data = await convolved.read();
  const combined = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const up = Math.max((data[i] - 0.5) * 2, 0);
    const down = Math.min(data[i] * 2, 1);
    combined[i] = up + (1 - down);
  }
  const combinedTensor = Tensor.fromArray(
    convolved.ctx,
    combined,
    convolved.shape,
  );
  const resampled = await resample(combinedTensor, shape);
  return blend(tensor, resampled, alpha);
}
register("conv_feedback", convFeedback, { iterations: 50, alpha: 0.5 });

export function blendLayers(control, shape, feather = 1, ...layers) {
  let layerCount = layers.length;
  const controlNorm = normalize(control);
  const ctrl = controlNorm.read();
  const [h, w, c] = shape;
  const scaled = new Float32Array(h * w);
  for (let i = 0; i < h * w; i++) scaled[i] = ctrl[i] * layerCount;
  const floors = new Int32Array(h * w);
  for (let i = 0; i < h * w; i++) floors[i] = Math.floor(scaled[i]);
  const layerData = layers.map((l) => l.read());
  layerData.push(layerData[layerData.length - 1]);
  layerCount += 1;
  const out = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    const f0 = floors[i] % layerCount;
    const f1 = (floors[i] + 1) % layerCount;
    const base = i * c;
    let t = scaled[i] - floors[i];
    t = Math.min(Math.max(t - (1 - feather), 0) / feather, 1);
    const l0 = layerData[f0];
    const l1 = layerData[f1];
    for (let k = 0; k < c; k++) {
      out[base + k] = l0[base + k] * (1 - t) + l1[base + k] * t;
    }
  }
  return Tensor.fromArray(control.ctx, out, shape);
}

export function centerMask(
  center,
  edges,
  shape,
  distMetric = DistanceMetric.chebyshev,
  power = 2,
) {
  const [h, w, c] = shape;
  const cx = (w - 1) / 2;
  const cy = (h - 1) / 2;
  const dists = new Float32Array(h * w);
  let max = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = Math.abs(x - cx);
      const dy = Math.abs(y - cy);
      const d = distance(dx, dy, distMetric);
      dists[y * w + x] = d;
      if (d > max) max = d;
    }
  }
  const mask = new Float32Array(h * w * c);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const d = dists[y * w + x] / (max || 1);
      const m = Math.pow(d, power);
      for (let k = 0; k < c; k++) mask[(y * w + x) * c + k] = m;
    }
  }
  const maskTensor = Tensor.fromArray(center.ctx, mask, shape);
  return blend(center, edges, maskTensor);
}

export function innerTile(tensor, shape, freq) {
  const baseFreq = Array.isArray(freq) ? freq : freqForShape(freq, shape);
  const freqY = Math.max(1, Math.trunc(baseFreq[0] ?? 1));
  const freqX = Math.max(1, Math.trunc(baseFreq[1] ?? 1));
  const [h, w, c] = shape;
  const innerH = Math.max(1, Math.trunc(h / freqY));
  const innerW = Math.max(1, Math.trunc(w / freqX));
  const tileH = Math.max(1, innerH * freqY);
  const tileW = Math.max(1, innerW * freqY);
  return withTensorData(tensor, (src) => {
    const tiled = new Float32Array(tileH * tileW * c);
    for (let y = 0; y < tileH; y++) {
      const srcY = (y % innerH) * freqY;
      const srcRow = srcY * w * c;
      const rowBase = y * tileW * c;
      for (let x = 0; x < tileW; x++) {
        const srcX = (x % innerW) * freqX;
        const srcIndex = srcRow + srcX * c;
        const dstIndex = rowBase + x * c;
        for (let k = 0; k < c; k++) {
          tiled[dstIndex + k] = src[srcIndex + k] ?? 0;
        }
      }
    }
    const tiledTensor = Tensor.fromArray(tensor.ctx, tiled, [tileH, tileW, c]);
    return resample(tiledTensor, shape, InterpolationType.linear);
  });
}

export function expandTile(
  tensor,
  inputShape,
  outputShape,
  withOffset = true,
) {
  const [inH, inW, c] = inputShape;
  const [outH, outW] = outputShape;
  const xOff = withOffset ? Math.floor(inW / 2) : 0;
  const yOff = withOffset ? Math.floor(inH / 2) : 0;
  return withTensorData(tensor, (src) => {
    const out = new Float32Array(outH * outW * c);
    const wrap = (n, mod) => {
      const r = n % mod;
      return r < 0 ? r + mod : r;
    };
    for (let y = 0; y < outH; y++) {
      const sy = wrap(y + yOff, inH);
      for (let x = 0; x < outW; x++) {
        const sx = wrap(x + xOff, inW);
        for (let k = 0; k < c; k++) {
          out[(y * outW + x) * c + k] = src[(sy * inW + sx) * c + k];
        }
      }
    }
    return Tensor.fromArray(tensor.ctx, out, [outH, outW, c]);
  });
}

export function offsetIndex(yIndex, height, xIndex, width) {
  return withTensorDatas([yIndex, xIndex], (yData, xData) => {
    const total = height * width;
    const out = new Float32Array(total * 2);
    const yOff = Math.floor(height * 0.5 + random() * height * 0.5);
    const xOff = Math.floor(random() * width * 0.5);
    for (let i = 0; i < total; i++) {
      const yVal = yData[i];
      const xVal = xData[i];
      out[i * 2] = ((yVal + yOff) % height + height) % height;
      out[i * 2 + 1] = ((xVal + xOff) % width + width) % width;
    }
    const ctx = (yIndex && yIndex.ctx) || (xIndex && xIndex.ctx) || null;
    return Tensor.fromArray(ctx, out, [height, width, 2]);
  });
}

export async function posterize(tensor, shape, time, speed, levels = 9) {
  if (levels === 0) return tensor;
  const outShape = shape.slice();
  let t = tensor;
  if (outShape[2] === 3) {
    t = await fromSRGB(t);
  }
  let src = await t.read();
  const expected = outShape[0] * outShape[1] * outShape[2];
  if (src.length !== expected) {
    const pixels = outShape[0] * outShape[1];
    const srcChannels = src.length / pixels;
    const tmp = new Float32Array(expected);
    for (let i = 0; i < pixels; i++) {
      const srcBase = i * srcChannels;
      const dstBase = i * outShape[2];
      for (let k = 0; k < outShape[2]; k++) {
        const srcIdx = srcBase + Math.min(k, srcChannels - 1);
        tmp[dstBase + k] = src[srcIdx];
      }
    }
    src = tmp;
  }
  const out = new Float32Array(expected);
  const levelFactor = levels;
  const halfStep = 0.5 / levelFactor;
  for (let i = 0; i < expected; i++) {
    out[i] = Math.floor(src[i] * levelFactor + halfStep) / levelFactor;
  }
  let result = Tensor.fromArray(t.ctx, out, outShape);
  if (outShape[2] === 3) {
    result = await toSRGB(result);
  }
  return result;
}
register("posterize", posterize, { levels: 9 });

export async function smoothstep(tensor, shape, time, speed, a = 0, b = 1) {
  const src = await tensor.read();
  const out = new Float32Array(src.length);
  const inv = 1 / (b - a || 1);
  for (let i = 0; i < src.length; i++) {
    let t = (src[i] - a) * inv;
    if (t < 0) t = 0;
    else if (t > 1) t = 1;
    out[i] = t * t * (3 - 2 * t);
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register("smoothstep", smoothstep, { a: 0, b: 1 });

export function convolve(
  tensor,
  shape,
  time,
  speed,
  kernel = ValueMask.conv2d_blur,
  withNormalize = true,
  alpha = 1,
) {
  let kernelArr = kernel;
  if (!Array.isArray(kernelArr[0])) {
    const kTensor = maskValues(kernel)[0];
    const [kh, kw] = maskShape(kernel);
    const flat = kTensor.read();
    kernelArr = [];
    for (let y = 0; y < kh; y++) {
      const row = [];
      for (let x = 0; x < kw; x++) {
        row.push(flat[y * kw + x]);
      }
      kernelArr.push(row);
    }
  }
  let out = convolution(tensor, kernelArr, { normalize: withNormalize });
  const finish = (o) => {
    const handle = (tensorOut) => {
      if (kernel === ValueMask.conv2d_edges) {
        const data = tensorOut.read();
        if (data && typeof data.then === 'function') {
          return data.then((arr) => {
            for (let i = 0; i < arr.length; i++) {
              arr[i] = Math.abs(arr[i] - 0.5) * 2;
            }
            let tOut = Tensor.fromArray(tensorOut.ctx, arr, tensorOut.shape);
            if (typeof alpha !== 'number' || alpha < 1) {
              const blended = blend(tensor, tOut, alpha);
              return blended && typeof blended.then === 'function'
                ? blended
                : blended;
            }
            return tOut;
          });
        }
        for (let i = 0; i < data.length; i++) {
          data[i] = Math.abs(data[i] - 0.5) * 2;
        }
        tensorOut = Tensor.fromArray(tensorOut.ctx, data, tensorOut.shape);
      }
      if (typeof alpha !== 'number' || alpha < 1) {
        const blended = blend(tensor, tensorOut, alpha);
        if (blended && typeof blended.then === 'function') return blended;
        return blended;
      }
      return tensorOut;
    };

    if (o && typeof o.then === 'function') {
      return o.then(handle);
    }
    return handle(o);
  };

  return finish(out);
}
register("convolve", convolve, {
  kernel: ValueMask.conv2d_blur,
  withNormalize: true,
  alpha: 1,
});

export async function fbm(
  tensor,
  shape,
  time,
  speed,
  freq = 4,
  octaves = 4,
  lacunarity = 2,
  gain = 0.5,
) {
  const [h, w, c] = shape;
  const ctx = tensor?.ctx ?? null;
  const baseFreq = Array.isArray(freq) ? freq : freqForShape(freq, [h, w]);
  const data = new Float32Array(h * w * c);
  for (let octave = 1; octave <= octaves; octave++) {
    const octaveFreq = baseFreq.map((f) =>
      Math.floor(f * Math.pow(lacunarity, octave - 1)),
    );
    if (octaveFreq[0] > h && octaveFreq[1] > w) {
      break;
    }
    const layerMaybe = await values(octaveFreq, shape, { ctx, time, speed });
    const layerDataMaybe = layerMaybe.read();
    const layerData =
      layerDataMaybe && typeof layerDataMaybe.then === "function"
        ? await layerDataMaybe
        : layerDataMaybe;
    const layerArray =
      layerData instanceof Float32Array
        ? layerData
        : new Float32Array(layerData ?? []);
    const weight = Math.pow(gain, octave);
    for (let i = 0; i < data.length; i++) {
      data[i] += layerArray[i] * weight;
    }
  }
  const tensorOut = Tensor.fromArray(ctx, data, shape);
  return await normalize(tensorOut);
}
register("fbm", fbm, { freq: 4, octaves: 4, lacunarity: 2, gain: 0.5 });

const TAU = Math.PI * 2;

export async function palette(tensor, shape, time, speed, name = null, alpha = 1) {
  if (!name) return tensor;
  const [h, w, c] = shape;
  if (c === 1 || c === 2) return tensor;

  const p = PALETTES[name];
  if (!p) throw new Error(`Unknown palette ${name}`);

  let alphaChan = null;
  let baseTensor = tensor;
  if (c === 4) {
    const srcMaybe = tensor.read();
    const src =
      srcMaybe && typeof srcMaybe.then === "function" ? await srcMaybe : srcMaybe;
    const rgbData = new Float32Array(h * w * 3);
    alphaChan = new Float32Array(h * w);
    for (let i = 0; i < h * w; i++) {
      const base = i * 4;
      rgbData[i * 3] = src[base];
      rgbData[i * 3 + 1] = src[base + 1];
      rgbData[i * 3 + 2] = src[base + 2];
      alphaChan[i] = src[base + 3];
    }
    baseTensor = Tensor.fromArray(tensor.ctx, rgbData, [h, w, 3]);
  }

  const clamped = await clamp01(baseTensor);
  const labTensor = await rgbToOklab(clamped);
  const labMaybe = labTensor.read();
  const lab =
    labMaybe && typeof labMaybe.then === "function" ? await labMaybe : labMaybe;
  const out = new Float32Array(h * w * 3);
  for (let i = 0; i < h * w; i++) {
    const t = lab[i * 3];
    out[i * 3] =
      p.offset[0] +
      p.amp[0] * Math.cos(TAU * (p.freq[0] * t * 0.875 + 0.0625 + p.phase[0] + time));
    out[i * 3 + 1] =
      p.offset[1] +
      p.amp[1] * Math.cos(TAU * (p.freq[1] * t * 0.875 + 0.0625 + p.phase[1] + time));
    out[i * 3 + 2] =
      p.offset[2] +
      p.amp[2] * Math.cos(TAU * (p.freq[2] * t * 0.875 + 0.0625 + p.phase[2] + time));
  }
  const colored = Tensor.fromArray(tensor.ctx, out, [h, w, 3]);

  let tBlend;
  if (typeof alpha === "number") {
    tBlend = (1 - Math.cos(alpha * Math.PI)) / 2;
  } else {
    const aDataMaybe = alpha.read();
    const aData =
      aDataMaybe && typeof aDataMaybe.then === "function"
        ? await aDataMaybe
        : aDataMaybe;
    const tData = new Float32Array(aData.length);
    for (let i = 0; i < aData.length; i++) {
      tData[i] = (1 - Math.cos(aData[i] * Math.PI)) / 2;
    }
    tBlend = Tensor.fromArray(alpha.ctx, tData, alpha.shape);
  }

  let blended = await blend(baseTensor, colored, tBlend);

  if (alphaChan) {
    const blendedDataMaybe = blended.read();
    const blendedData =
      blendedDataMaybe && typeof blendedDataMaybe.then === "function"
        ? await blendedDataMaybe
        : blendedDataMaybe;
    const final = new Float32Array(h * w * 4);
    for (let i = 0; i < h * w; i++) {
      final[i * 4] = blendedData[i * 3];
      final[i * 4 + 1] = blendedData[i * 3 + 1];
      final[i * 4 + 2] = blendedData[i * 3 + 2];
      final[i * 4 + 3] = alphaChan[i];
    }
    blended = Tensor.fromArray(tensor.ctx, final, [h, w, 4]);
  }

  return markPresentationNormalized(blended);
}
register("palette", palette, { name: null, alpha: 1 });

export async function invert(tensor, shape, time, speed) {
  const ctx = tensor.ctx;
  if (ctx && ctx.device) {
    try {
      const tex = ensureTextureTensor(tensor);
      return await runUnaryShader(tex, UNARY_OP_INVERT);
    } catch (e) {
      console.warn("WebGPU invert fallback to CPU", e);
    }
  }
  const src = await tensor.read();
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    out[i] = 1 - src[i];
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register("invert", invert, {});

export async function vortex(tensor, shape, time, speed, displacement = 64) {
  const valueShape = [shape[0], shape[1], 1];
  let dispMap = await singularity(null, valueShape, time, speed);
  dispMap = await normalize(dispMap);
  let x = await convolve(
    dispMap,
    valueShape,
    time,
    speed,
    ValueMask.conv2d_deriv_x,
    false,
  );
  let y = await convolve(
    dispMap,
    valueShape,
    time,
    speed,
    ValueMask.conv2d_deriv_y,
    false,
  );
  let fader = await singularity(
    null,
    valueShape,
    time,
    speed,
    VoronoiDiagramType.range,
    DistanceMetric.chebyshev,
  );
  fader = await invert(await normalize(fader), valueShape, time, speed);
  const disp = simplexRandom(time, undefined, speed) * 100 * displacement;
  const ctx = tensor.ctx;
  if (
    ctx &&
    ctx.device &&
    tensor.handle instanceof GPUTexture &&
    x.handle instanceof GPUTexture &&
    y.handle instanceof GPUTexture &&
    fader.handle instanceof GPUTexture
  ) {
    try {
      const outSize = shape[0] * shape[1] * shape[2];
      const outBuf = ctx.device.createBuffer({
        size: outSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const paramsBuf = ctx.createGPUBuffer(
        new Float32Array([shape[1], shape[0], shape[2], disp, 0, 0, 0, 0]),
        GPUBufferUsage.UNIFORM,
      );
      await ctx.runCompute(
        VORTEX_WGSL,
        [
          { binding: 0, resource: tensor.handle.createView() },
          { binding: 1, resource: x.handle.createView() },
          { binding: 2, resource: y.handle.createView() },
          { binding: 3, resource: fader.handle.createView() },
          { binding: 4, resource: { buffer: outBuf } },
          { binding: 5, resource: { buffer: paramsBuf } },
        ],
        Math.ceil(shape[1] / 8),
        Math.ceil(shape[0] / 8),
      );
      const out = await ctx.readGPUBuffer(outBuf, outSize * 4);
      return Tensor.fromArray(ctx, out, shape);
    } catch (e) {
      console.warn('WebGPU vortex fallback to CPU', e);
    }
  }
  const [xData, yData, fData] = await Promise.all([
    x.read(),
    y.read(),
    fader.read(),
  ]);
  for (let i = 0; i < xData.length; i++) {
    xData[i] *= fData[i];
    yData[i] *= fData[i];
  }
  x = Tensor.fromArray(tensor.ctx, xData, valueShape);
  y = Tensor.fromArray(tensor.ctx, yData, valueShape);
  return refractOp(tensor, x, y, disp);
}
register("vortex", vortex, { displacement: 64 });

export async function aberration(
  tensor,
  shape,
  time,
  speed,
  displacement = 0.005,
) {
  const [h, w, c] = shape;
  if (c !== 3) return tensor;

  const displacementPixels = Math.floor(
    w * displacement * simplexRandom(time, undefined, speed),
  );
  const hueShift = random() * 0.1 - 0.05;

  const maskTensor = await singularity(null, [h, w, 1], time, speed);
  const maskData = await maskTensor.read();
  for (let i = 0; i < maskData.length; i++) {
    maskData[i] = Math.pow(maskData[i], 3);
  }

  const xIndex = new Float32Array(h * w);
  const yIndex = new Int32Array(h * w);
  const gradient = new Float32Array(h * w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      xIndex[idx] = x;
      yIndex[idx] = y;
      gradient[idx] = w > 1 ? x / (w - 1) : 0;
    }
  }

  const tinted = await adjustHue(tensor, hueShift);
  const src = await tinted.read();
  const blendLinear = (a, b, t) => a * (1 - t) + b * t;
  const blendCosine = (a, b, g) => {
    const clamped = Math.max(0, Math.min(1, g));
    const weight = (1 - Math.cos(clamped * Math.PI)) / 2;
    return a * (1 - weight) + b * weight;
  };
  const clampIndex = (value) => {
    if (value <= 0) return 0;
    if (value >= w - 1) return w - 1;
    return Math.floor(value);
  };

  const out = new Float32Array(h * w * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const base = idx * 3;
      const xFloat = xIndex[idx];
      const g = gradient[idx];
      const m = maskData[idx];

      let redOffset = Math.min(w - 1, xFloat + displacementPixels);
      redOffset = blendLinear(redOffset, xFloat, g);
      redOffset = blendCosine(xFloat, redOffset, m);
      const redX = clampIndex(redOffset);

      let blueOffset = Math.max(0, xFloat - displacementPixels);
      blueOffset = blendLinear(xFloat, blueOffset, g);
      blueOffset = blendCosine(xFloat, blueOffset, m);
      const blueX = clampIndex(blueOffset);

      const greenOffset = blendCosine(xFloat, xFloat, m);
      const greenX = clampIndex(greenOffset);

      const rowBase = yIndex[idx] * w;
      out[base] = src[(rowBase + redX) * 3];
      out[base + 1] = src[(rowBase + greenX) * 3 + 1];
      out[base + 2] = src[(rowBase + blueX) * 3 + 2];
    }
  }

  const displaced = Tensor.fromArray(tensor.ctx, out, shape);
  return adjustHue(displaced, -hueShift);
}
register("aberration", aberration, { displacement: 0.005 });

export async function glitch(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const baseMaybe = values(4, [h, w, 1], {
    ctx: tensor.ctx,
    time,
    speed: speed * 50,
  });
  const base =
    baseMaybe && typeof baseMaybe.then === "function"
      ? await baseMaybe
      : baseMaybe;
  const noiseMaybe = base.read();
  const noise =
    noiseMaybe && typeof noiseMaybe.then === "function"
      ? await noiseMaybe
      : noiseMaybe;
  const src = await tensor.read();
  const out = new Float32Array(h * w * c);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const shiftAmt = Math.floor(noise[idx] * 4);
      for (let k = 0; k < c; k++) {
        let sx = x;
        if (k === 0) sx = (x + shiftAmt) % w;
        else if (k === 2) sx = ((x - shiftAmt) % w + w) % w;
        const srcBase = (y * w + sx) * c + k;
        out[idx * c + k] = src[srcBase];
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register("glitch", glitch, {});

export async function vhs(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const scanNoise = await values(Math.floor(h * 0.5) + 1, [h, w, 1], {
    time,
    speed: speed * 100,
    ctx,
  });
  const gradNoise = await values(5, [h, w, 1], { time, speed, ctx });

  if (
    ctx &&
    ctx.device &&
    tensor.handle instanceof GPUTexture &&
    scanNoise.handle instanceof GPUTexture &&
    gradNoise.handle instanceof GPUTexture
  ) {
    const outBuf = ctx.createGPUBuffer(
      new Float32Array(h * w * c),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    const paramsBuf = ctx.createGPUBuffer(
      new Float32Array([w, h, c, 0, 0, 0, 0, 0]),
      GPUBufferUsage.UNIFORM,
    );
    await ctx.runCompute(
      VHS_WGSL,
      [
        { binding: 0, resource: tensor.handle.createView() },
        { binding: 1, resource: scanNoise.handle.createView() },
        { binding: 2, resource: gradNoise.handle.createView() },
        { binding: 3, resource: { buffer: outBuf } },
        { binding: 4, resource: { buffer: paramsBuf } },
      ],
      Math.ceil(w / 8),
      Math.ceil(h / 8),
    );
    return new Tensor(ctx, outBuf, shape);
  }

  const scanArr = await scanNoise.read();
  const gradArr = await gradNoise.read();
  const src = await tensor.read();
  const blended = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    let g = gradArr[i] - 0.5;
    if (g < 0) g = 0;
    g = Math.min(g * 2, 1);
    const noise = scanArr[i];
    for (let k = 0; k < c; k++) {
      blended[i * c + k] = src[i * c + k] * (1 - g) + noise * g;
    }
  }
  const out = new Float32Array(h * w * c);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      let g = gradArr[idx] - 0.5;
      if (g < 0) g = 0;
      g = Math.min(g * 2, 1);
      const xOff = Math.floor(scanArr[idx] * w * g * g);
      const srcX = (x - xOff + w) % w;
      for (let k = 0; k < c; k++) {
        out[idx * c + k] = blended[(y * w + srcX) * c + k];
      }
    }
  }
  return Tensor.fromArray(ctx, out, shape);
}
register("vhs", vhs, {});

export async function lensWarp(tensor, shape, time, speed, displacement = 0.0625) {
  const valueShape = [shape[0], shape[1], 1];
  const maskTensor = await singularity(null, valueShape, time, speed);
  const mask = await maskTensor.read();
  for (let i = 0; i < mask.length; i++) mask[i] = mask[i] ** 5;
  const noiseTensor = await values(2, valueShape, {
    ctx: tensor.ctx,
    time,
    speed,
    splineOrder: InterpolationType.cosine,
  });
  const noise = await noiseTensor.read();
  const cosData = new Float32Array(noise.length);
  const sinData = new Float32Array(noise.length);
  for (let i = 0; i < noise.length; i++) {
    const base = (noise[i] * 2 - 1) * mask[i];
    const angle = base * Math.PI * 2;
    let cx = Math.cos(angle) * 0.5 + 0.5;
    let sy = Math.sin(angle) * 0.5 + 0.5;
    if (cx < 0) cx = 0;
    else if (cx > 1) cx = 1;
    if (sy < 0) sy = 0;
    else if (sy > 1) sy = 1;
    cosData[i] = cx;
    sinData[i] = sy;
  }
  const refX = Tensor.fromArray(tensor.ctx, cosData, valueShape);
  const refY = Tensor.fromArray(tensor.ctx, sinData, valueShape);
  return await refractOp(tensor, refX, refY, displacement);
}
register("lensWarp", lensWarp, { displacement: 0.0625 });
register("lens_warp", lensWarp, { displacement: 0.0625 });

export async function lensDistortion(tensor, shape, time, speed, displacement = 1) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  if (
    ctx &&
    ctx.device &&
    typeof GPUTexture !== "undefined"
  ) {
    let tex = tensor;
    if (!(tensor.handle instanceof GPUTexture)) {
      const data = await tensor.read();
      tex = Tensor.fromArray(ctx, data, shape);
    }
    const outBuf = ctx.createGPUBuffer(
      new Float32Array(h * w * c),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    const paramsBuf = ctx.createGPUBuffer(
      new Float32Array([w, h, c, displacement, 0, 0, 0, 0]),
      GPUBufferUsage.UNIFORM,
    );
    await ctx.runCompute(
      LENS_DISTORTION_WGSL,
      [
        { binding: 0, resource: tex.handle.createView() },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: paramsBuf } },
      ],
      Math.ceil(w / 8),
      Math.ceil(h / 8),
    );
    return new Tensor(ctx, outBuf, shape);
  }
  const src = await tensor.read();
  const out = new Float32Array(h * w * c);
  const maxDist = Math.sqrt(0.5 * 0.5 + 0.5 * 0.5) || 1;
  const zoom = displacement < 0 ? displacement * -0.25 : 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const xIndex = x / w;
      const yIndex = y / h;
      const xDist = xIndex - 0.5;
      const yDist = yIndex - 0.5;
      let centerDist = 1 - distance(xDist, yDist) / maxDist;
      if (centerDist < 0) centerDist = 0;
      else if (centerDist > 1) centerDist = 1;
      const xOff =
        (xIndex -
          xDist * zoom -
          xDist * centerDist * centerDist * displacement) *
        w;
      const yOff =
        (yIndex -
          yDist * zoom -
          yDist * centerDist * centerDist * displacement) *
        h;
      const xi = ((Math.trunc(xOff) % w) + w) % w;
      const yi = ((Math.trunc(yOff) % h) + h) % h;
      const srcIdx = (yi * w + xi) * c;
      const dstIdx = (y * w + x) * c;
      for (let k = 0; k < c; k++) {
        out[dstIdx + k] = src[srcIdx + k] || 0;
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register("lensDistortion", lensDistortion, { displacement: 1 });
register("lens_distortion", lensDistortion, { displacement: 1 });

export async function degauss(tensor, shape, time, speed, displacement = 0.0625) {
  const [h, w, c] = shape;
  const channelShape = [h, w, 1];
  const src = await tensor.read();
  const channels = Math.min(3, c);
  const ctx = tensor.ctx;
    if (
      ctx &&
      ctx.device &&
      typeof GPUTexture !== "undefined" &&
      tensor.handle instanceof GPUTexture
    ) {
    const channelTensors = [];
    for (let k = 0; k < channels; k++) {
      const channelData = new Float32Array(h * w);
      for (let i = 0; i < h * w; i++) channelData[i] = src[i * c + k] || 0;
      const channelTensor = Tensor.fromArray(ctx, channelData, channelShape);
      channelTensors.push(
        await lensWarp(channelTensor, channelShape, time, speed, displacement),
      );
    }
    let alphaTensor = null;
    if (c > 3) {
      const alphaData = new Float32Array(h * w);
      for (let i = 0; i < h * w; i++) alphaData[i] = src[i * c + 3] || 0;
      alphaTensor = Tensor.fromArray(ctx, alphaData, channelShape);
    }
    const device = ctx.device;
    const outSize = h * w * c;
    const outBuf = device.createBuffer({
      size: outSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const paramsArr = new Float32Array([w, h, c, 0, 0, 0, 0, 0]);
    const paramsBuf = ctx.createGPUBuffer(paramsArr, GPUBufferUsage.UNIFORM);
    await ctx.runCompute(
      DEGAUSS_WGSL,
      [
        { binding: 0, resource: channelTensors[0].handle.createView() },
        {
          binding: 1,
          resource: (channelTensors[1] || channelTensors[0]).handle.createView(),
        },
        {
          binding: 2,
          resource: (channelTensors[2] || channelTensors[0]).handle.createView(),
        },
        {
          binding: 3,
          resource: (alphaTensor || channelTensors[0]).handle.createView(),
        },
        { binding: 4, resource: { buffer: outBuf } },
        { binding: 5, resource: { buffer: paramsBuf } },
      ],
      Math.ceil(w / 8),
      Math.ceil(h / 8),
    );
    const out = await ctx.readGPUBuffer(outBuf, outSize * 4);
    return Tensor.fromArray(ctx, out, shape);
  }
  const out = new Float32Array(h * w * c);
  for (let k = 0; k < channels; k++) {
    const channelData = new Float32Array(h * w);
    for (let i = 0; i < h * w; i++) channelData[i] = src[i * c + k] || 0;
    const channelTensor = Tensor.fromArray(
      tensor.ctx,
      channelData,
      channelShape,
    );
    const warpedTensor = await lensWarp(
      channelTensor,
      channelShape,
      time,
      speed,
      displacement,
    );
    const warped = await warpedTensor.read();
    for (let i = 0; i < h * w; i++) out[i * c + k] = warped[i];
  }
  if (c > 3) {
    for (let i = 0; i < h * w; i++) out[i * c + 3] = src[i * c + 3];
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register("degauss", degauss, { displacement: 0.0625 });

  export async function scanlineError(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const errorFreq = Math.floor(h * 0.5) || 1;
    let errorLineTensor = await values(errorFreq, [h, w, 1], {
      time,
      speed: speed * 10,
      distrib: ValueDistribution.exp,
    });
    let errorLine = await errorLineTensor.read();
    let errorSwerveTensor = await values(Math.floor(h * 0.01) || 1, [h, w, 1], {
      time,
      speed,
      distrib: ValueDistribution.exp,
    });
    let errorSwerve = await errorSwerveTensor.read();
    const whiteNoiseTensor = await values(errorFreq, [h, w, 1], {
      time,
      speed: speed * 100,
    });
    const whiteNoise = await whiteNoiseTensor.read();
  const src = await tensor.read();
  const out = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    let el = errorLine[i] - 0.5;
    if (el < 0) el = 0;
    let es = errorSwerve[i] - 0.5;
    if (es < 0) es = 0;
    el *= es;
    es *= 2;
    let wn = whiteNoise[i] * es;
    errorLine[i] = el;
    whiteNoise[i] = wn;
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const shift = Math.floor((errorLine[idx] + whiteNoise[idx]) * w * 0.025);
      const srcX = (x - shift + w) % w;
      const srcIdx = (y * w + srcX) * c;
      for (let k = 0; k < c; k++) {
        let val = src[srcIdx + k];
        val = Math.min(val + errorLine[idx] * whiteNoise[idx] * 4, 1);
        out[idx * c + k] = val;
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register("scanline_error", scanlineError, {});
register("scanline_error", scanlineError, {});

export async function crt(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const valueShape = [h, w, 1];

  // Horizontal scanlines
  let scanNoise = await values([2, 1], [2, 1, 1], {
    time,
    speed: speed * 0.1,
    splineOrder: InterpolationType.constant,
    ctx,
  });
  scanNoise = await normalize(scanNoise);
  const tileH = Math.max(1, Math.floor(h * 0.125));
  scanNoise = await expandTile(scanNoise, [2, 1, 1], [tileH * 2, w, 1], false);
  scanNoise = await resample(scanNoise, valueShape);
  scanNoise = await lensWarp(scanNoise, valueShape, time, speed);

  if (
    ctx &&
    ctx.device &&
    tensor.handle instanceof GPUTexture &&
    scanNoise.handle instanceof GPUTexture
  ) {
    const displacement = 0.0125 + random() * 0.00625; // RNG call 1
    const disp = Math.round(w * displacement * simplexRandom(time, undefined, speed));
    const hueShift = random() * 0.1 - 0.05; // RNG call 2
    const randHue = random() * 0.25 - 0.125; // RNG call 3
    const vigAlpha = random() * 0.175; // RNG call 4
    const outBuf = ctx.createGPUBuffer(
      new Float32Array(h * w * c),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    const paramsBuf = ctx.createGPUBuffer(
      new Float32Array([
        w,
        h,
        c,
        disp,
        hueShift,
        randHue,
        1.125,
        vigAlpha,
        0,
        0,
        0,
      ]),
      GPUBufferUsage.UNIFORM,
    );
    await ctx.runCompute(
      CRT_WGSL,
      [
        { binding: 0, resource: tensor.handle.createView() },
        { binding: 1, resource: scanNoise.handle.createView() },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: paramsBuf } },
      ],
      Math.ceil(w / 8),
      Math.ceil(h / 8),
    );
    let outTensor = new Tensor(ctx, outBuf, shape);
    const data = await outTensor.read();
    let mean = 0;
    for (let i = 0; i < data.length; i++) mean += data[i];
    mean /= data.length;
    for (let i = 0; i < data.length; i++) {
      data[i] = (data[i] - mean) * 1.25 + mean;
    }
    return Tensor.fromArray(ctx, data, shape);
  }

  const scan = await scanNoise.read();
  const src = await tensor.read();
  const blended = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    const s = scan[i];
    for (let k = 0; k < c; k++) {
      const t = src[i * c + k];
      blended[i * c + k] = t * 0.95 + (t + s) * s * 0.05;
    }
  }
  let outTensor = await clamp01(Tensor.fromArray(tensor.ctx, blended, shape));

  if (c === 3) {
    outTensor = await aberration(
      outTensor,
      shape,
      time,
      speed,
      0.0125 + random() * 0.00625,
    );
    outTensor = await randomHue(outTensor, shape, time, speed, 0.125);
    outTensor = await saturation(outTensor, shape, time, speed, 1.125);
  }

  outTensor = await vignette(outTensor, shape, time, speed, 0, random() * 0.175);

  const data = await outTensor.read();
  let mean = 0;
  for (let i = 0; i < data.length; i++) mean += data[i];
  mean /= data.length;
  for (let i = 0; i < data.length; i++) {
    data[i] = (data[i] - mean) * 1.25 + mean;
  }

  return Tensor.fromArray(outTensor.ctx, data, shape);
}
register("crt", crt, {});

export async function reindex(tensor, shape, time, speed, displacement = 0.5) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  if (ctx && ctx.device && tensor.handle instanceof GPUTexture) {
    const outBuf = ctx.createGPUBuffer(
      new Float32Array(h * w * c),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    const paramsBuf = ctx.createGPUBuffer(
      new Float32Array([w, h, c, displacement, Math.min(h, w), 0, 0, 0]),
      GPUBufferUsage.UNIFORM,
    );
    await ctx.runCompute(
      REINDEX_WGSL,
      [
        { binding: 0, resource: tensor.handle.createView() },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: paramsBuf } },
      ],
      Math.ceil(w / 8),
      Math.ceil(h / 8),
    );
    return new Tensor(ctx, outBuf, shape);
  }
  const src = await tensor.read();
  let lumTensor;
  if (c === 1 || c === 2) {
    const lum = new Float32Array(h * w);
    for (let i = 0; i < h * w; i++) {
      lum[i] = src[i * c];
    }
    lumTensor = Tensor.fromArray(tensor.ctx, lum, [h, w, 1]);
  } else {
    let rgbTensor;
    if (c === 3) {
      rgbTensor = await clamp01(tensor);
    } else {
      const clamped = await clamp01(tensor);
      const clampedData = await clamped.read();
      const rgb = new Float32Array(h * w * 3);
      for (let i = 0; i < h * w; i++) {
        const base = i * c;
        rgb[i * 3] = clampedData[base];
        rgb[i * 3 + 1] = clampedData[base + 1];
        rgb[i * 3 + 2] = clampedData[base + 2];
      }
      rgbTensor = Tensor.fromArray(tensor.ctx, rgb, [h, w, 3]);
    }
    const labTensor = await rgbToOklab(rgbTensor);
    const lab = await labTensor.read();
    const lum = new Float32Array(h * w);
    for (let i = 0; i < h * w; i++) lum[i] = lab[i * 3];
    lumTensor = Tensor.fromArray(tensor.ctx, lum, [h, w, 1]);
  }
  lumTensor = await normalize(lumTensor);
  const lum = await lumTensor.read();
  const mod = Math.min(h, w);
  const out = new Float32Array(h * w * c);
  const EPSILON = 5e-6;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const r = lum[idx];
      const off = r * displacement * mod + r;
      let xPos = off % w;
      let yPos = off % h;
      if (xPos < 0) xPos += w;
      if (yPos < 0) yPos += h;
      let xInt = Math.floor(xPos);
      let yInt = Math.floor(yPos);
      const xFrac = xPos - xInt;
      const yFrac = yPos - yInt;
      if (xFrac > 1 - EPSILON) {
        xInt = (xInt + 1) % w;
      }
      if (yFrac > 1 - EPSILON) {
        yInt = (yInt + 1) % h;
      }
      const srcIdx = (yInt * w + xInt) * c;
      for (let k = 0; k < c; k++) {
        out[idx * c + k] = src[srcIdx + k];
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register("reindex", reindex, { displacement: 0.5 });

export async function ripple(
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
  let ref = reference || (await values(freq, [h, w, 1], { ctx, time, speed, splineOrder }));
  const rand = simplexRandom(time, undefined, speed);
  if (ctx && ctx.device) {
    let refTex = ref;
    if (refTex.ctx !== ctx) refTex = Tensor.fromArray(ctx, await ref.read(), ref.shape);
    if (tensor.handle instanceof GPUTexture && refTex.handle instanceof GPUTexture) {
      const outBuf = ctx.createGPUBuffer(
        new Float32Array(h * w * c),
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      );
      const paramsBuf = ctx.createGPUBuffer(
        new Float32Array([w, h, c, displacement, kink, rand, 0, 0]),
        GPUBufferUsage.UNIFORM,
      );
      await ctx.runCompute(
        RIPPLE_WGSL,
        [
          { binding: 0, resource: tensor.handle.createView() },
          { binding: 1, resource: refTex.handle.createView() },
          { binding: 2, resource: { buffer: outBuf } },
          { binding: 3, resource: { buffer: paramsBuf } },
        ],
        Math.ceil(w / 8),
        Math.ceil(h / 8),
      );
      return new Tensor(ctx, outBuf, shape);
    }
  }
  let refTensor = ref;
  if (refTensor && typeof refTensor.then === "function") {
    refTensor = await refTensor;
  }
  let refValue = toValueMap(refTensor);
  if (refValue && typeof refValue.then === "function") {
    refValue = await refValue;
  }
  if (refValue.shape[0] !== h || refValue.shape[1] !== w) {
    refValue = await resample(refValue, [h, w, 1]);
  }
  const refData = await refValue.read();
  const src = await tensor.read();
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
        out[idx * c + k] = c0 * (1 - sy) + c1 * sy;
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register("ripple", ripple, {
  freq: 2,
  displacement: 1,
  kink: 1,
  reference: null,
  splineOrder: InterpolationType.bicubic,
});

export async function colorMap(
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
  if (
    ctx &&
    ctx.device &&
    clut.ctx === ctx &&
    tensor.handle instanceof GPUTexture &&
    clut.handle instanceof GPUTexture
  ) {
    const cc = clut.shape[2];
    const outBuf = ctx.createGPUBuffer(
      new Float32Array(h * w * cc),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    const paramsBuf = ctx.createGPUBuffer(
      new Float32Array([
        w,
        h,
        c,
        displacement,
        horizontal ? 1 : 0,
        clut.shape[1],
        clut.shape[0],
        cc,
      ]),
      GPUBufferUsage.UNIFORM,
    );
    await ctx.runCompute(
      COLOR_MAP_WGSL,
      [
        { binding: 0, resource: tensor.handle.createView() },
        { binding: 1, resource: clut.handle.createView() },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: paramsBuf } },
      ],
      Math.ceil(w / 8),
      Math.ceil(h / 8),
    );
    return new Tensor(ctx, outBuf, [h, w, cc]);
  }
  const [ch, cw, cc] = clut.shape;
  const clutRaw = await clut.read();
  const clutFloat = new Float32Array(clutRaw.length);
  for (let i = 0; i < clutRaw.length; i++) {
    const v = clutRaw[i];
    clutFloat[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  let clutTensor = Tensor.fromArray(clut.ctx, clutFloat, [ch, cw, cc]);
  if (ch !== h || cw !== w) {
    clutTensor = await resample(clutTensor, [h, w, cc]);
  }
  const clutData = await clutTensor.read();

  const valueMap = await toValueMap(tensor);
  const normalized = await normalize(valueMap);
  const refData = await normalized.read();

  const out = new Float32Array(h * w * cc);
  const mod = (n, m) => ((n % m) + m) % m;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const ref = refData[idx] * displacement;
      const xOffset = Math.floor(ref * (w - 1));
      const yOffset = Math.floor(ref * (h - 1));
      const xi = mod(x + xOffset, w);
      const yi = horizontal ? y : mod(y + yOffset, h);
      const srcIdx = (yi * w + xi) * cc;
      const outIdx = idx * cc;
      for (let k = 0; k < cc; k++) {
        out[outIdx + k] = clutData[srcIdx + k];
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, cc]);
}
register("color_map", colorMap, {
  clut: null,
  horizontal: false,
  displacement: 0.5,
});

export function falseColor(
  tensor,
  shape,
  time,
  speed,
  horizontal = false,
  displacement = 0.5,
) {
  const clut = values(2, shape, { ctx: tensor.ctx, time, speed });
  return normalize(
    colorMap(tensor, shape, time, speed, clut, horizontal, displacement),
  );
}
register("false_color", falseColor, { horizontal: false, displacement: 0.5 });

export async function tint(tensor, shape, time, speed, alpha = 0.5) {
  const [h, w, c] = shape;
  if (c < 3) return tensor;
  // Consume similar noise to maintain randomness parity with Python impl
  values(3, shape, { ctx: tensor.ctx, time, speed, corners: true });

  const rand1 = random() * 0.333;
  const rand2 = random();
  const ctx = tensor.ctx;
  if (ctx && ctx.device && tensor.handle instanceof GPUTexture) {
    const outBuf = ctx.createGPUBuffer(
      new Float32Array(h * w * c),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    const paramsBuf = ctx.createGPUBuffer(
      // Pad to 32 bytes for uniform buffer alignment
      new Float32Array([w, h, c, alpha, rand1, rand2, 0, 0]),
      GPUBufferUsage.UNIFORM,
    );
    await ctx.runCompute(
      TINT_WGSL,
      [
        { binding: 0, resource: tensor.handle.createView() },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: paramsBuf } },
      ],
      Math.ceil(w / 8),
      Math.ceil(h / 8),
    );
    return new Tensor(ctx, outBuf, shape);
  }

  const src = await tensor.read();
  let alphaChan = null;
  let rgbData;

  if (c === 4) {
    rgbData = new Float32Array(h * w * 3);
    alphaChan = new Float32Array(h * w);
    for (let i = 0; i < h * w; i++) {
      const base = i * 4;
      rgbData[i * 3] = src[base];
      rgbData[i * 3 + 1] = src[base + 1];
      rgbData[i * 3 + 2] = src[base + 2];
      alphaChan[i] = src[base + 3];
    }
  } else {
    rgbData = src.slice();
  }

  const colorData = new Float32Array(h * w * 3);
  for (let i = 0; i < h * w; i++) {
    const r = rgbData[i * 3];
    const g = rgbData[i * 3 + 1];
    const b = rgbData[i * 3 + 2];
    colorData[i * 3] = (r * 0.333 + rand1 + rand2) % 1.0;
    colorData[i * 3 + 1] = g;
    colorData[i * 3 + 2] = b;
  }

  const baseTensor = Tensor.fromArray(tensor.ctx, rgbData, [h, w, 3]);
  const hsvTensor = await rgbToHsv(baseTensor);
  const hsv = await hsvTensor.read();
  const hsvMix = new Float32Array(h * w * 3);
  for (let i = 0; i < h * w; i++) {
    hsvMix[i * 3] = colorData[i * 3];
    hsvMix[i * 3 + 1] = colorData[i * 3 + 1];
    hsvMix[i * 3 + 2] = hsv[i * 3 + 2];
  }
  const colorized = await hsvToRgb(
    Tensor.fromArray(tensor.ctx, hsvMix, [h, w, 3])
  );
  let out = await blend(baseTensor, colorized, alpha);

  if (c === 4) {
    const outData = await out.read();
    const final = new Float32Array(h * w * 4);
    for (let i = 0; i < h * w; i++) {
      final[i * 4] = outData[i * 3];
      final[i * 4 + 1] = outData[i * 3 + 1];
      final[i * 4 + 2] = outData[i * 3 + 2];
      final[i * 4 + 3] = alphaChan[i];
    }
    out = Tensor.fromArray(tensor.ctx, final, [h, w, 4]);
  }

  return out;
}
register("tint", tint, { alpha: 0.5 });

export async function valueRefract(
  tensor,
  shape,
  time,
  speed,
  freq = 4,
  distrib = ValueDistribution.center_circle,
  displacement = 0.125,
) {
  const valueShape = [shape[0], shape[1], 1];
  const blendValues = await values(freq, valueShape, {
    ctx: tensor.ctx,
    distrib,
    time,
    speed,
  });
  return refractEffect(
    tensor,
    shape,
    time,
    speed,
    displacement,
    blendValues,
  );
}
register("value_refract", valueRefract, {
  freq: 4,
  distrib: ValueDistribution.center_circle,
  displacement: 0.125,
});

export async function refractEffect(
  tensor,
  shape,
  time,
  speed,
  displacement = 0.5,
  referenceX = null,
  referenceY = null,
  warpFreq = null,
  splineOrder = InterpolationType.bicubic,
  fromDerivative = false,
  signedRange = true,
  yFromOffset = false,
) {
  const [h, w, c] = shape;
  const valueShape = [h, w, 1];
  let rx = referenceX;
  let ry = referenceY;
  if (fromDerivative) {
    const kx = [
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1],
    ];
    const ky = [
      [-1, -2, -1],
      [0, 0, 0],
      [1, 2, 1],
    ];
    let gray = tensor;
    if (c > 1) {
      const src = await tensor.read();
      const out = new Float32Array(h * w);
      for (let i = 0; i < h * w; i++) {
        const base = i * c;
        const r = src[base];
        const g = src[base + 1] || 0;
        const b = src[base + 2] || 0;
        out[i] = r * 0.299 + g * 0.587 + b * 0.114;
      }
      gray = Tensor.fromArray(tensor.ctx, out, valueShape);
    }
    rx = await convolution(gray, kx, { normalize: false });
    ry = await convolution(gray, ky, { normalize: false });
  } else if (warpFreq !== null && warpFreq !== undefined) {
    rx = await values(warpFreq, valueShape, {
      ctx: tensor.ctx,
      distrib: ValueDistribution.simplex,
      time,
      speed,
      splineOrder,
    });
    ry = await values(warpFreq, valueShape, {
      ctx: tensor.ctx,
      distrib: ValueDistribution.simplex,
      time,
      speed,
      splineOrder,
    });
  } else {
      if (!rx) rx = tensor;
    if (!ry) {
      if (yFromOffset) {
        const xHalf = Math.floor(w * 0.5);
        const yHalf = Math.floor(h * 0.5);
        ry = await offsetTensor(rx, xHalf, yHalf);
      } else {
        rx = await rx;
        const rData = await rx.read();
        const cx = new Float32Array(rData.length);
        const cy = new Float32Array(rData.length);
        const tau32 = Math.fround(TAU);
        for (let i = 0; i < rData.length; i++) {
          const ang = Math.fround(rData[i]) * tau32;
          const cosVal = Math.fround(Math.cos(ang));
          const sinVal = Math.fround(Math.sin(ang));
          const cx01 = Math.fround(cosVal * 0.5 + 0.5);
          const cy01 = Math.fround(sinVal * 0.5 + 0.5);
          cx[i] = Math.min(Math.max(cx01, 0), 1);
          cy[i] = Math.min(Math.max(cy01, 0), 1);
        }
        rx = Tensor.fromArray(tensor.ctx, cx, rx.shape);
        ry = Tensor.fromArray(tensor.ctx, cy, rx.shape);
      }
    }
  }
  rx = await rx;
  ry = await ry;
  rx = await toValueMap(rx);
  ry = await toValueMap(ry);
  if (rx.shape[0] !== h || rx.shape[1] !== w) {
    rx = await upsample(rx, h / rx.shape[0]);
  }
  if (ry.shape[0] !== h || ry.shape[1] !== w) {
    ry = await upsample(ry, h / ry.shape[0]);
  }
  return await refractOp(
    tensor,
    rx,
    ry,
    displacement,
    splineOrder,
    signedRange && !fromDerivative,
  );
}
register("refract_effect", refractEffect, {
  displacement: 0.5,
  referenceX: null,
  referenceY: null,
  warpFreq: null,
  splineOrder: InterpolationType.bicubic,
  fromDerivative: false,
  signedRange: true,
  yFromOffset: false,
});
register("refract", refractEffect, {
  displacement: 0.5,
  referenceX: null,
  referenceY: null,
  warpFreq: null,
  splineOrder: InterpolationType.bicubic,
  fromDerivative: false,
  signedRange: true,
  yFromOffset: false,
});

export async function fxaaEffect(tensor, shape, time, speed) {
  return await fxaa(tensor);
}
register("fxaa_effect", fxaaEffect, {});
register("fxaa", fxaaEffect, {});

function periodicValue(t, v) {
  return (Math.sin((t - v) * TAU) + 1) * 0.5;
}

function makeRots(beh, n, time = 0, speed = 1) {
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
    rot.set(makeRots(1, q, time, speed), 0);
    rot.set(makeRots(2, q, time, speed), q);
    rot.set(makeRots(3, q, time, speed), q * 2);
    rot.set(makeRots(4, n - q * 3, time, speed), q * 3);
  } else if (beh === 10) {
    for (let i = 0; i < n; i++) rot[i] = periodicValue(time * speed, random());
  } else {
    rot.fill(base);
  }
  return rot;
}

export function wormsParams(
  shape,
  density = 4.0,
  stride = 1.0,
  strideDeviation = 0.05,
  behavior = 1,
  time = 0,
  speed = 1,
) {
  // RNG: random→y, random→x, randomNormal→stride, makeRots consumes further RNG
  const [h, w] = shape;
  const count = Math.floor(Math.max(w, h) * density);
  const y = new Float32Array(count);
  const x = new Float32Array(count);
  const strideVals = new Float32Array(count);
  const uniformY = randomUniform(count, 0, 1);
  const uniformX = randomUniform(count, 0, 1);
  const strideNoise = randomNormalArray(count, stride, strideDeviation);
  const strideScale = Math.max(w, h) / 1024.0;
  for (let i = 0; i < count; i++) {
    y[i] = uniformY[i] * (h - 1); // RNG[1]
    x[i] = uniformX[i] * (w - 1); // RNG[2]
    strideVals[i] = strideNoise[i] * strideScale; // RNG[3]
  }
  const rot = makeRots(behavior, count, time, speed);
  return { x: Array.from(x), y: Array.from(y), stride: Array.from(strideVals), rot: Array.from(rot) };
}

function offsetIndexInternal(yArr, height, xArr, width) {
  const yOff = Math.floor(height * 0.5 + random() * height * 0.5);
  const xOff = Math.floor(random() * width * 0.5);
  const n = yArr.length;
  const oy = new Int32Array(n);
  const ox = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    oy[i] = ((yArr[i] + yOff) % height + height) % height;
    ox[i] = ((xArr[i] + xOff) % width + width) % width;
  }
  return { y: oy, x: ox };
}

function offsetTensor(tensor, xOff, yOff) {
  const [h, w, c] = tensor.shape;
  if (xOff === 0 && yOff === 0) return tensor;
  const srcMaybe = tensor.read();
  if (srcMaybe && typeof srcMaybe.then === "function") {
    return srcMaybe.then((src) => {
      const out = new Float32Array(h * w * c);
      for (let y = 0; y < h; y++) {
        const yy = (((y + yOff) % h) + h) % h;
        for (let x = 0; x < w; x++) {
          const xx = (((x + xOff) % w) + w) % w;
          const srcIdx = (yy * w + xx) * c;
          const dstIdx = (y * w + x) * c;
          for (let k = 0; k < c; k++) out[dstIdx + k] = src[srcIdx + k];
        }
      }
      return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
    });
  }
  const src = srcMaybe;
  const out = new Float32Array(h * w * c);
  for (let y = 0; y < h; y++) {
    const yy = (((y + yOff) % h) + h) % h;
    for (let x = 0; x < w; x++) {
      const xx = (((x + xOff) % w) + w) % w;
      const srcIdx = (yy * w + xx) * c;
      const dstIdx = (y * w + x) * c;
      for (let k = 0; k < c; k++) out[dstIdx + k] = src[srcIdx + k];
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
}

async function centerMaskInternal(
  center,
  edges,
  shape,
  power = 2,
  distMetric = DistanceMetric.chebyshev,
) {
  const [h, w] = shape;
  const maskTensor = await singularity(
    null,
    [h, w, 1],
    0,
    1,
    VoronoiDiagramType.range,
    distMetric,
  );
  const mask = await maskTensor.read();
  for (let i = 0; i < mask.length; i++) {
    mask[i] = Math.min(Math.pow(mask[i], power), 1);
  }
  const maskT = Tensor.fromArray(center.ctx, mask, [h, w, 1]);
  return await blend(center, edges, maskT);
}

function voronoiColorRegions(tensor, shape, time, speed, xPts, yPts) {
  return voronoi(
    tensor,
    shape,
    time,
    speed,
    VoronoiDiagramType.color_regions,
    0,
    DistanceMetric.euclidean,
    3,
    1.0,
    0.0,
    false,
    true,
    3,
    1,
    PointDistribution.random,
    0,
    false,
    [xPts, yPts, xPts.length],
    true,
  );
}

export async function erosionWorms(
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
  xyBlend = 0,
) {
  const [h, w, c] = shape;
  if (tensor && tensor.ctx && tensor.ctx.device) {
    return await erosionWormsWebGPU(
      tensor,
      shape,
      time,
      speed,
      density,
      iterations,
      contraction,
      quantize,
      alpha,
      inverse,
      xyBlend,
    );
  }
  const ctx = tensor.ctx;
  const count = Math.max(0, Math.floor(Math.sqrt(h * w) * density));
  const x = randomUniform(count, 0, 1);
  const y = randomUniform(count, 0, 1);
  const xDir = randomNormalArray(count, 0, 1);
  const yDir = randomNormalArray(count, 0, 1);
  const inertia = randomNormalArray(count, 0.75, 0.25);

  const srcMaybe = tensor.read();
  const src =
    srcMaybe && typeof srcMaybe.then === "function"
      ? await srcMaybe
      : srcMaybe;
  const srcData =
    src instanceof Float32Array ? src : Float32Array.from(src || []);

  const blurTensor = maskValues(ValueMask.conv2d_blur)[0];
  const blurFlatMaybe = blurTensor.read();
  const blurFlat =
    blurFlatMaybe && typeof blurFlatMaybe.then === "function"
      ? await blurFlatMaybe
      : blurFlatMaybe;
  const [kh, kw] = maskShape(ValueMask.conv2d_blur);
  const blurKernel = [];
  for (let j = 0; j < kh; j++) {
    const row = [];
    for (let i = 0; i < kw; i++) {
      row.push(blurFlat[j * kw + i]);
    }
    blurKernel.push(row);
  }

  const convolved = await convolution(tensor, blurKernel);
  let valueTensor = await toValueMap(convolved);
  const valueShape = [h, w, 1];
  if (valueTensor.shape[0] !== h || valueTensor.shape[1] !== w) {
    valueTensor = await resample(valueTensor, valueShape);
  }
  valueTensor = await normalize(valueTensor);
  const valuesMaybe = valueTensor.read();
  const valuesRaw =
    valuesMaybe && typeof valuesMaybe.then === "function"
      ? await valuesMaybe
      : valuesMaybe;
  const valuesData =
    valuesRaw instanceof Float32Array
      ? valuesRaw
      : new Float32Array(valuesRaw ?? []);

  const widthScale = w - 1;
  const heightScale = h - 1;
  for (let i = 0; i < count; i++) {
    x[i] = Math.fround(x[i] * widthScale);
    y[i] = Math.fround(y[i] * heightScale);
    const len = Math.hypot(xDir[i], yDir[i]);
    xDir[i] = Math.fround(xDir[i] / len);
    yDir[i] = Math.fround(yDir[i] / len);
  }

  const startColors = new Float32Array(count * c);
  for (let i = 0; i < count; i++) {
    const xi = ((Math.floor(x[i]) % w) + w) % w;
    const yi = ((Math.floor(y[i]) % h) + h) % h;
    const base = (yi * w + xi) * c;
    for (let k = 0; k < c; k++) {
      startColors[i * c + k] = srcData[base + k];
    }
  }

  const wrap = (value, max) => {
    const mod = value % max;
    return mod < 0 ? mod + max : mod;
  };
  const blendScalar = (a, b, t) =>
    Math.fround(a * (1 - t) + b * t);

  const out = new Float32Array(h * w * c);
  for (let iter = 0; iter < iterations; iter++) {
    const exposure = Math.fround(
      1 - Math.abs(1 - (iter / (iterations - 1)) * 2),
    );
    for (let j = 0; j < count; j++) {
      const baseXi = wrap(Math.floor(x[j]), w);
      const baseYi = wrap(Math.floor(y[j]), h);
      const idx = baseYi * w + baseXi;
      const base = idx * c;
      for (let k = 0; k < c; k++) {
        const accum = out[base + k] || 0;
        out[base + k] = Math.fround(
          accum + startColors[j * c + k] * exposure,
        );
      }

      if (!valuesData || valuesData.length === 0) {
        continue;
      }

      const x1 = (baseXi + 1) % w;
      const y1 = (baseYi + 1) % h;
      const sv = valuesData[idx];
      const x1v = valuesData[baseYi * w + x1];
      const y1v = valuesData[y1 * w + baseXi];
      const x1y1v = valuesData[y1 * w + x1];
      const floorX = Math.floor(x[j]);
      const floorY = Math.floor(y[j]);
      const u = Math.fround(x[j] - floorX);
      const v = Math.fround(y[j] - floorY);
      const gX = Math.fround(
        blendScalar(y1v - sv, x1y1v - x1v, u),
      );
      const gY = Math.fround(
        blendScalar(x1v - sv, x1y1v - y1v, v),
      );
      const gx = quantize ? Math.floor(gX) : gX;
      const gy = quantize ? Math.floor(gY) : gY;
      const lenRaw = Math.fround(
        distance(gx, gy, DistanceMetric.euclidean) * contraction,
      );
      const len = lenRaw;
      const inertiaVal = inertia[j];
      const targetX = Math.fround(gx / len);
      const targetY = Math.fround(gy / len);
      xDir[j] = blendScalar(xDir[j], targetX, inertiaVal);
      yDir[j] = blendScalar(yDir[j], targetY, inertiaVal);
      x[j] = wrap(Math.fround(x[j] + xDir[j]), w);
      y[j] = wrap(Math.fround(y[j] + yDir[j]), h);
    }
  }

  let outTensor = Tensor.fromArray(ctx, out, shape);
  outTensor = await clamp01(outTensor);
  if (inverse) {
    const data = await outTensor.read();
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.fround(1 - data[i]);
    }
    outTensor = Tensor.fromArray(ctx, data, shape);
  }

  const xyAmount =
    typeof xyBlend === "boolean" ? (xyBlend ? 1 : 0) : xyBlend || 0;
  if (xyAmount) {
    const maskData = new Float32Array(h * w);
    for (let i = 0; i < h * w; i++) {
      maskData[i] = Math.fround((valuesData?.[i] || 0) * xyAmount);
    }
    const mask = Tensor.fromArray(ctx, maskData, [h, w, 1]);
    const shaded = await shadow(tensor, shape, time, speed);
    const reindexed = await reindex(tensor, shape, time, speed, 1);
    tensor = await blend(shaded, reindexed, mask);
  }

  return await blend(tensor, outTensor, alpha);
}
register("erosion_worms", erosionWorms, {
  density: 50,
  iterations: 50,
  contraction: 1.0,
  quantize: false,
  alpha: 0.25,
  inverse: false,
  xyBlend: 0,
});

async function wormsWebGPU(
  tensor,
  shape,
  time = 0,
  speed = 1,
  behavior = 1,
  density = 4.0,
  duration = 4.0,
  stride = 1.0,
  strideDeviation = 0.05,
  alpha = 0.5,
  kink = 1.0,
  drunkenness = 0.0,
  quantize = false,
  colors = null,
) {
  const ctx = tensor ? tensor.ctx : null;
  if (
    !ctx ||
    !ctx.device ||
    (quantize && shape[2] > 1) ||
    behavior !== WormBehavior.obedient
  ) {
    return await wormsCPU(
      tensor,
      shape,
      time,
      speed,
      behavior,
      density,
      duration,
      stride,
      strideDeviation,
      alpha,
      kink,
      drunkenness,
      quantize,
      colors,
    );
  }
  const [h, w, c] = shape;
  const { x, y, stride: strideVals, rot } = wormsParams(
    shape,
    density,
    stride,
    strideDeviation,
    behavior,
    time,
    speed,
  );
  const count = x.length;
  if (count === 0) return tensor;
  const posArr = new Float32Array(count * 2);
  for (let i = 0; i < count; i++) {
    posArr[i * 2] = x[i];
    posArr[i * 2 + 1] = y[i];
  }
  const strideArr = new Float32Array(strideVals);
  const rotArr = new Float32Array(rot);
  const src = colors
    ? colors.read
      ? await colors.read()
      : colors
    : await tensor.read();
  const wormColors = new Float32Array(count * c);
  for (let i = 0; i < count; i++) {
    const xi = Math.floor(x[i]);
    const yi = Math.floor(y[i]);
    const base = (yi * w + xi) * c;
    for (let k = 0; k < c; k++) {
      wormColors[i * c + k] = src[base + k];
    }
  }
  let valueData;
  if (c === 1) {
    valueData = await tensor.read();
  } else if (c === 2) {
    const srcVals = await tensor.read();
    valueData = new Float32Array(h * w);
    for (let i = 0; i < h * w; i++) valueData[i] = srcVals[i * 2];
  } else {
    let rgbTensor;
    const clamped = await clamp01(tensor);
    if (c === 3) {
      rgbTensor = clamped;
    } else {
      const srcVals = await clamped.read();
      const rgbVals = new Float32Array(h * w * 3);
      for (let i = 0; i < h * w; i++) {
        const base = i * c;
        rgbVals[i * 3] = srcVals[base];
        rgbVals[i * 3 + 1] = srcVals[base + 1];
        rgbVals[i * 3 + 2] = srcVals[base + 2];
      }
      rgbTensor = Tensor.fromArray(tensor.ctx, rgbVals, [h, w, 3]);
    }
    const labTensor = await rgbToOklab(rgbTensor);
    const labData = await labTensor.read();
    valueData = new Float32Array(h * w);
    for (let i = 0; i < h * w; i++) valueData[i] = labData[i * 3];
  }
  const indexArr = new Float32Array(h * w);
  for (let i = 0; i < h * w; i++) indexArr[i] = valueData[i] * TAU;
  const iterations = Math.floor(Math.sqrt(Math.min(w, h)) * duration);
  const drunkArr = new Float32Array(iterations * count);
  if (drunkenness) {
    for (let iter = 0; iter < iterations; iter++) {
      const start = Math.floor(
        Math.min(h, w) * time * speed + iter * speed * 10,
      );
      for (let i = 0; i < count; i++) {
        drunkArr[iter * count + i] =
          (periodicValue(start, random()) * 2 - 1) * Math.PI;
      }
    }
  }
  const posBuf = ctx.createGPUBuffer(posArr, GPUBufferUsage.STORAGE);
  const strideBuf = ctx.createGPUBuffer(strideArr, GPUBufferUsage.STORAGE);
  const rotBuf = ctx.createGPUBuffer(rotArr, GPUBufferUsage.STORAGE);
  const colorBuf = ctx.createGPUBuffer(wormColors, GPUBufferUsage.STORAGE);
  const indexBuf = ctx.createGPUBuffer(indexArr, GPUBufferUsage.STORAGE);
  const drunkBuf = ctx.createGPUBuffer(drunkArr, GPUBufferUsage.STORAGE);
  const outBuf = ctx.device.createBuffer({
    size: h * w * c * 4,
    usage:
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const paramsBuf = ctx.createGPUBuffer(
    new Float32Array([
      w,
      h,
      c,
      count,
      iterations,
      quantize ? 1 : 0,
      kink,
      drunkenness,
    ]),
    GPUBufferUsage.UNIFORM,
  );
  try {
    await ctx.runCompute(
      WORMS_WGSL,
      [
        { binding: 0, resource: { buffer: posBuf } },
        { binding: 1, resource: { buffer: strideBuf } },
        { binding: 2, resource: { buffer: rotBuf } },
        { binding: 3, resource: { buffer: colorBuf } },
        { binding: 4, resource: { buffer: indexBuf } },
        { binding: 5, resource: { buffer: drunkBuf } },
        { binding: 6, resource: { buffer: outBuf } },
        { binding: 7, resource: { buffer: paramsBuf } },
      ],
      Math.ceil(count / 64),
    );
  } catch (e) {
    return await wormsCPU(
      tensor,
      shape,
      time,
      speed,
      behavior,
      density,
      duration,
      stride,
      strideDeviation,
      alpha,
      kink,
      drunkenness,
      quantize,
      colors,
    );
  }
  const out = await ctx.readGPUBuffer(outBuf, h * w * c * 4);
  let outTensor = Tensor.fromArray(ctx, out, shape);
  outTensor = await normalize(outTensor);
  const d = await outTensor.read();
  for (let i = 0; i < d.length; i++) d[i] = Math.sqrt(d[i]);
  outTensor = Tensor.fromArray(ctx, d, shape);
  return blend(tensor, outTensor, alpha);
}

async function wormsCPU(
  tensor,
  shape,
  time = 0,
  speed = 1,
  behavior = 1,
  density = 4.0,
  duration = 4.0,
  stride = 1.0,
  strideDeviation = 0.05,
  alpha = 0.5,
  kink = 1.0,
  drunkenness = 0.0,
  quantize = false,
  colors = null,
) {
  tensor = await tensor;
  colors = colors ? await colors : null;
  const [h, w, c] = shape;
  const count = Math.floor(Math.max(w, h) * density);
  const wormsY = randomUniform(count, 0, 1);
  const wormsX = randomUniform(count, 0, 1);
  const wormsStride = randomNormalArray(count, stride, strideDeviation);
  const strideScale = Math.max(w, h) / 1024.0;
  for (let i = 0; i < count; i++) {
    wormsY[i] = Math.fround(wormsY[i] * (h - 1));
    wormsX[i] = Math.fround(wormsX[i] * (w - 1));
    wormsStride[i] = Math.fround(wormsStride[i] * strideScale);
  }
  const src = colors
    ? colors.read
      ? await colors.read()
      : colors
    : await tensor.read();
  const wormColors = new Float32Array(count * c);
  for (let i = 0; i < count; i++) {
    const xi = Math.floor(wormsX[i]);
    const yi = Math.floor(wormsY[i]);
    const base = (yi * w + xi) * c;
    for (let k = 0; k < c; k++) {
      wormColors[i * c + k] = src[base + k];
    }
  }
  const wormsRot = makeRots(behavior, count, time, speed);

  const valueTensor = await toValueMap(tensor);
  const valueDataMaybe = valueTensor.read();
  const valueData =
    valueDataMaybe && typeof valueDataMaybe.then === "function"
      ? await valueDataMaybe
      : valueDataMaybe;
  const valueChannels = valueTensor.shape[2] || 1;
  const indexArr = new Float32Array(h * w);
  for (let i = 0; i < h * w; i++) {
    indexArr[i] = Math.fround(valueData[i * valueChannels] * TAU * kink);
  }
  const iterations = Math.floor(Math.sqrt(Math.min(w, h)) * duration);
  const out = new Float32Array(h * w * c);
  for (let iter = 0; iter < iterations; iter++) {
    if (drunkenness) {
      const start = Math.floor(
        Math.min(h, w) * time * speed + iter * speed * 10,
      );
      for (let i = 0; i < count; i++) {
        wormsRot[i] +=
          (periodicValue(start, random()) * 2 - 1) * drunkenness * Math.PI;
      }
    }
    const exposure =
      iterations > 1 ? 1 - Math.abs(1 - (iter / (iterations - 1)) * 2) : 1;
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
  outTensor = await normalize(outTensor);
  const d = await outTensor.read();
  for (let i = 0; i < d.length; i++) d[i] = Math.sqrt(d[i]);
  outTensor = Tensor.fromArray(outTensor.ctx, d, shape);
  return blend(tensor, outTensor, alpha);
}
export async function worms(
  tensor,
  shape,
  time = 0,
  speed = 1,
  behavior = 1,
  density = 4.0,
  duration = 4.0,
  stride = 1.0,
  strideDeviation = 0.05,
  alpha = 0.5,
  kink = 1.0,
  drunkenness = 0.0,
  quantize = false,
  colors = null,
) {
  if (tensor && tensor.ctx && tensor.ctx.device) {
    return await wormsWebGPU(
      tensor,
      shape,
      time,
      speed,
      behavior,
      density,
      duration,
      stride,
      strideDeviation,
      alpha,
      kink,
      drunkenness,
      quantize,
      colors,
    );
  }
  return await wormsCPU(
    tensor,
    shape,
    time,
    speed,
    behavior,
    density,
    duration,
    stride,
    strideDeviation,
    alpha,
    kink,
    drunkenness,
    quantize,
    colors,
  );
}
register("worms", worms, {
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

export async function wormhole(
  tensor,
  shape,
  time,
  speed,
  kink = 1.0,
  inputStride = 1.0,
  alpha = 1.0,
) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const valueTensorMaybe = toValueMap(tensor);
  const valueTensor =
    valueTensorMaybe && typeof valueTensorMaybe.then === "function"
      ? await valueTensorMaybe
      : valueTensorMaybe;
  const valueDataMaybe = valueTensor.read();
  const valueRaw =
    valueDataMaybe && typeof valueDataMaybe.then === "function"
      ? await valueDataMaybe
      : valueDataMaybe;
  const valueChannels = valueTensor.shape[2] || 1;
  const valuesArr = new Float32Array(h * w);
  if (valueChannels === 1) {
    if (valueRaw instanceof Float32Array) {
      valuesArr.set(valueRaw);
    } else {
      for (let i = 0; i < h * w; i++) valuesArr[i] = valueRaw[i] ?? 0;
    }
  } else {
    for (let i = 0; i < h * w; i++) {
      valuesArr[i] = valueRaw[i * valueChannels] ?? 0;
    }
  }
  const stride = 1024 * inputStride;
  if (ctx && ctx.device && tensor.handle instanceof GPUTexture) {
    const yOff = Math.floor(h * 0.5 + random() * h * 0.5);
    const xOff = Math.floor(random() * w * 0.5);
    const lumBuf = ctx.createGPUBuffer(valuesArr, GPUBufferUsage.STORAGE);
    const outBuf = ctx.device.createBuffer({
      size: h * w * c * 4,
      usage:
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const paramsBuf = ctx.createGPUBuffer(
      new Float32Array([w, h, c, stride, kink, xOff, yOff, 0]),
      GPUBufferUsage.UNIFORM,
    );
    try {
      await ctx.runCompute(
        WORMHOLE_WGSL,
        [
          { binding: 0, resource: tensor.handle.createView() },
          { binding: 1, resource: { buffer: lumBuf } },
          { binding: 2, resource: { buffer: outBuf } },
          { binding: 3, resource: { buffer: paramsBuf } },
        ],
        Math.ceil(w / 8),
        Math.ceil(h / 8),
      );
    } catch (e) {
      const cpuTensor = Tensor.fromArray(null, await tensor.read(), shape);
      return wormhole(cpuTensor, shape, time, speed, kink, inputStride, alpha);
    }
    const out = await ctx.readGPUBuffer(outBuf, h * w * c * 4);
    let outTensor = Tensor.fromArray(ctx, out, shape);
    outTensor = await normalize(outTensor);
    const d = await outTensor.read();
    for (let i = 0; i < d.length; i++) d[i] = Math.sqrt(d[i]);
    outTensor = Tensor.fromArray(ctx, d, shape);
    return await blend(tensor, outTensor, alpha);
  }
  const srcMaybe = tensor.read();
  const src =
    srcMaybe && typeof srcMaybe.then === "function"
      ? await srcMaybe
      : srcMaybe;
  const xArr = new Int32Array(h * w);
  const yArr = new Int32Array(h * w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const deg = valuesArr[idx] * TAU * kink;
      const xo = (Math.cos(deg) + 1) * stride;
      const yo = (Math.sin(deg) + 1) * stride;
      xArr[idx] = Math.floor(x + xo) % w;
      yArr[idx] = Math.floor(y + yo) % h;
    }
  }
  const offs = offsetIndexInternal(yArr, h, xArr, w);
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
      out[i] = (out[i] - min) / range;
    }
  }
  for (let i = 0; i < out.length; i++) {
    out[i] = Math.sqrt(out[i]);
  }
  const outTensor = Tensor.fromArray(tensor.ctx, out, shape);
  return await blend(tensor, outTensor, alpha);
}
register("wormhole", wormhole, { kink: 1.0, inputStride: 1.0, alpha: 1.0 });

export async function vignette(
  tensor,
  shape,
  time,
  speed,
  brightness = 0.0,
  alpha = 1.0,
) {
  const [h, w, c] = shape;
  const norm = await normalize(tensor);
  const ctx = tensor.ctx;
  if (ctx && ctx.device && norm.handle instanceof GPUTexture) {
    const outBuf = ctx.createGPUBuffer(
      new Float32Array(h * w * c),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    const paramsBuf = ctx.createGPUBuffer(
      new Float32Array([w, h, c, brightness, alpha, 0, 0, 0]),
      GPUBufferUsage.UNIFORM,
    );
    await ctx.runCompute(
      VIGNETTE_WGSL,
      [
        { binding: 0, resource: norm.handle.createView() },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: paramsBuf } },
      ],
      Math.ceil(w / 8),
      Math.ceil(h / 8),
    );
    return new Tensor(ctx, outBuf, [h, w, c]);
  }
  const edgeData = new Float32Array(h * w * c);
  edgeData.fill(brightness);
  const edges = Tensor.fromArray(tensor.ctx, edgeData, shape);
  const maskTensor = await singularity(
    null,
    [h, w, 1],
    time,
    speed,
    VoronoiDiagramType.range,
    DistanceMetric.euclidean,
  );
  const maskData = await maskTensor.read();
  for (let i = 0; i < maskData.length; i++) {
    maskData[i] = Math.pow(maskData[i], 2);
  }
  const maskT = Tensor.fromArray(tensor.ctx, maskData, [h, w, 1]);
  const vignetted = await blend(norm, edges, maskT);
  return await blend(norm, vignetted, alpha);
}
register("vignette", vignette, { brightness: 0.0, alpha: 1.0 });

async function vaselineWebGPU(tensor, shape, alpha) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const format = c === 1 ? 'r32float' : c === 2 ? 'rg32float' : 'rgba32float';
  const blurTex = ctx.device.createTexture({
    size: { width: w, height: h, depthOrArrayLayers: 1 },
    format,
    usage: GPUTextureUsage.STORAGE_BINDING |
           GPUTextureUsage.TEXTURE_BINDING |
           GPUTextureUsage.COPY_SRC |
           GPUTextureUsage.COPY_DST,
  });
  const blurParams = ctx.createGPUBuffer(new Float32Array([w, h, c]), GPUBufferUsage.UNIFORM);
  const maskTex = ctx.device.createTexture({
    size: { width: w, height: h, depthOrArrayLayers: 1 },
    format: 'r32float',
    usage: GPUTextureUsage.STORAGE_BINDING |
           GPUTextureUsage.TEXTURE_BINDING |
           GPUTextureUsage.COPY_SRC |
           GPUTextureUsage.COPY_DST,
  });
  const maskParams = ctx.createGPUBuffer(new Float32Array([w, h]), GPUBufferUsage.UNIFORM);
  await ctx.withEncoder(async () => {
    const blurShader = VASELINE_BLUR_WGSL.replace('rgba32float', format);
    await ctx.runCompute(
      blurShader,
      [
        { binding: 0, resource: tensor.handle.createView() },
        { binding: 1, resource: blurTex.createView() },
        { binding: 2, resource: { buffer: blurParams } },
      ],
      Math.ceil(w / 8),
      Math.ceil(h / 8),
    );
    await ctx.runCompute(
      VASELINE_MASK_WGSL,
      [
        { binding: 0, resource: maskTex.createView() },
        { binding: 1, resource: { buffer: maskParams } },
      ],
      Math.ceil(w / 8),
      Math.ceil(h / 8),
    );
  });
  const blurred = new Tensor(ctx, blurTex, shape);
  const mask = new Tensor(ctx, maskTex, [h, w, 1]);
  const masked = await blend(tensor, blurred, mask);
  return await blend(tensor, masked, alpha);
}
export async function vaseline(tensor, shape, time, speed, alpha = 1.0) {
  const ctx = tensor.ctx;
  if (ctx && ctx.device) {
    return await vaselineWebGPU(tensor, shape, alpha);
  }
  const blurred = await bloom(tensor, shape, 0, 1, 1.0);
  const masked = await centerMaskInternal(tensor, blurred, shape);
  return await blend(tensor, masked, alpha);
}
register("vaseline", vaseline, { alpha: 1.0 });

export async function lightLeak(tensor, shape, time, speed, alpha = 0.25) {
  const gridMembers = [
    PointDistribution.square,
    PointDistribution.waffle,
    PointDistribution.chess,
    PointDistribution.h_hex,
    PointDistribution.v_hex,
  ];
  // randomInt() returns values inclusive of both endpoints. Clamp to
  // length - 1 so the selected index matches Python's rng.random_int
  // behavior and never exceeds the array bounds.
  const distrib = gridMembers[randomInt(0, gridMembers.length - 1)];
  const [xPts, yPts] = pointCloud(6, {
    distrib,
    drift: 0.05,
    shape,
    time,
    speed,
  });
  let leak = await voronoiColorRegions(tensor, shape, time, speed, xPts, yPts);
  leak = await wormhole(leak, shape, time, speed, 1.0, 0.25, 1.0);
  leak = await bloom(leak, shape, time, speed, 1.0);
  const src = await tensor.read();
  const leakData = await leak.read();
  const screened = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    screened[i] = 1 - (1 - src[i]) * (1 - leakData[i]);
  }
  leak = Tensor.fromArray(tensor.ctx, screened, shape);
  leak = await centerMaskInternal(
    tensor,
    leak,
    shape,
    undefined,
    DistanceMetric.octagram,
  );
  const blended = await blend(tensor, leak, alpha);
  // ``vaseline`` ignores the temporal parameters but expects them for
  // registry consistency.
  return await vaseline(blended, shape, 0, 1, alpha);
}
register("lightLeak", lightLeak, { alpha: 0.25 });
register("light_leak", lightLeak, { alpha: 0.25 });

export async function dither(tensor, shape, time, speed, levels = 2) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  if (ctx && ctx.device) {
    const noise = values(Math.max(h, w), [h, w, 1], {
      ctx,
      time,
      seed: 0,
      speed: speed * 1000,
    });
    if (tensor.handle instanceof GPUTexture && noise.handle instanceof GPUTexture) {
      const outBuf = ctx.createGPUBuffer(
        new Float32Array(h * w * c),
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      );
      const paramsBuf = ctx.createGPUBuffer(
        new Float32Array([w, h, c, levels, 0, 0, 0, 0]),
        GPUBufferUsage.UNIFORM,
      );
      await ctx.runCompute(
        DITHER_WGSL,
        [
          { binding: 0, resource: tensor.handle.createView() },
          { binding: 1, resource: noise.handle.createView() },
          { binding: 2, resource: { buffer: outBuf } },
          { binding: 3, resource: { buffer: paramsBuf } },
        ],
        Math.ceil(w / 8),
        Math.ceil(h / 8),
      );
      return new Tensor(ctx, outBuf, shape);
    }
  }
  const noise = values(Math.max(h, w), [h, w, 1], {
    ctx: tensor.ctx,
    time,
    seed: 0,
    speed: speed * 1000,
  });
  const n = noise.read();
  const src = await tensor.read();
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
register("dither", dither, { levels: 2 });

export function grain(tensor, shape, time, speed, alpha = 0.25) {
  const [h, w] = shape;
  const noise = values([h, w], [h, w, 1], {
    ctx: tensor.ctx,
    time,
    speed: speed * 100,
  });
  return blend(tensor, noise, alpha);
}
register("grain", grain, { alpha: 0.25 });

export async function snow(tensor, shape, time, speed, alpha = 0.25) {
  const [h, w] = shape;
  const ctx = tensor.ctx;
  const valueShape = [h, w, 1];

  const resolveTensor = async (maybeTensor) =>
    maybeTensor && typeof maybeTensor.then === "function"
      ? await maybeTensor
      : maybeTensor;

  let staticNoise = values([h, w], valueShape, {
    ctx,
    time,
    speed: speed * 100,
    splineOrder: InterpolationType.constant,
  });
  let limiter = values([h, w], valueShape, {
    ctx,
    time,
    speed: speed * 100,
    distrib: ValueDistribution.exp,
    splineOrder: InterpolationType.constant,
  });

  staticNoise = await resolveTensor(staticNoise);
  limiter = await resolveTensor(limiter);

  const limiterDataMaybe = limiter.read();
  const limiterData =
    limiterDataMaybe && typeof limiterDataMaybe.then === "function"
      ? await limiterDataMaybe
      : limiterDataMaybe;

  const scaled = new Float32Array(limiterData.length);

  if (typeof alpha === "number") {
    for (let i = 0; i < scaled.length; i++) {
      scaled[i] = limiterData[i] * alpha;
    }
    const mask = Tensor.fromArray(ctx, scaled, valueShape);
    return blend(tensor, staticNoise, mask);
  }

  let alphaTensor = alpha;
  if (alphaTensor && typeof alphaTensor.then === "function") {
    alphaTensor = await alphaTensor;
  }

  if (!alphaTensor || typeof alphaTensor.read !== "function") {
    const mask = Tensor.fromArray(ctx, limiterData.slice(), valueShape);
    return blend(tensor, staticNoise, mask);
  }

  const alphaDataMaybe = alphaTensor.read();
  const alphaData =
    alphaDataMaybe && typeof alphaDataMaybe.then === "function"
      ? await alphaDataMaybe
      : alphaDataMaybe;
  const alphaChannels = alphaTensor.shape?.[2] || 1;
  const pixelCount = alphaData ? alphaData.length / alphaChannels : 0;
  for (let i = 0; i < scaled.length; i++) {
    const idx = alphaData && pixelCount ? Math.min(i, pixelCount - 1) : i;
    const alphaVal =
      alphaData && pixelCount ? alphaData[idx * alphaChannels] : 0;
    scaled[i] = limiterData[i] * alphaVal;
  }
  const mask = Tensor.fromArray(ctx, scaled, valueShape);
  return blend(tensor, staticNoise, mask);
}
register("snow", snow, { alpha: 0.25 });

export function saturation(tensor, shape, time, speed, amount = 0.75) {
  if (shape[2] !== 3) return tensor;
  const ctx = tensor.ctx;
  const cpuSat = (t) => {
    const hsvMaybe = rgbToHsv(t);
    const process = (hsv) => {
      const dataMaybe = hsv.read();
      const adjust = (data) => {
        for (let i = 0; i < shape[0] * shape[1]; i++) {
          data[i * 3 + 1] = data[i * 3 + 1] * amount;
        }
        return hsvToRgb(Tensor.fromArray(t.ctx, data, hsv.shape));
      };
      if (dataMaybe && typeof dataMaybe.then === "function") {
        return dataMaybe.then(adjust);
      }
      return adjust(dataMaybe);
    };
    if (hsvMaybe && typeof hsvMaybe.then === "function") {
      return hsvMaybe.then(process);
    }
    return process(hsvMaybe);
  };

  if (ctx && ctx.device && tensor.handle instanceof GPUTexture) {
    return (async () => {
      try {
        const hsv = await rgbToHsv(tensor);
        const data = await hsv.read();
        for (let i = 0; i < shape[0] * shape[1]; i++) {
          data[i * 3 + 1] = data[i * 3 + 1] * amount;
        }
        return await hsvToRgb(Tensor.fromArray(ctx, data, hsv.shape));
      } catch (e) {
        console.warn('WebGPU saturation fallback to CPU', e);
        const data = await tensor.read();
        const cpuTensor = Tensor.fromArray(null, data, tensor.shape);
        return cpuSat(cpuTensor);
      }
    })();
  }
  return cpuSat(tensor);
}
register("saturation", saturation, { amount: 0.75 });
register("adjust_saturation", saturation, { amount: 0.75 });

// Provide snake_case aliases to mirror the Python API for parity tests.
export const adjust_saturation = saturation;

export function randomHue(tensor, shape, time, speed, range = 0.05) {
  const shift = random() * range * 2 - range;
  return adjustHue(tensor, shift);
}
register("random_hue", randomHue, { range: 0.05 });

export function normalizeEffect(tensor, shape, time, speed) {
  return normalize(tensor);
}
register("normalize", normalizeEffect, {});

export async function adjustBrightness(tensor, shape, time, speed, amount = 0.125) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  if (ctx && ctx.device && tensor.handle instanceof GPUTexture) {
    const outBuf = ctx.createGPUBuffer(
      new Float32Array(h * w * c),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    const paramsBuf = ctx.createGPUBuffer(
      new Float32Array([w, h, c, amount]),
      GPUBufferUsage.UNIFORM,
    );
    await ctx.runCompute(
      ADJUST_BRIGHTNESS_WGSL,
      [
        { binding: 0, resource: tensor.handle.createView() },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: paramsBuf } },
      ],
      Math.ceil(w / 8),
      Math.ceil(h / 8),
    );
    return new Tensor(ctx, outBuf, shape);
  }
  const src = await tensor.read();
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const v = src[i] + amount;
    out[i] = v < -1 ? -1 : v > 1 ? 1 : v;
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register("adjust_brightness", adjustBrightness, { amount: 0.125 });

export const adjust_brightness = adjustBrightness;

export async function adjustContrast(tensor, shape, time, speed, amount = 1.25) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const src = await tensor.read();
  const mean = new Float32Array(c);
  const pixelCount = h * w;
  for (let i = 0; i < pixelCount; i++) {
    for (let ch = 0; ch < c; ch++) {
      mean[ch] = Math.fround(mean[ch] + src[i * c + ch]);
    }
  }
  for (let ch = 0; ch < c; ch++) {
    mean[ch] = Math.fround(mean[ch] / pixelCount);
  }
  if (ctx && ctx.device && tensor.handle instanceof GPUTexture) {
    const outBuf = ctx.createGPUBuffer(
      new Float32Array(h * w * c),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    const paramsBuf = ctx.createGPUBuffer(
      new Float32Array([
        w,
        h,
        c,
        amount,
        mean[0],
        mean[1] || mean[0],
        mean[2] || mean[0],
        0,
      ]),
      GPUBufferUsage.UNIFORM,
    );
    await ctx.runCompute(
      ADJUST_CONTRAST_WGSL,
      [
        { binding: 0, resource: tensor.handle.createView() },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: paramsBuf } },
      ],
      Math.ceil(w / 8),
      Math.ceil(h / 8),
    );
    return new Tensor(ctx, outBuf, shape);
  }
  const out = new Float32Array(src.length);
  for (let i = 0; i < pixelCount; i++) {
    for (let ch = 0; ch < c; ch++) {
      const idx = i * c + ch;
      const v = Math.fround((src[idx] - mean[ch]) * amount + mean[ch]);
      out[idx] = v < 0 ? 0 : v > 1 ? 1 : v;
    }
  }
  return Tensor.fromArray(ctx, out, shape);
}
register("adjust_contrast", adjustContrast, { amount: 1.25 });

export const adjust_contrast = adjustContrast;

export function adjustHueEffect(tensor, shape, time, speed, amount = 0.25) {
  if (shape[2] !== 3 || amount === 0 || amount === 1 || amount === null)
    return tensor;
  return adjustHue(tensor, amount);
}
register("adjust_hue", adjustHueEffect, { amount: 0.25 });

export const adjust_hue = adjustHueEffect;

export function ridgeEffect(tensor, shape, time, speed) {
  return ridge(tensor);
}
register("ridge", ridgeEffect, {});

export async function sine(
  tensor,
  shape,
  time,
  speed,
  amount = 1.0,
  rgb = false,
  freq = 1,
  octaves = 1,
) {
  const [h, w, c] = shape;
  const src = await tensor.read();
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
  // freq and octaves currently unused; included for compatibility
}
register("sine", sine, { amount: 1.0, rgb: false, freq: 1, octaves: 1 });

export function blur(
  tensor,
  shape,
  time,
  speed,
  amount = 10.0,
  splineOrder = InterpolationType.bicubic,
) {
  const [h, w, c] = shape;
  const newShape = [
    Math.max(1, Math.floor(h / amount)),
    Math.max(1, Math.floor(w / amount)),
    c,
  ];
  let small = proportionalDownsample(tensor, shape, newShape);
  const process = (s) =>
    withTensorData(s, (data) => {
      const arr = data instanceof Float32Array ? data.slice() : new Float32Array(data);
      for (let i = 0; i < arr.length; i++) arr[i] *= 4;
      const sm = Tensor.fromArray(tensor.ctx, arr, s.shape);
      return resample(sm, shape, splineOrder);
    });
  if (small && typeof small.then === 'function') {
    return small.then(process);
  }
  return process(small);
}
register("blur", blur, {
  amount: 10.0,
  splineOrder: InterpolationType.bicubic,
});

export async function wobble(tensor, shape, time, speed) {
  const xOffset = Math.floor(
    simplexRandom(time, undefined, speed * 0.5) * shape[1],
  );
  const yOffset = Math.floor(
    simplexRandom(time, undefined, speed * 0.5) * shape[0],
  );
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  if (ctx && ctx.device && tensor.handle instanceof GPUTexture) {
    const outBuf = ctx.createGPUBuffer(
      new Float32Array(h * w * c),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    const paramsBuf = ctx.createGPUBuffer(
      new Float32Array([w, h, c, xOffset, yOffset, 0, 0, 0]),
      GPUBufferUsage.UNIFORM,
    );
    await ctx.runCompute(
      WOBBLE_WGSL,
      [
        { binding: 0, resource: tensor.handle.createView() },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: paramsBuf } },
      ],
      Math.ceil(w / 8),
      Math.ceil(h / 8),
    );
    return new Tensor(ctx, outBuf, shape);
  }
  return await offsetTensor(tensor, xOffset, yOffset);
}
register("wobble", wobble, {});

export async function reverb(
  tensor,
  shape,
  time,
  speed,
  octaves = 2,
  iterations = 1,
  ridges = true,
) {
  if (!octaves) return tensor;
  const [h, w, c] = shape;
  const reference = ridges ? await ridge(tensor) : tensor;
  const base = await reference.read();
  const ctx = tensor.ctx;
  if (ctx && ctx.device && tensor.handle instanceof GPUTexture) {
    const outBuf = ctx.createGPUBuffer(
      base,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    if (iterations > 0) {
      await ctx.withEncoder(async () => {
        for (let i = 0; i < iterations; i++) {
          for (let octave = 1; octave <= octaves; octave++) {
            const mult = 2 ** octave;
            const nh = Math.floor(h / mult) || 1;
            const nw = Math.floor(w / mult) || 1;
            if (nh === 0 || nw === 0) break;
            const layer = await downsample(reference, mult);
            const paramsBuf = ctx.createGPUBuffer(
              new Float32Array([w, h, nw, nh, c, 1 / mult, 0, 0]),
              GPUBufferUsage.UNIFORM,
            );
            await ctx.runCompute(
              REVERB_WGSL,
              [
                { binding: 0, resource: layer.handle.createView() },
                { binding: 1, resource: { buffer: outBuf } },
                { binding: 2, resource: { buffer: paramsBuf } },
              ],
              Math.ceil(w / 8),
              Math.ceil(h / 8),
            );
          }
        }
      });
    }
    const outData = await ctx.readGPUBuffer(outBuf, h * w * c * 4);
    const outTensor = Tensor.fromArray(ctx, outData, shape);
    return normalize(outTensor);
  }
  const outData = base.slice();
  for (let i = 0; i < iterations; i++) {
    for (let octave = 1; octave <= octaves; octave++) {
      const mult = 2 ** octave;
      const nh = Math.floor(h / mult) || 1;
      const nw = Math.floor(w / mult) || 1;
      if (nh === 0 || nw === 0) break;
      const octaveShape = [nh, nw, c];
      let layer = proportionalDownsample(reference, shape, octaveShape);
      layer = await expandTileInternal(layer, octaveShape, shape);
      const layerData = await layer.read();
      for (let j = 0; j < outData.length; j++) {
        outData[j] += layerData[j] / mult;
      }
    }
  }
  const outTensor = Tensor.fromArray(tensor.ctx, outData, shape);
  return normalize(outTensor);
}
register("reverb", reverb, { octaves: 2, iterations: 1, ridges: true });

async function expandTileInternal(tensor, inputShape, outputShape) {
  const [ih, iw, c] = inputShape;
  const [oh, ow] = outputShape;
  const t = tensor && typeof tensor.then === 'function' ? await tensor : tensor;
  const src = await t.read();
  const out = new Float32Array(oh * ow * c);
  const xOff = Math.floor((ow - iw) / 2);
  const yOff = Math.floor((oh - ih) / 2);
  for (let y = 0; y < oh; y++) {
    const sy = ((y - yOff) % ih + ih) % ih;
    for (let x = 0; x < ow; x++) {
      const sx = ((x - xOff) % iw + iw) % iw;
      const srcBase = (sy * iw + sx) * c;
      const dstBase = (y * ow + x) * c;
      for (let k = 0; k < c; k++) out[dstBase + k] = src[srcBase + k];
    }
  }
  return Tensor.fromArray(t.ctx, out, [oh, ow, c]);
}

async function resizeWithCropOrPad(tensor, shape, size) {
  const [h, w, c] = shape;
  const src = await tensor.read();
  const out = new Float32Array(size * size * c);
  const yOff = Math.floor((size - h) / 2);
  const xOff = Math.floor((size - w) / 2);
  for (let y = 0; y < size; y++) {
    const sy = y - yOff;
    if (sy < 0 || sy >= h) continue;
    for (let x = 0; x < size; x++) {
      const sx = x - xOff;
      if (sx < 0 || sx >= w) continue;
      const srcBase = (sy * w + sx) * c;
      const dstBase = (y * size + x) * c;
      for (let k = 0; k < c; k++) out[dstBase + k] = src[srcBase + k];
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [size, size, c]);
}

async function rotate2D(tensor, shape, angle) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  if (ctx && ctx.device && tensor.handle instanceof GPUTexture) {
    const outBuf = ctx.createGPUBuffer(
      new Float32Array(h * w * c),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    const paramsBuf = ctx.createGPUBuffer(
      new Float32Array([w, h, c, angle, 0, 0, 0, 0]),
      GPUBufferUsage.UNIFORM,
    );
    await ctx.runCompute(
      ROTATE_WGSL,
      [
        { binding: 0, resource: tensor.handle.createView() },
        { binding: 1, resource: { buffer: outBuf } },
        { binding: 2, resource: { buffer: paramsBuf } },
      ],
      Math.ceil(w / 8),
      Math.ceil(h / 8),
    );
    return new Tensor(ctx, outBuf, [h, w, c]);
  }
  const src = await tensor.read();
  const out = new Float32Array(h * w * c);
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = x / w - 0.5;
      const dy = y / h - 0.5;
      const sx = Math.floor((cos * dx + sin * dy + 0.5) * w);
      const sy = Math.floor((-sin * dx + cos * dy + 0.5) * h);
      const sxw = ((sx % w) + w) % w;
      const syw = ((sy % h) + h) % h;
      const srcBase = (syw * w + sxw) * c;
      const dstBase = (y * w + x) * c;
      for (let k = 0; k < c; k++) out[dstBase + k] = src[srcBase + k];
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
}

async function cropTensor(tensor, inputShape, outputShape) {
  const [H, W, c] = inputShape;
  const [h, w] = outputShape;
  const src = await tensor.read();
  const out = new Float32Array(h * w * c);
  const yOff = Math.floor((H - h) / 2);
  const xOff = Math.floor((W - w) / 2);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const srcBase = ((yOff + y) * W + (xOff + x)) * c;
      const dstBase = (y * w + x) * c;
      for (let k = 0; k < c; k++) out[dstBase + k] = src[srcBase + k];
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
}

async function resizeBilinear(tensor, size) {
  const [h, w, c] = tensor.shape;
  const src = await tensor.read();
  const out = new Float32Array(size * size * c);
  for (let y = 0; y < size; y++) {
    const sy = (y + 0.5) * h / size - 0.5;
    const y0 = Math.max(0, Math.floor(sy));
    const y1 = Math.min(h - 1, y0 + 1);
    const wy = sy - y0;
    for (let x = 0; x < size; x++) {
      const sx = (x + 0.5) * w / size - 0.5;
      const x0 = Math.max(0, Math.floor(sx));
      const x1 = Math.min(w - 1, x0 + 1);
      const wx = sx - x0;
      const dstBase = (y * size + x) * c;
      for (let k = 0; k < c; k++) {
        const v00 = src[(y0 * w + x0) * c + k];
        const v01 = src[(y0 * w + x1) * c + k];
        const v10 = src[(y1 * w + x0) * c + k];
        const v11 = src[(y1 * w + x1) * c + k];
        const v0 = v00 * (1 - wx) + v01 * wx;
        const v1 = v10 * (1 - wx) + v11 * wx;
        out[dstBase + k] = v0 * (1 - wy) + v1 * wy;
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [size, size, c]);
}

export async function squareCropAndResize(tensor, shape, length = 1024) {
  const [h, w, c] = shape;
  const have = Math.min(h, w);
  let out = tensor;
  if (h !== w) {
    out = await cropTensor(out, shape, [have, have]);
  }
  if (have !== length) {
    out = await resample(out, [length, length, c]);
  }
  return out;
}

export async function rotate(tensor, shape, time, speed, angle = null) {
  if (angle === null || angle === undefined) angle = random() * 360;
  const [h, w, c] = shape;
  const want = Math.max(h, w) * 2;
  let padded = await expandTileInternal(tensor, shape, [want, want, c]);
  padded = await rotate2D(padded, [want, want, c], (angle * Math.PI) / 180);
  return await cropTensor(padded, [want, want, c], shape);
}
register("rotate", rotate, { angle: 0 });

async function _pixelSort(tensor, shape, angle, darkest) {
  const ctx = tensor.ctx;
  const [h, w, c] = shape;
  const srcMaybe = tensor.read();
  const srcData =
    srcMaybe && typeof srcMaybe.then === "function"
      ? await srcMaybe
      : srcMaybe;
  let baseData;
  if (srcData instanceof Float32Array) {
    baseData = srcData.slice();
  } else if (srcData) {
    baseData = Float32Array.from(srcData);
  } else {
    baseData = new Float32Array(h * w * c);
  }
  if (darkest) {
    for (let i = 0; i < baseData.length; i++) {
      baseData[i] = 1 - baseData[i];
    }
  }
  const baseTensor = Tensor.fromArray(ctx, baseData, shape);
  let working = baseTensor;
  const want = Math.max(h, w) * 2;
  working = await resizeWithCropOrPad(working, shape, want);
  if (angle !== false) {
    working = await rotate2D(working, [want, want, c], (angle * Math.PI) / 180);
  }
  const data = await working.read();
  const finalizeOutput = async (tensorOut) => {
    const sortedMaybe = tensorOut.read();
    const sortedData =
      sortedMaybe && typeof sortedMaybe.then === "function"
        ? await sortedMaybe
        : sortedMaybe;
    let sortedArr;
    if (sortedData instanceof Float32Array) {
      sortedArr = sortedData;
    } else if (sortedData) {
      sortedArr = Float32Array.from(sortedData);
    } else {
      sortedArr = new Float32Array(h * w * c);
    }
    const out = new Float32Array(h * w * c);
    for (let i = 0; i < out.length; i++) {
      const baseVal = baseData[i] ?? 0;
      const sortedVal = sortedArr[i] ?? 0;
      const maxVal = Math.max(baseVal, sortedVal);
      out[i] = darkest ? 1 - maxVal : maxVal;
    }
    return Tensor.fromArray(ctx, out, shape);
  };
  if (!ctx || !ctx.device) {
    const valueTensor = await toValueMap(working);
    const valuesData = await valueTensor.read();
    const sorted = new Float32Array(want * want * c);
    for (let y = 0; y < want; y++) {
      const rowOffset = y * want;
      const indices = new Array(want);
      for (let i = 0; i < want; i++) indices[i] = i;
      indices.sort((a, b) => valuesData[rowOffset + b] - valuesData[rowOffset + a]);
      const sortedRow = new Float32Array(want * c);
      for (let pos = 0; pos < want; pos++) {
        const srcIndex = indices[pos];
        const srcBase = (rowOffset + srcIndex) * c;
        const dstBase = pos * c;
        for (let k = 0; k < c; k++) {
          sortedRow[dstBase + k] = data[srcBase + k];
        }
      }
      const shift = indices[0] || 0;
      for (let x = 0; x < want; x++) {
        const srcPos = (x - shift + want) % want;
        const dstBase = (rowOffset + x) * c;
        const srcBase = srcPos * c;
        for (let k = 0; k < c; k++) {
          sorted[dstBase + k] = sortedRow[srcBase + k];
        }
      }
    }
    let sortedTensor = Tensor.fromArray(ctx, sorted, [want, want, c]);
    if (angle !== false) {
      sortedTensor = await rotate2D(
        sortedTensor,
        [want, want, c],
        (-angle * Math.PI) / 180,
      );
    }
    sortedTensor = await cropTensor(sortedTensor, [want, want, c], shape);
    return finalizeOutput(sortedTensor);
  }
  const sorted = new Float32Array(want * want * c);
  const rowBuffers = [];
  await ctx.withEncoder(async () => {
    for (let y = 0; y < want; y++) {
      const row = data.subarray(y * want * c, (y + 1) * want * c);
      const srcBuf = ctx.createGPUBuffer(row, GPUBufferUsage.STORAGE);
      const outBuf = ctx.createGPUBuffer(
        new Float32Array(want * c),
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      );
      const paramsBuf = ctx.createGPUBuffer(
        new Uint32Array([want, c, darkest ? 1 : 0]),
        GPUBufferUsage.UNIFORM,
      );
      await ctx.runCompute(
        PIXEL_SORT_WGSL,
        [
          { binding: 0, resource: { buffer: srcBuf } },
          { binding: 1, resource: { buffer: outBuf } },
          { binding: 2, resource: { buffer: paramsBuf } },
        ],
        1,
      );
      rowBuffers.push({ outBuf, srcBuf, paramsBuf });
    }
  });
  for (let y = 0; y < rowBuffers.length; y++) {
    const { outBuf, srcBuf, paramsBuf } = rowBuffers[y];
    const outRow = await ctx.readGPUBuffer(outBuf, want * c * 4);
    sorted.set(outRow, y * want * c);
    if (typeof srcBuf.destroy === "function") srcBuf.destroy();
    if (typeof paramsBuf.destroy === "function") paramsBuf.destroy();
    if (typeof outBuf.destroy === "function") outBuf.destroy();
  }
  let sortedTensor = Tensor.fromArray(ctx, sorted, [want, want, c]);
  if (angle !== false) {
    sortedTensor = await rotate2D(
      sortedTensor,
      [want, want, c],
      (-angle * Math.PI) / 180,
    );
  }
  sortedTensor = await cropTensor(sortedTensor, [want, want, c], shape);
  return finalizeOutput(sortedTensor);
}

export async function pixelSort(
  tensor,
  shape,
  time,
  speed,
  angled = false,
  darkest = false,
) {
  let angle = false;
  if (angled) angle = angled === true ? random() * 360 : angled;
  return _pixelSort(tensor, shape, angle, darkest);
}
register("pixel_sort", pixelSort, { angled: false, darkest: false });
register("pixel_sort", pixelSort, { angled: false, darkest: false });

export async function glyphMap(
  tensor,
  shape,
  time,
  speed,
  mask = ValueMask.truetype,
  colorize = true,
  zoom = 1,
  alpha = 1,
  splineOrder = InterpolationType.constant,
) {
  if (mask === null || mask === undefined) mask = ValueMask.truetype;
  const ctx = tensor.ctx;
  if (
    ctx &&
    ctx.device &&
    tensor.handle instanceof GPUTexture &&
    mask === ValueMask.truetype
  ) {
    const glyphShape = [15, 15, 1];
    const glyphs = loadGlyphs(glyphShape) || [];
    if (!glyphs.length) return tensor;
    const [h, w, c] = shape;
    const gh = glyphShape[0];
    const gw = glyphShape[1];
    const gCount = glyphs.length;
    const atlasData = new Float32Array(gw * gh * gCount * 4);
    for (let gi = 0; gi < gCount; gi++) {
      const glyph = glyphs[gi];
      for (let gy = 0; gy < gh; gy++) {
        for (let gx = 0; gx < gw; gx++) {
          atlasData[((gi * gh + gy) * gw + gx) * 4] = glyph[gy][gx][0];
        }
      }
    }
    const glyphTex = ctx.createTexture(gw, gh * gCount, atlasData);
    const outBuf = ctx.createGPUBuffer(
      new Float32Array(h * w * c),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    const paramsBuf = ctx.createGPUBuffer(
      new Float32Array([
        w,
        h,
        c,
        gw,
        gh,
        gCount,
        colorize ? 1 : 0,
        zoom,
        alpha,
        0,
        0,
        0,
      ]),
      GPUBufferUsage.UNIFORM,
    );
    await ctx.runCompute(
      GLYPH_MAP_WGSL,
      [
        { binding: 0, resource: tensor.handle.createView() },
        { binding: 1, resource: glyphTex.createView() },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: paramsBuf } },
      ],
      Math.ceil(w / 8),
      Math.ceil(h / 8),
    );
    return new Tensor(ctx, outBuf, shape);
  }
  let glyphShape;
  const glyphEntries = [];
  if (mask === ValueMask.truetype) {
    glyphShape = [15, 15, 1];
    const loaded = loadGlyphs(glyphShape) || [];
    for (const glyph of loaded) {
      const gh = glyphShape[0];
      const gw = glyphShape[1];
      const flat = new Float32Array(gh * gw);
      let sum = 0;
      for (let y = 0; y < gh; y++) {
        for (let x = 0; x < gw; x++) {
          const v = glyph[y][x][0];
          flat[y * gw + x] = v;
          sum += v;
        }
      }
      glyphEntries.push({ data: flat, sum });
    }
  } else {
    glyphShape = maskShape(mask);
    if (!glyphShape) return tensor;
    const atlas = getAtlas(mask);
    const baseShape = maskShape(mask);
    const levels = 100;
    const uvShapeY = Math.max(1, Math.floor(glyphShape[0] / baseShape[0]));
    const uvShapeX = Math.max(1, Math.floor(glyphShape[1] / baseShape[1]));
    for (let i = 0; i < levels; i++) {
      const uvVal = i / levels;
      const uvNoise = [];
      for (let uy = 0; uy < uvShapeY; uy++) {
        const row = new Array(uvShapeX).fill(uvVal);
        uvNoise.push(row);
      }
      const [glyphTensor] = maskValues(mask, [...glyphShape], {
        atlas,
        time,
        speed,
        uvNoise,
      });
      const dataMaybe = glyphTensor.read();
      const glyphData =
        dataMaybe && typeof dataMaybe.then === "function"
          ? await dataMaybe
          : dataMaybe;
      const flat = glyphData instanceof Float32Array
        ? glyphData.slice()
        : new Float32Array(glyphData);
      let sum = 0;
      for (let j = 0; j < flat.length; j++) sum += flat[j];
      glyphEntries.push({ data: flat, sum });
    }
  }
  if (!glyphEntries.length) return tensor;
  glyphEntries.sort((a, b) => a.sum - b.sum);
  const glyphDataList = glyphEntries.map((entry) => entry.data);
  const [h, w, c] = shape;
  const gh = glyphShape[0];
  const gw = glyphShape[1];
  const glyphChannels = glyphShape[2] ?? 1;
  const glyphCount = glyphDataList.length;

  const inH = Math.max(1, Math.floor(h / zoom));
  const inW = Math.max(1, Math.floor(w / zoom));
  const uvH = Math.max(1, Math.floor(inH / gh));
  const uvW = Math.max(1, Math.floor(inW / gw));
  const approxH = gh * uvH;
  const approxW = gw * uvW;

  let valueTensor = await toValueMap(tensor);
  if (valueTensor.shape[0] !== inH || valueTensor.shape[1] !== inW) {
    valueTensor = await resample(valueTensor, [inH, inW, 1]);
  }
  valueTensor = await normalize(valueTensor);
  let uvNoise = await proportionalDownsample(
    valueTensor,
    [inH, inW, 1],
    [uvH, uvW],
  );
  uvNoise = await resample(uvNoise, [approxH, approxW, 1], splineOrder);
  const uvDataMaybe = uvNoise.read();
  const uvData =
    uvDataMaybe && typeof uvDataMaybe.then === "function"
      ? await uvDataMaybe
      : uvDataMaybe;

  const approxData = new Float32Array(approxH * approxW);
  for (let y = 0; y < approxH; y++) {
    for (let x = 0; x < approxW; x++) {
      const idx = y * approxW + x;
      let gIdx = Math.floor(uvData[idx] * glyphCount);
      if (glyphCount > 0) {
        gIdx = ((gIdx % glyphCount) + glyphCount) % glyphCount;
      }
      const glyph = glyphDataList[gIdx];
      if (!glyph) continue;
      const gy = y % gh;
      const gx = x % gw;
      approxData[idx] = glyph[(gy * gw + gx) * glyphChannels];
    }
  }

  const finalSplineOrder =
    mask === ValueMask.truetype ? InterpolationType.cosine : splineOrder;
  let maskTensor = Tensor.fromArray(ctx, approxData, [approxH, approxW, 1]);
  maskTensor = await resample(maskTensor, [h, w, 1], finalSplineOrder);

  if (!colorize) {
    if (c === 1) {
      if (alpha !== 1) {
        return await blend(tensor, maskTensor, alpha);
      }
      return maskTensor;
    }
    const maskDataMaybe = maskTensor.read();
    const maskData =
      maskDataMaybe && typeof maskDataMaybe.then === "function"
        ? await maskDataMaybe
        : maskDataMaybe;
    const expanded = new Float32Array(h * w * c);
    for (let i = 0; i < h * w; i++) {
      const base = i * c;
      const v = maskData[i];
      for (let k = 0; k < c; k++) {
        expanded[base + k] = v;
      }
    }
    let outTensor = Tensor.fromArray(ctx, expanded, shape);
    if (alpha !== 1) {
      outTensor = await blend(tensor, outTensor, alpha);
    }
    return outTensor;
  }

  let colorTensor = await proportionalDownsample(tensor, shape, [uvH, uvW]);
  colorTensor = await resample(colorTensor, shape, finalSplineOrder);
  const maskDataMaybe = maskTensor.read();
  const maskData =
    maskDataMaybe && typeof maskDataMaybe.then === "function"
      ? await maskDataMaybe
      : maskDataMaybe;
  const colorDataMaybe = colorTensor.read();
  const colorData =
    colorDataMaybe && typeof colorDataMaybe.then === "function"
      ? await colorDataMaybe
      : colorDataMaybe;
  const out = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    const base = i * c;
    const v = maskData[i];
    for (let k = 0; k < c; k++) {
      out[base + k] = v * colorData[base + k];
    }
  }
  let outTensor = Tensor.fromArray(ctx, out, shape);
  if (alpha !== 1) {
    outTensor = await blend(tensor, outTensor, alpha);
  }
  return outTensor;
}
register("glyph_map", glyphMap, {
  mask: ValueMask.truetype,
  colorize: true,
  zoom: 1,
  alpha: 1,
  splineOrder: InterpolationType.constant,
});

async function dlaCPU(
  tensor,
  shape,
  time,
  speed,
  padding = 2,
  seedDensity = 0.01,
  density = 0.125,
  xy = null,
  alpha = 1,
) {
  const [height, width, channels] = shape;
  const neighborhoods = new Set();
  const expandedNeighborhoods = new Set();
  const clustered = [];
  const walkers = [];
  const scale = 1 / padding;
  const halfWidth = Math.floor(width * scale);
  const halfHeight = Math.floor(height * scale);
  let x, y, seedCount;
  if (xy === null) {
    seedCount = Math.floor(
      Math.sqrt(Math.floor(halfHeight * seedDensity) || 1),
    );
    [x, y] = pointCloud(seedCount, {
      distrib: PointDistribution.random,
      shape,
      time,
      speed,
    });
  } else {
    [x, y, seedCount] = xy;
  }
  const walkersCount = halfHeight * halfWidth * density;
  const walkersPerSeed = Math.floor(walkersCount / seedCount);
  const offsets = [-1, 0, 1];
  const expandedRange = 8;
  const expandedOffsets = [];
  for (let i = -expandedRange; i <= expandedRange; i++) expandedOffsets.push(i);
  for (let i = 0; i < seedCount; i++) {
    const node = [Math.floor(y[i] * scale), Math.floor(x[i] * scale)];
    clustered.push(node);
    for (const xo of offsets) {
      for (const yo of offsets) {
        neighborhoods.add(`${node[0] + yo},${node[1] + xo}`);
      }
    }
    for (const xo of expandedOffsets) {
      for (const yo of expandedOffsets) {
        expandedNeighborhoods.add(`${node[0] + yo},${node[1] + xo}`);
      }
    }
    for (let w = 0; w < walkersPerSeed; w++) {
      walkers.push([
        Math.floor(random() * halfHeight),
        Math.floor(random() * halfWidth),
      ]);
    }
  }
  const iterations = Math.floor(Math.sqrt(walkersCount) * time * time);
  for (let i = 0; i < iterations; i++) {
    const remove = [];
    for (const walker of walkers) {
      const key = `${walker[0]},${walker[1]}`;
      if (neighborhoods.has(key)) remove.push(walker);
    }
    for (const walker of remove) {
      const idx = walkers.indexOf(walker);
      if (idx !== -1) walkers.splice(idx, 1);
      for (const xo of offsets) {
        for (const yo of offsets) {
          neighborhoods.add(
            `${(walker[0] + yo + halfHeight) % halfHeight},${
              (walker[1] + xo + halfWidth) % halfWidth
            }`,
          );
        }
      }
      for (const xo of expandedOffsets) {
        for (const yo of expandedOffsets) {
          expandedNeighborhoods.add(
            `${(walker[0] + yo + halfHeight) % halfHeight},${
              (walker[1] + xo + halfWidth) % halfWidth
            }`,
          );
        }
      }
      clustered.push(walker);
    }
    if (!walkers.length) break;
    for (let w = 0; w < walkers.length; w++) {
      const walker = walkers[w];
      const key = `${walker[0]},${walker[1]}`;
      let yo, xo;
      if (expandedNeighborhoods.has(key)) {
        yo = offsets[randomInt(0, offsets.length - 1)];
        xo = offsets[randomInt(0, offsets.length - 1)];
      } else {
        yo = expandedOffsets[randomInt(0, expandedOffsets.length - 1)];
        xo = expandedOffsets[randomInt(0, expandedOffsets.length - 1)];
      }
      walker[0] = (walker[0] + yo + halfHeight) % halfHeight;
      walker[1] = (walker[1] + xo + halfWidth) % halfWidth;
    }
  }
  const uniqueMap = new Map();
  for (const c of clustered) {
    const key = `${c[0]},${c[1]}`;
    if (!uniqueMap.has(key)) uniqueMap.set(key, c);
  }
  const unique = Array.from(uniqueMap.values());
  const count = unique.length;
  const hot = new Float32Array(count * channels);
  for (let i = 0; i < count; i++) {
    const val = count - 1 - i;
    for (let k = 0; k < channels; k++) hot[i * channels + k] = val;
  }
  const grid = new Float32Array(height * width * channels);
  for (let i = 0; i < count; i++) {
    const [yy, xx] = unique[i];
    const sy = yy * padding;
    const sx = xx * padding;
    const base = (sy * width + sx) * channels;
    for (let k = 0; k < channels; k++) {
      grid[base + k] = hot[i * channels + k];
    }
  }
  const scattered = Tensor.fromArray(tensor.ctx, grid, shape);
  const kernelTensor = maskValues(ValueMask.conv2d_blur)[0];
  const kData = await kernelTensor.read();
  const kernel = [];
  for (let i = 0; i < 5; i++) {
    kernel.push(Array.from(kData.slice(i * 5, i * 5 + 5)));
  }
  const convolved = await convolution(scattered, kernel);
  const convData = await convolved.read();
  const tensorData = await tensor.read();
  const mult = new Float32Array(convData.length);
  for (let i = 0; i < convData.length; i++) {
    mult[i] = convData[i] * tensorData[i];
  }
  const out = Tensor.fromArray(tensor.ctx, mult, shape);
  return blend(tensor, out, alpha);
}

async function dlaWebGPU(
  tensor,
  shape,
  time,
  speed,
  padding = 2,
  seedDensity = 0.01,
  density = 0.125,
  xy = null,
  alpha = 1,
) {
  const ctx = tensor ? tensor.ctx : null;
  if (!ctx || !ctx.device) {
    return await dlaCPU(
      tensor,
      shape,
      time,
      speed,
      padding,
      seedDensity,
      density,
      xy,
      alpha,
    );
  }
  const [height, width, channels] = shape;
  const scale = 1 / padding;
  const halfWidth = Math.floor(width * scale);
  const halfHeight = Math.floor(height * scale);
  let x, y, seedCount;
  if (xy === null) {
    seedCount = Math.floor(
      Math.sqrt(Math.floor(halfHeight * seedDensity) || 1),
    );
    [x, y] = pointCloud(seedCount, {
      distrib: PointDistribution.random,
      shape,
      time,
      speed,
    });
  } else {
    [x, y, seedCount] = xy;
  }
  const walkersCount = halfHeight * halfWidth * density;
  const walkersPerSeed = Math.floor(walkersCount / seedCount);
  const offsets = [-1, 0, 1];
  const expandedRange = 8;
  const expandedOffsets = [];
  for (let i = -expandedRange; i <= expandedRange; i++) expandedOffsets.push(i);
  const clusterSize = halfHeight * halfWidth;
  const cluster = new Uint32Array(clusterSize);
  const neighborhood = new Uint32Array(clusterSize);
  const expanded = new Uint32Array(clusterSize);
  const walkersArr = new Uint32Array(walkersCount * 2);
  let wIndex = 0;
  for (let i = 0; i < seedCount; i++) {
    const ny = Math.floor(y[i] * scale);
    const nx = Math.floor(x[i] * scale);
    cluster[ny * halfWidth + nx] = i + 1;
    for (const xo of offsets) {
      for (const yo of offsets) {
        const nyo = (ny + yo + halfHeight) % halfHeight;
        const nxo = (nx + xo + halfWidth) % halfWidth;
        neighborhood[nyo * halfWidth + nxo] = 1;
      }
    }
    for (const xo of expandedOffsets) {
      for (const yo of expandedOffsets) {
        const nyo = (ny + yo + halfHeight) % halfHeight;
        const nxo = (nx + xo + halfWidth) % halfWidth;
        expanded[nyo * halfWidth + nxo] = 1;
      }
    }
    for (let w = 0; w < walkersPerSeed; w++) {
      walkersArr[wIndex * 2] = randomInt(0, halfHeight - 1);
      walkersArr[wIndex * 2 + 1] = randomInt(0, halfWidth - 1);
      wIndex++;
    }
  }
  const count = wIndex;
  const iterations = Math.floor(Math.sqrt(walkersCount) * time * time);
  const randLen = iterations * count;
  const randY = new Float32Array(randLen);
  const randX = new Float32Array(randLen);
  for (let i = 0; i < randLen; i++) {
    randY[i] = random();
    randX[i] = random();
  }
  const walkersBuf = ctx.createGPUBuffer(
    walkersArr.subarray(0, count * 2),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  );
  const clusterBuf = ctx.createGPUBuffer(
    cluster,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  );
  const neighborhoodBuf = ctx.createGPUBuffer(
    neighborhood,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  );
  const expandedBuf = ctx.createGPUBuffer(
    expanded,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  );
  const randYBuf = ctx.createGPUBuffer(randY, GPUBufferUsage.STORAGE);
  const randXBuf = ctx.createGPUBuffer(randX, GPUBufferUsage.STORAGE);
  const counterBuf = ctx.createGPUBuffer(
    new Uint32Array([seedCount]),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  );
  const paramsBuf = ctx.createGPUBuffer(
    new Uint32Array([halfWidth, halfHeight, count, iterations]),
    GPUBufferUsage.UNIFORM,
  );
  try {
    await ctx.runCompute(
      DLA_WGSL,
      [
        { binding: 0, resource: { buffer: walkersBuf } },
        { binding: 1, resource: { buffer: clusterBuf } },
        { binding: 2, resource: { buffer: neighborhoodBuf } },
        { binding: 3, resource: { buffer: expandedBuf } },
        { binding: 4, resource: { buffer: randYBuf } },
        { binding: 5, resource: { buffer: randXBuf } },
        { binding: 6, resource: { buffer: counterBuf } },
        { binding: 7, resource: { buffer: paramsBuf } },
      ],
      Math.ceil(count / 64),
    );
  } catch (e) {
    console.warn('dlaWebGPU fallback to CPU', e);
    return await dlaCPU(
      tensor,
      shape,
      time,
      speed,
      padding,
      seedDensity,
      density,
      xy,
      alpha,
    );
  }
  const clusterOut = await ctx.readGPUBuffer(clusterBuf, clusterSize * 4);
  let maxOrder = 0;
  for (let i = 0; i < clusterOut.length; i++) {
    if (clusterOut[i] > maxOrder) maxOrder = clusterOut[i];
  }
  const grid = new Float32Array(height * width * channels);
  for (let i = 0; i < clusterOut.length; i++) {
    const order = clusterOut[i];
    if (order === 0) continue;
    const val = maxOrder - order;
    const yy = Math.floor(i / halfWidth);
    const xx = i % halfWidth;
    const sy = yy * padding;
    const sx = xx * padding;
    const base = (sy * width + sx) * channels;
    for (let k = 0; k < channels; k++) {
      grid[base + k] = val;
    }
  }
  const scattered = Tensor.fromArray(ctx, grid, shape);
  const kernelTensor = maskValues(ValueMask.conv2d_blur)[0];
  const kData = await kernelTensor.read();
  const kernel = [];
  for (let i = 0; i < 5; i++) {
    kernel.push(Array.from(kData.slice(i * 5, i * 5 + 5)));
  }
  const convolved = await convolution(scattered, kernel);
  const zeroTensor = Tensor.fromArray(
    ctx,
    new Float32Array(height * width * channels),
    shape,
  );
  const multiplied = await blend(zeroTensor, tensor, convolved);
  return blend(tensor, multiplied, alpha);
}

export async function dla(
  tensor,
  shape,
  time,
  speed,
  padding = 2,
  seedDensity = 0.01,
  density = 0.125,
  xy = null,
  alpha = 1,
) {
  const ctx = tensor ? tensor.ctx : null;
  if (ctx && ctx.device) {
    return await dlaWebGPU(
      tensor,
      shape,
      time,
      speed,
      padding,
      seedDensity,
      density,
      xy,
      alpha,
    );
  }
  return await dlaCPU(
    tensor,
    shape,
    time,
    speed,
    padding,
    seedDensity,
    density,
    xy,
    alpha,
  );
}

register("dla", dla, {
  padding: 2,
  seedDensity: 0.01,
  density: 0.125,
  xy: null,
  alpha: 1,
});

export async function simpleFrame(tensor, shape, time, speed, brightness = 0) {
  const [h, w, c] = shape;
  const cx = (w - 1) / 2;
  const cy = (h - 1) / 2;
  const maskData = new Float32Array(h * w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = Math.abs(x - cx) / (cx || 1);
      const dy = Math.abs(y - cy) / (cy || 1);
      maskData[y * w + x] = Math.max(dx, dy);
    }
  }
  let border = Tensor.fromArray(tensor.ctx, maskData, [h, w, 1]);
  border = await blend(
    Tensor.fromArray(tensor.ctx, new Float32Array(h * w), [h, w, 1]),
    border,
    0.55,
  );
  border = await posterize(border, [h, w, 1], time, speed, 1);
  const bData = await border.read();
  const maskC = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    for (let k = 0; k < c; k++) maskC[i * c + k] = bData[i];
  }
  const maskTensor = Tensor.fromArray(tensor.ctx, maskC, shape);
  const bright = new Float32Array(h * w * c);
  bright.fill(brightness);
  const brightTensor = Tensor.fromArray(tensor.ctx, bright, shape);
  return await blend(tensor, brightTensor, maskTensor);
}
register("simple_frame", simpleFrame, { brightness: 0 });

async function simpleMultiresTensor(
  freq,
  shape,
  time,
  speed,
  octaves,
  ctx,
  ridges = false,
  opts = {},
) {
  const [h, w, c] = shape;
  const freqArr = Array.isArray(freq)
    ? freq
    : freqForShape(freq, [h, w]);
  const data = new Float32Array(h * w * c);
  for (let octave = 1; octave <= octaves; octave++) {
    const multiplier = 2 ** octave;
    const baseFreq = freqArr.map((f) =>
      Math.floor(f * 0.5 * multiplier),
    );
    if (baseFreq[0] > h && baseFreq[1] > w) {
      break;
    }
    let layerTensor = await values(baseFreq, shape, {
      ctx,
      time,
      speed,
      ...opts,
    });
    if (layerTensor && typeof layerTensor.then === "function") {
      layerTensor = await layerTensor;
    }
    if (ridges) {
      layerTensor = await ridge(layerTensor);
    }
    const layerDataMaybe = layerTensor.read();
    const layerData =
      layerDataMaybe && typeof layerDataMaybe.then === "function"
        ? await layerDataMaybe
        : layerDataMaybe;
    const layerArray =
      layerData instanceof Float32Array
        ? layerData
        : new Float32Array(layerData ?? []);
    for (let i = 0; i < data.length; i++) {
      data[i] += layerArray[i] / multiplier;
    }
  }
  const tensor = Tensor.fromArray(ctx, data, shape);
  return normalize(tensor);
}

export async function frame(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const halfH = Math.max(1, Math.floor(h * 0.5));
  const halfW = Math.max(1, Math.floor(w * 0.5));
  const halfShape = [halfH, halfW, c];
  const halfValueShape = [halfH, halfW, 1];

  const noiseMaybe = await simpleMultiresTensor(
    64,
    halfValueShape,
    time,
    speed,
    8,
    ctx,
  );
  const noise =
    noiseMaybe && typeof noiseMaybe.then === "function"
      ? await noiseMaybe
      : noiseMaybe;
  const noiseDataMaybe = noise.read();
  const noiseData =
    noiseDataMaybe && typeof noiseDataMaybe.then === "function"
      ? await noiseDataMaybe
      : noiseDataMaybe;

  let maskBase = await singularity(
    null,
    halfValueShape,
    time,
    speed,
    VoronoiDiagramType.range,
    DistanceMetric.chebyshev,
  );
  let maskDataMaybe = maskBase.read();
  let maskData =
    maskDataMaybe && typeof maskDataMaybe.then === "function"
      ? await maskDataMaybe
      : maskDataMaybe;
  for (let i = 0; i < maskData.length; i++) {
    maskData[i] = 1 - maskData[i];
  }
  const combined = new Float32Array(maskData.length);
  for (let i = 0; i < maskData.length; i++) {
    combined[i] = maskData[i] + noiseData[i] * 0.005;
  }
  let mask = await normalize(Tensor.fromArray(ctx, combined, halfValueShape));
  const normDataMaybe = mask.read();
  const normData =
    normDataMaybe && typeof normDataMaybe.then === "function"
      ? await normDataMaybe
      : normDataMaybe;
  for (let i = 0; i < normData.length; i++) {
    normData[i] = Math.sqrt(Math.max(normData[i], 0));
  }
  const sqrtMask = Tensor.fromArray(ctx, normData, halfValueShape);
  const white = Tensor.fromArray(
    ctx,
    new Float32Array(halfH * halfW).fill(1),
    halfValueShape,
  );
  const black = Tensor.fromArray(
    ctx,
    new Float32Array(halfH * halfW),
    halfValueShape,
  );
  mask = blendLayers(sqrtMask, halfValueShape, 0.0125, white, black, black, black);
  if (mask && typeof mask.then === "function") {
    mask = await mask;
  }

  let faded = await proportionalDownsample(tensor, shape, [halfH, halfW]);
  faded = faded && typeof faded.then === "function" ? await faded : faded;
  faded = await adjustBrightness(faded, halfShape, time, speed, 0.1);
  faded = await adjustContrast(faded, halfShape, time, speed, 0.75);
  if (halfH > 1 && halfW > 1) {
    faded = await lightLeak(faded, halfShape, time, speed, 0.125);
    faded = await vignette(faded, halfShape, time, speed, 0.05, 0.75);
  }

  const shadeTensor = await shadow(noise, halfValueShape, time, speed, 1.0);
  const shadeDataMaybe = shadeTensor.read();
  const shadeData =
    shadeDataMaybe && typeof shadeDataMaybe.then === "function"
      ? await shadeDataMaybe
      : shadeDataMaybe;
  const edgeData = new Float32Array(halfH * halfW * c);
  for (let i = 0; i < halfH * halfW; i++) {
    const shade = 0.9 + shadeData[i] * 0.1;
    const base = i * c;
    for (let k = 0; k < c; k++) {
      edgeData[base + k] = shade;
    }
  }
  const edgeTex = Tensor.fromArray(ctx, edgeData, halfShape);

  let out = await blend(faded, edgeTex, mask);
  out = await aberration(out, halfShape, time, speed, 0.00666);
  out = await grime(out, halfShape, time, speed);
  if (c >= 3) {
    out = await saturation(out, halfShape, time, speed, 0.5);
    out = await randomHue(out, halfShape, time, speed, 0.05);
  }

  out = await resample(out, shape);
  out = await scratches(out, shape, time, speed);
  out = await strayHair(out, shape, time, speed);
  return out;
}
register("frame", frame, {});

async function sketchCPU(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const valueShape = [h, w, 1];
  let valuesTensor = tensor;
  if (c !== 1) {
    valuesTensor = toValueMap(tensor);
    if (valuesTensor && typeof valuesTensor.then === "function") {
      valuesTensor = await valuesTensor;
    }
  }
  valuesTensor = await adjustContrast(valuesTensor, valueShape, time, speed, 2.0);
  valuesTensor = await clamp01(valuesTensor);
  const outlineTensorTmp = await derivative(
    valuesTensor,
    valueShape,
    time,
    speed
  );
  let outline = await outlineTensorTmp.read();
  const invValues = await valuesTensor.read();
  const invData = new Float32Array(invValues.length);
  for (let i = 0; i < invValues.length; i++) invData[i] = 1 - invValues[i];
  const d2Tensor = await derivative(
    Tensor.fromArray(tensor.ctx, invData, valueShape),
    valueShape,
    time,
    speed
  );
  const d2 = await d2Tensor.read();
  for (let i = 0; i < outline.length; i++) {
    outline[i] = Math.min(1 - outline[i], 1 - d2[i]);
  }
  let outlineTensor = Tensor.fromArray(tensor.ctx, outline, valueShape);
  outlineTensor = await adjustContrast(
    outlineTensor,
    valueShape,
    time,
    speed,
    0.25
  );
  outlineTensor = await normalize(outlineTensor);
  valuesTensor = await vignette(
    valuesTensor,
    valueShape,
    time,
    speed,
    1.0,
    0.875
  );
  const invValTensor = Tensor.fromArray(tensor.ctx, invData, valueShape);
  let wormsTensor = await worms(
    invValTensor,
    valueShape,
    time,
    speed,
    2,
    125,
    0.5,
    1,
    0.25,
    1.0
  );
  let wormsOut = await wormsTensor.read();
  for (let i = 0; i < wormsOut.length; i++) wormsOut[i] = 1 - wormsOut[i];
  let cross = Tensor.fromArray(tensor.ctx, wormsOut, [h, w, 1]);
  cross = await normalize(cross);
  let combined = await blend(cross, outlineTensor, 0.75);
  combined = await warp(
    combined,
    valueShape,
    time,
    speed,
    [
      Math.max(1, Math.floor(h * 0.125)),
      Math.max(1, Math.floor(w * 0.125)),
    ],
    1,
    0.0025
  );
  const combData = await combined.read();
  for (let i = 0; i < combData.length; i++) combData[i] *= combData[i];
  combined = Tensor.fromArray(tensor.ctx, combData, valueShape);
  if (c === 1) return combined;
  const out = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    const v = combData[i];
    for (let k = 0; k < c; k++) out[i * c + k] = v;
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}

async function sketchWebGPU(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  let valuesTensor = ensureTextureTensor(tensor);
  if (c !== 1) {
    valuesTensor = await runGrayscaleShader(valuesTensor, tensor.shape[2]);
  }
  valuesTensor = await adjustContrast(valuesTensor, [h, w, valuesTensor.shape[2]], time, speed, 2.0);
  valuesTensor = ensureTextureTensor(valuesTensor);
  valuesTensor = await runUnaryShader(valuesTensor, UNARY_OP_CLAMP01, valuesTensor);
  let outlineTensor = await derivative(valuesTensor, [h, w, 1], time, speed);
  outlineTensor = ensureTextureTensor(outlineTensor);
  const invValTensor = await runUnaryShader(valuesTensor, UNARY_OP_INVERT);
  let d2Tensor = await derivative(invValTensor, [h, w, 1], time, speed);
  d2Tensor = ensureTextureTensor(d2Tensor);
  outlineTensor = await runBinaryShader(
    outlineTensor,
    d2Tensor,
    BINARY_OP_MIN_ONE_MINUS,
    outlineTensor,
  );
  outlineTensor = await adjustContrast(outlineTensor, [h, w, 1], time, speed, 0.25);
  outlineTensor = ensureTextureTensor(outlineTensor);
  outlineTensor = await normalize(outlineTensor);
  outlineTensor = ensureTextureTensor(outlineTensor);
  valuesTensor = await vignette(valuesTensor, [h, w, 1], time, speed, 1.0, 0.875);
  valuesTensor = ensureTextureTensor(valuesTensor);
  let wormsTensor = await worms(
    invValTensor,
    [h, w, 1],
    time,
    speed,
    2,
    125,
    0.5,
    1,
    0.25,
    1.0,
  );
  wormsTensor = ensureTextureTensor(wormsTensor);
  wormsTensor = await runUnaryShader(wormsTensor, UNARY_OP_INVERT, wormsTensor);
  wormsTensor = await normalize(wormsTensor);
  wormsTensor = ensureTextureTensor(wormsTensor);
  let combined = await blend(wormsTensor, outlineTensor, 0.75);
  combined = ensureTextureTensor(combined);
  combined = await warp(
    combined,
    [h, w, 1],
    time,
    speed,
    Math.max(1, Math.floor(Math.max(h, w) * 0.125)),
    1,
    0.0025,
  );
  combined = ensureTextureTensor(combined);
  combined = await runUnaryShader(combined, UNARY_OP_SQUARE, combined);
  if (c === 1) return combined;
  return await expandChannelsShader(combined, c);
}

export async function sketch(tensor, shape, time, speed) {
  const ctx = tensor.ctx;
  if (ctx && ctx.device) {
    try {
      return await sketchWebGPU(tensor, shape, time, speed);
    } catch (e) {
      console.warn("WebGPU sketch fallback to CPU", e);
    }
  }
  return await sketchCPU(tensor, shape, time, speed);
}
register("sketch", sketch, {});

export async function nebula(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const valueShape = [h, w, 1];

  async function simpleMultires(freq, octaves, distrib = ValueDistribution.simplex) {
    const freqArr = Array.isArray(freq)
      ? freq
      : freqForShape(freq, [h, w]);
    const accum = new Float32Array(h * w);
    for (let octave = 1; octave <= octaves; octave++) {
      const multiplier = 2 ** octave;
      const baseFreq = freqArr.map((f) => Math.floor(f * 0.5 * multiplier));
      if (baseFreq[0] > h && baseFreq[1] > w) {
        break;
      }
      let layer = await values(baseFreq, valueShape, {
        ctx,
        time,
        speed,
        distrib,
      });
      layer = await ridge(layer);
      const layerData = await layer.read();
      for (let i = 0; i < accum.length; i++) {
        accum[i] += layerData[i] / multiplier;
      }
    }
    const tensorOut = Tensor.fromArray(ctx, accum, valueShape);
    return await normalize(tensorOut);
  }

  const overlayFreq = [randomInt(3, 4), 1];
  const subtractFreq = [randomInt(2, 4), 1];
  let overlay = await simpleMultires(overlayFreq, 6, ValueDistribution.exp);
  const subtractor = await simpleMultires(subtractFreq, 4);
  const overlayData = await overlay.read();
  const subtractData = await subtractor.read();
  const diff = new Float32Array(h * w);
  for (let i = 0; i < diff.length; i++) {
    diff[i] = (overlayData[i] - subtractData[i]) * 0.125;
  }
  overlay = Tensor.fromArray(ctx, diff, valueShape);

  overlay = await rotate(overlay, valueShape, time, speed, randomInt(-15, 15));
  const rotatedData = await overlay.read();
  const baseData = await tensor.read();
  for (let i = 0; i < h * w; i++) {
    const mult = 1 - rotatedData[i];
    for (let k = 0; k < c; k++) {
      baseData[i * c + k] *= mult;
    }
  }

  const expanded = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    const v = Math.max(rotatedData[i], 0);
    for (let k = 0; k < c; k++) {
      expanded[i * c + k] = v;
    }
  }
  const overlayTensor = Tensor.fromArray(ctx, expanded, shape);
  const tinted = await tint(overlayTensor, shape, time, 1.0, 1.0);
  const tintedData = await tinted.read();
  for (let i = 0; i < baseData.length; i++) {
    baseData[i] += tintedData[i];
  }

  return Tensor.fromArray(ctx, baseData, shape);
}
register("nebula", nebula, {});

async function spatterCPU(tensor, shape, time, speed, color = true) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const valueShape = [h, w, 1];
  let smear = await values(randomInt(3, 6), valueShape, {
    ctx,
    time,
    speed,
    distrib: ValueDistribution.exp,
  });
  smear = await warp(
    smear,
    valueShape,
    time,
    speed,
    randomInt(2, 3),
    randomInt(1, 2),
    1 + random(),
  );
  let sp1 = await values(randomInt(32, 64), valueShape, {
    ctx,
    time,
    speed,
    distrib: ValueDistribution.exp,
    splineOrder: InterpolationType.linear,
  });
  let d1 = await sp1.read();
  for (let i = 0; i < d1.length; i++) {
    let v = d1[i] - 1.0;
    v = (v - 0.5) * 4 + 0.5;
    d1[i] = Math.min(1, Math.max(0, v));
  }
  sp1 = Tensor.fromArray(ctx, d1, valueShape);
  let smData = await smear.read();
  let spData = await sp1.read();
  for (let i = 0; i < smData.length; i++)
    smData[i] = Math.max(smData[i], spData[i]);
  smear = Tensor.fromArray(ctx, smData, valueShape);
  let sp2 = await values(randomInt(150, 200), valueShape, {
    ctx,
    time,
    speed,
    distrib: ValueDistribution.exp,
    splineOrder: InterpolationType.linear,
  });
  let d2 = await sp2.read();
  for (let i = 0; i < d2.length; i++) {
    let v = d2[i] - 1.25;
    v = (v - 0.5) * 4 + 0.5;
    d2[i] = Math.min(1, Math.max(0, v));
  }
  sp2 = Tensor.fromArray(ctx, d2, valueShape);
  smData = await smear.read();
  spData = await sp2.read();
  for (let i = 0; i < smData.length; i++)
    smData[i] = Math.max(smData[i], spData[i]);
  smear = Tensor.fromArray(ctx, smData, valueShape);
  const remover = await values(randomInt(2, 3), valueShape, {
    ctx,
    time,
    speed,
    distrib: ValueDistribution.exp,
  });
  const remData = await remover.read();
  smData = await smear.read();
  for (let i = 0; i < smData.length; i++)
    smData[i] = Math.max(0, smData[i] - remData[i]);
  smear = Tensor.fromArray(ctx, smData, valueShape);
  let mask = await normalize(smear);
  let maskData = await mask.read();
  for (let i = 0; i < maskData.length; i++) maskData[i] *= 0.005;
  const alphaTensor = Tensor.fromArray(ctx, maskData, valueShape);
  let overlay;
  if (c === 3 && color) {
    if (Array.isArray(color)) {
      const colData = new Float32Array(h * w * 3);
      for (let i = 0; i < h * w; i++) {
        colData[i * 3] = color[0];
        colData[i * 3 + 1] = color[1];
        colData[i * 3 + 2] = color[2];
      }
      overlay = Tensor.fromArray(ctx, colData, shape);
    } else {
      const baseData = new Float32Array(h * w * 3);
      for (let i = 0; i < h * w; i++) {
        baseData[i * 3] = 0.875;
        baseData[i * 3 + 1] = 0.125;
        baseData[i * 3 + 2] = 0.125;
      }
      let base = Tensor.fromArray(ctx, baseData, shape);
      const hsv = await rgbToHsv(base);
      const hsvData = await hsv.read();
      const delta = random() - 0.5;
      for (let i = 0; i < h * w; i++) {
        hsvData[i * 3] = (hsvData[i * 3] + delta + 1) % 1;
      }
      overlay = hsvToRgb(Tensor.fromArray(ctx, hsvData, hsv.shape));
    }
  } else {
    overlay = await values(1, shape, { ctx, distrib: ValueDistribution.ones });
  }
  return blend(tensor, overlay, alphaTensor);
}

export async function spatter(tensor, shape, time, speed, color = true) {
  const ctx = tensor.ctx;
  if (ctx && ctx.device) {
    try {
      const [h, w] = shape;
      const valueShape = [h, w, 1];
      let smear = await values(randomInt(3, 6), valueShape, {
        ctx,
        time,
        speed,
        distrib: ValueDistribution.exp,
      });
      smear = await warp(
        smear,
        valueShape,
        time,
        speed,
        randomInt(2, 3),
        randomInt(1, 2),
        1 + random(),
      );
      let sp1 = await values(randomInt(32, 64), valueShape, {
        ctx,
        time,
        speed,
        distrib: ValueDistribution.exp,
        splineOrder: InterpolationType.linear,
      });
      sp1 = await adjustBrightness(sp1, valueShape, time, speed, -1.0);
      sp1 = await adjustContrast(sp1, valueShape, time, speed, 4.0);
      let sp2 = await values(randomInt(150, 200), valueShape, {
        ctx,
        time,
        speed,
        distrib: ValueDistribution.exp,
        splineOrder: InterpolationType.linear,
      });
      sp2 = await adjustBrightness(sp2, valueShape, time, speed, -1.25);
      sp2 = await adjustContrast(sp2, valueShape, time, speed, 4.0);
      const remover = await values(randomInt(2, 3), valueShape, {
        ctx,
        time,
        speed,
        distrib: ValueDistribution.exp,
      });
      const outTex = ctx.device.createTexture({
        size: { width: w, height: h, depthOrArrayLayers: 1 },
        format: 'r32float',
        usage:
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.COPY_DST,
      });
      const paramsBuf = ctx.createGPUBuffer(
        new Float32Array([w, h]),
        GPUBufferUsage.UNIFORM,
      );
      await ctx.runCompute(
        SPATTER_MASK_WGSL,
        [
          { binding: 0, resource: smear.handle.createView() },
          { binding: 1, resource: sp1.handle.createView() },
          { binding: 2, resource: sp2.handle.createView() },
          { binding: 3, resource: remover.handle.createView() },
          { binding: 4, resource: outTex.createView() },
          { binding: 5, resource: { buffer: paramsBuf } },
        ],
        Math.ceil(w / 8),
        Math.ceil(h / 8),
      );
      let mask = new Tensor(ctx, outTex, valueShape);
      mask = await normalize(mask);
      const zero = await values(1, valueShape, {
        ctx,
        distrib: ValueDistribution.zeros,
      });
      mask = await blend(zero, mask, 0.005);
      let overlay;
      if (shape[2] === 3 && color) {
        if (Array.isArray(color)) {
          const colData = new Float32Array(h * w * 3);
          for (let i = 0; i < h * w; i++) {
            colData[i * 3] = color[0];
            colData[i * 3 + 1] = color[1];
            colData[i * 3 + 2] = color[2];
          }
          overlay = Tensor.fromArray(ctx, colData, shape);
        } else {
          const baseData = new Float32Array(h * w * 3);
          for (let i = 0; i < h * w; i++) {
            baseData[i * 3] = 0.875;
            baseData[i * 3 + 1] = 0.125;
            baseData[i * 3 + 2] = 0.125;
          }
          let base = Tensor.fromArray(ctx, baseData, shape);
          const hsv = await rgbToHsv(base);
          const hsvData = await hsv.read();
          const delta = random() - 0.5;
          for (let i = 0; i < h * w; i++) {
            hsvData[i * 3] = (hsvData[i * 3] + delta + 1) % 1;
          }
          overlay = hsvToRgb(Tensor.fromArray(ctx, hsvData, hsv.shape));
        }
      } else {
        overlay = await values(1, shape, {
          ctx,
          distrib: ValueDistribution.ones,
        });
      }
      return blend(tensor, overlay, mask);
    } catch (e) {
      console.warn('WebGPU spatter fallback to CPU', e);
      return spatterCPU(tensor, shape, time, speed, color);
    }
  }
  return spatterCPU(tensor, shape, time, speed, color);
}
register("spatter", spatter, { color: true });

export async function clouds(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const preH = Math.max(1, Math.floor(h * 0.25));
  const preW = Math.max(1, Math.floor(w * 0.25));
  const preShape = [preH, preW, 1];

  const baseFreq = freqForShape(randomInt(2, 4), [preH, preW]);
  const accum = new Float32Array(preH * preW);
  for (let octave = 1; octave <= 8; octave++) {
    const mult = 2 ** octave;
    const octaveFreq = baseFreq.map((f) => Math.floor(f * 0.5 * mult));
    if (octaveFreq[0] > preH && octaveFreq[1] > preW) break;
    let layer = await values(octaveFreq, preShape, { ctx, time, speed });
    layer = await ridge(layer);
    const data = await layer.read();
    const inv = 1 / mult;
    for (let i = 0; i < accum.length; i++) accum[i] += data[i] * inv;
  }
  let control = Tensor.fromArray(ctx, accum, preShape);
  control = await normalize(control);
  control = await warp(control, preShape, time, speed, 3, 2, 0.125);

  const ones = new Float32Array(preH * preW);
  ones.fill(1);
  const zeros = new Float32Array(preH * preW);
  const layer0 = Tensor.fromArray(ctx, ones, preShape);
  const layer1 = Tensor.fromArray(ctx, zeros, preShape);
  let combined = blendLayers(control, preShape, 1.0, layer0, layer1);

  let shaded = await offsetTensor(
    combined,
    randomInt(-15, 15),
    randomInt(-15, 15),
  );
  let shadeData = await shaded.read();
  for (let i = 0; i < shadeData.length; i++) {
    shadeData[i] = Math.min(shadeData[i] * 2.5, 1);
  }
  shaded = Tensor.fromArray(ctx, shadeData, preShape);

  const blurTensor = maskValues(ValueMask.conv2d_blur)[0];
  const blurData = await blurTensor.read();
  const blurSize = blurTensor.shape[0];
  const blurKernel = [];
  for (let y = 0; y < blurSize; y++) {
    blurKernel.push(
      Array.from(blurData.slice(y * blurSize, y * blurSize + blurSize)),
    );
  }
  for (let i = 0; i < 3; i++) {
    shaded = await convolution(shaded, blurKernel);
  }

  const postShape = [h, w, 1];
  shaded = await resample(shaded, postShape);
  combined = await resample(combined, postShape);

  const shadedArr = await shaded.read();
  for (let i = 0; i < shadedArr.length; i++) shadedArr[i] *= 0.75;
  shaded = Tensor.fromArray(ctx, shadedArr, postShape);

  const zerosTensor = await values(1, shape, {
    ctx,
    distrib: ValueDistribution.zeros,
  });
  const onesTensor = await values(1, shape, {
    ctx,
    distrib: ValueDistribution.ones,
  });

  let out = await blend(tensor, zerosTensor, shaded);
  out = await blend(out, onesTensor, combined);
  return await shadow(out, shape, time, speed, 0.5);
}
register("clouds", clouds, {});

export async function fibers(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const valueShape = [h, w, 1];
  let out = tensor;
  for (let i = 0; i < 4; i++) {
    let mask = values(4, valueShape, { ctx, time, speed });
    const density = 0.05 + random() * 0.00125;
    const kink = randomInt(5, 10);
    mask = await worms(
      mask,
      valueShape,
      time,
      speed,
      WormBehavior.chaotic,
      density,
      1,
      0.75,
      0.125,
      1,
      kink,
    );
    const brightness = values(128, shape, { ctx, time, speed });
    let maskData = await mask.read();
    for (let j = 0; j < maskData.length; j++) maskData[j] *= 0.5;
    mask = Tensor.fromArray(ctx, maskData, valueShape);
    out = await blend(out, brightness, mask);
  }
  return out;
}
register("fibers", fibers, {});

async function scratchesCPU(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const valueShape = [h, w, 1];
  let out = tensor;
  for (let i = 0; i < 4; i++) {
    let mask = await values(randomInt(2, 4), valueShape, { ctx, time, speed });
    const behavior = [WormBehavior.obedient, WormBehavior.unruly][
      randomInt(0, 1)
    ];
    const density = 0.25 + random() * 0.25;
    const duration = 2 + random() * 2;
    const kink = 0.125 + random() * 0.125;
    mask = await worms(
      mask,
      valueShape,
      time,
      speed,
      behavior,
      density,
      duration,
      0.75,
      0.5,
      1,
      kink,
    );
    const sub = await values(randomInt(2, 4), valueShape, { ctx, time, speed });
    // ``mask`` and ``sub`` may resolve to promises; ensure data is awaited
    let maskData = await mask.read();
    const subData = await sub.read();
    for (let j = 0; j < maskData.length; j++) {
      maskData[j] = Math.max(maskData[j] - subData[j] * 2.0, 0);
    }
    mask = Tensor.fromArray(ctx, maskData, valueShape);
    const outData = await out.read();
    maskData = await mask.read();
    for (let j = 0; j < h * w; j++) {
      const m = Math.min(maskData[j] * 8.0, 1.0);
      for (let k = 0; k < c; k++) {
        const idx = j * c + k;
        outData[idx] = Math.max(outData[idx], m);
      }
    }
    out = Tensor.fromArray(ctx, outData, shape);
  }
  return out;
}

async function scratchesWebGPU(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const valueShape = [h, w, 1];
  const iterations = [];
  for (let i = 0; i < 4; i++) {
    let mask = await values(randomInt(2, 4), valueShape, { ctx, time, speed });
    const behavior = [WormBehavior.obedient, WormBehavior.unruly][
      randomInt(0, 1)
    ];
    const density = 0.25 + random() * 0.25;
    const duration = 2 + random() * 2;
    const kink = 0.125 + random() * 0.125;
    mask = await worms(
      mask,
      valueShape,
      time,
      speed,
      behavior,
      density,
      duration,
      0.75,
      0.5,
      1,
      kink,
    );
    const sub = await values(randomInt(2, 4), valueShape, { ctx, time, speed });
    iterations.push({ mask, sub });
  }
  let out = tensor;
  const resources = [];
  await ctx.withEncoder(async () => {
    for (const { mask, sub } of iterations) {
      resources.push(out);
      const maskTex = ctx.device.createTexture({
        size: { width: w, height: h, depthOrArrayLayers: 1 },
        format: 'r32float',
        usage:
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.COPY_DST,
      });
      const maskParams = ctx.createGPUBuffer(
        new Float32Array([w, h]),
        GPUBufferUsage.UNIFORM,
      );
      const outTex = ctx.device.createTexture({
        size: { width: w, height: h, depthOrArrayLayers: 1 },
        format: c === 1 ? 'r32float' : c === 2 ? 'rg32float' : 'rgba32float',
        usage:
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.COPY_DST,
      });
      const blendParams = ctx.createGPUBuffer(
        new Float32Array([w, h, c]),
        GPUBufferUsage.UNIFORM,
      );
      resources.push(maskParams, blendParams);
      await ctx.runCompute(
        SCRATCHES_MASK_WGSL,
        [
          { binding: 0, resource: mask.handle.createView() },
          { binding: 1, resource: sub.handle.createView() },
          { binding: 2, resource: maskTex.createView() },
          { binding: 3, resource: { buffer: maskParams } },
        ],
        Math.ceil(w / 8),
        Math.ceil(h / 8),
      );
      const maskTensor = new Tensor(ctx, maskTex, valueShape);
      resources.push(maskTensor);
      await ctx.runCompute(
        SCRATCHES_BLEND_WGSL,
        [
          { binding: 0, resource: out.handle.createView() },
          { binding: 1, resource: maskTensor.handle.createView() },
          { binding: 2, resource: outTex.createView() },
          { binding: 3, resource: { buffer: blendParams } },
        ],
        Math.ceil(w / 8),
        Math.ceil(h / 8),
      );
      const newOut = new Tensor(ctx, outTex, shape);
      resources.push(newOut);
      out = newOut;
    }
  });
  iterations.length = 0;
  resources.length = 0;
  return out;
}

export async function scratches(tensor, shape, time, speed) {
  const ctx = tensor.ctx;
  if (ctx && ctx.device && shape[2] >= 3) {
    try {
      return await scratchesWebGPU(tensor, shape, time, speed);
    } catch (e) {
      console.warn('WebGPU scratches fallback to CPU', e);
      return scratchesCPU(tensor, shape, time, speed);
    }
  }
  return scratchesCPU(tensor, shape, time, speed);
}
register("scratches", scratches, {});

export async function strayHair(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const valueShape = [h, w, 1];
  let mask = await values(4, valueShape, { ctx, time, speed });
  const density = 0.0025 + random() * 0.00125;
  const duration = randomInt(8, 16);
  const kink = randomInt(5, 50);
  mask = await worms(
    mask,
    valueShape,
    time,
    speed,
    WormBehavior.unruly,
    density,
    duration,
    0.5,
    0.25,
    1,
    kink,
  );
  let brightness = await values(32, valueShape, { ctx, time, speed });
  let bData = await brightness.read();
  for (let i = 0; i < bData.length; i++) bData[i] *= 0.333;
  brightness = Tensor.fromArray(ctx, bData, valueShape);
  let mData = await mask.read();
  for (let i = 0; i < mData.length; i++) mData[i] *= 0.666;
  mask = Tensor.fromArray(ctx, mData, valueShape);
  return await blend(tensor, brightness, mask);
}
register("stray_hair", strayHair, {});
register("stray_hair", strayHair, {});

async function expandChannels(tensor, channels) {
  const [h, w, c] = tensor.shape;
  if (c === channels) return tensor;
  const data = await tensor.read();
  const out = new Float32Array(h * w * channels);
  for (let i = 0; i < h * w; i++) {
    const v = data[i * c];
    for (let k = 0; k < channels; k++) out[i * channels + k] = v;
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, channels]);
}

function randomGlyphMask(shape, glyphs) {
  const [h, w, c = 1] = shape;
  const gShape = maskShape(glyphs[0]);
  const gh = gShape[0];
  const gw = gShape[1];
  const out = new Float32Array(h * w * c);
  for (let y = 0; y < h; y += gh) {
    for (let x = 0; x < w; x += gw) {
      const id = glyphs[randomInt(0, glyphs.length - 1)];
      const [g] = maskValues(id, gShape);
      const gData = g.read();
      for (let gy = 0; gy < gh; gy++) {
        for (let gx = 0; gx < gw; gx++) {
          const yy = y + gy;
          const xx = x + gx;
          if (yy >= h || xx >= w) continue;
          const v = gData[gy * gw + gx];
          for (let k = 0; k < c; k++) out[(yy * w + xx) * c + k] = v;
        }
      }
    }
  }
  return Tensor.fromArray(null, out, [h, w, c]);
}

async function grimeCPU(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const valueShape = [h, w, 1];

  const baseMask = await fbm(null, valueShape, time, speed, 5, 8);
  const baseDataMaybe = baseMask.read();
  const baseDataRaw =
    baseDataMaybe && typeof baseDataMaybe.then === "function"
      ? await baseDataMaybe
      : baseDataMaybe;
  const baseData =
    baseDataRaw instanceof Float32Array
      ? baseDataRaw
      : new Float32Array(baseDataRaw ?? []);
  const rxTensor = Tensor.fromArray(ctx, baseData.slice(), valueShape);
  const offsetData = new Float32Array(h * w);
  const xOffset = Math.floor(w * 0.5);
  const yOffset = Math.floor(h * 0.5);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const oy = (y + yOffset) % h;
      const ox = (x + xOffset) % w;
      offsetData[idx] = baseData[oy * w + ox];
    }
  }
  const ryTensor = Tensor.fromArray(ctx, offsetData, valueShape);
  let mask = await refractOp(
    baseMask,
    rxTensor,
    ryTensor,
    1.0,
    InterpolationType.bicubic,
    true,
  );
  mask = await derivative(
    mask,
    valueShape,
    time,
    speed,
    DistanceMetric.chebyshev,
    true,
    0.125,
  );
  let gateDataMaybe = mask.read();
  let gateData =
    gateDataMaybe && typeof gateDataMaybe.then === "function"
      ? await gateDataMaybe
      : gateDataMaybe;
  gateData =
    gateData instanceof Float32Array
      ? gateData
      : new Float32Array(gateData ?? []);
  const gate = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    const v = gateData[i] * gateData[i] * 0.075;
    for (let k = 0; k < c; k++) gate[i * c + k] = v;
  }
  const gateTensor = Tensor.fromArray(ctx, gate, shape);
  const baseValues = new Float32Array(h * w * c).fill(0.25);
  const baseTensor = Tensor.fromArray(ctx, baseValues, shape);
  let dusty = await blend(tensor, baseTensor, gateTensor);

  const speckFreq = [
    Math.max(1, Math.floor(h * 0.25)),
    Math.max(1, Math.floor(w * 0.25)),
  ];
  let specks = await values(speckFreq, valueShape, {
    ctx,
    time,
    speed,
    distrib: ValueDistribution.exp,
    mask: ValueMask.dropout,
  });
  const speckDataMaybe = specks.read();
  const speckDataRaw =
    speckDataMaybe && typeof speckDataMaybe.then === "function"
      ? await speckDataMaybe
      : speckDataMaybe;
  const speckData =
    speckDataRaw instanceof Float32Array
      ? speckDataRaw
      : new Float32Array(speckDataRaw ?? []);
  const cosData = new Float32Array(h * w);
  const sinData = new Float32Array(h * w);
  for (let i = 0; i < h * w; i++) {
    const angle = speckData[i] * TAU;
    const cosVal = Math.cos(angle) * 0.5 + 0.5;
    const sinVal = Math.sin(angle) * 0.5 + 0.5;
    cosData[i] = Math.fround(Math.min(Math.max(cosVal, 0), 1));
    sinData[i] = Math.fround(Math.min(Math.max(sinVal, 0), 1));
  }
  const cosTensor = Tensor.fromArray(ctx, cosData, valueShape);
  const sinTensor = Tensor.fromArray(ctx, sinData, valueShape);
  specks = await refractOp(
    specks,
    cosTensor,
    sinTensor,
    0.25,
    InterpolationType.bicubic,
    true,
  );

  let speckValuesMaybe = specks.read();
  let speckValues =
    speckValuesMaybe && typeof speckValuesMaybe.then === "function"
      ? await speckValuesMaybe
      : speckValuesMaybe;
  speckValues =
    speckValues instanceof Float32Array
      ? speckValues
      : new Float32Array(speckValues ?? []);
  for (let i = 0; i < speckValues.length; i++) {
    speckValues[i] = Math.max(speckValues[i] - 0.625, 0);
  }
  specks = Tensor.fromArray(ctx, speckValues, valueShape);
  specks = await normalize(specks);
  speckValuesMaybe = specks.read();
  speckValues =
    speckValuesMaybe && typeof speckValuesMaybe.then === "function"
      ? await speckValuesMaybe
      : speckValuesMaybe;
  speckValues =
    speckValues instanceof Float32Array
      ? speckValues
      : new Float32Array(speckValues ?? []);
  for (let i = 0; i < speckValues.length; i++) {
    speckValues[i] = 1 - Math.sqrt(speckValues[i]);
  }
  specks = Tensor.fromArray(ctx, speckValues, valueShape);

  let noise = await values([h, w], valueShape, {
    ctx,
    time,
    speed,
    distrib: ValueDistribution.exp,
    mask: ValueMask.sparse,
  });
  dusty = await blend(dusty, noise, 0.075);
  let dustyDataMaybe = dusty.read();
  let dustyData =
    dustyDataMaybe && typeof dustyDataMaybe.then === "function"
      ? await dustyDataMaybe
      : dustyDataMaybe;
  dustyData =
    dustyData instanceof Float32Array
      ? dustyData
      : new Float32Array(dustyData ?? []);
  const speckMaskMaybe = specks.read();
  let speckMask =
    speckMaskMaybe && typeof speckMaskMaybe.then === "function"
      ? await speckMaskMaybe
      : speckMaskMaybe;
  speckMask =
    speckMask instanceof Float32Array
      ? speckMask
      : new Float32Array(speckMask ?? []);
  for (let i = 0; i < dustyData.length; i++) {
    dustyData[i] *= speckMask[Math.floor(i / c)];
  }
  dusty = Tensor.fromArray(ctx, dustyData, shape);

  gateDataMaybe = mask.read();
  gateData =
    gateDataMaybe && typeof gateDataMaybe.then === "function"
      ? await gateDataMaybe
      : gateDataMaybe;
  gateData =
    gateData instanceof Float32Array
      ? gateData
      : new Float32Array(gateData ?? []);
  const maskScaled = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    const v = gateData[i] * 0.75;
    for (let k = 0; k < c; k++) maskScaled[i * c + k] = v;
  }
  const maskTensor = Tensor.fromArray(ctx, maskScaled, shape);
  return await blend(tensor, dusty, maskTensor);
}

async function grimeWebGPU(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const valueShape = [h, w, 1];
  const maskTex = ctx.device.createTexture({
    size: { width: w, height: h, depthOrArrayLayers: 1 },
    format: 'r32float',
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.COPY_DST,
  });
  const maskParams = ctx.createGPUBuffer(
    new Float32Array([w, h, time, speed]),
    GPUBufferUsage.UNIFORM,
  );
  await ctx.runCompute(
    GRIME_MASK_WGSL,
    [
      { binding: 0, resource: maskTex.createView() },
      { binding: 1, resource: { buffer: maskParams } },
    ],
    Math.ceil(w / 8),
    Math.ceil(h / 8),
  );
  let mask = new Tensor(ctx, maskTex, valueShape);
  mask = await refractOp(mask);
  mask = await derivative(
    mask,
    valueShape,
    time,
    speed,
    DistanceMetric.chebyshev,
    true,
    0.125,
  );
  let specks = await values(Math.max(1, Math.floor(h * 0.25)), valueShape, {
    ctx,
    time,
    speed,
    distrib: ValueDistribution.exp,
    splineOrder: InterpolationType.constant,
  });
  specks = await refractOp(specks, null, null, 0.25);
  let sData = await specks.read();
  for (let i = 0; i < sData.length; i++) {
    sData[i] = Math.max(sData[i] - 0.625, 0);
  }
  specks = Tensor.fromArray(ctx, sData, valueShape);
  specks = await normalize(specks);
  sData = await specks.read();
  for (let i = 0; i < sData.length; i++) sData[i] = 1 - Math.sqrt(sData[i]);
  specks = Tensor.fromArray(ctx, sData, valueShape);
  let noise = await values(Math.max(h, w), valueShape, {
    ctx,
    time,
    speed,
    distrib: ValueDistribution.exp,
  });
  const format = c === 1 ? 'r32float' : c === 2 ? 'rg32float' : 'rgba32float';
  const outTex = ctx.device.createTexture({
    size: { width: w, height: h, depthOrArrayLayers: 1 },
    format,
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.COPY_DST,
  });
  const blendParams = ctx.createGPUBuffer(
    new Float32Array([w, h, c]),
    GPUBufferUsage.UNIFORM,
  );
  const shader = GRIME_BLEND_WGSL.replace('rgba32float', format);
  await ctx.runCompute(
    shader,
    [
      { binding: 0, resource: tensor.handle.createView() },
      { binding: 1, resource: mask.handle.createView() },
      { binding: 2, resource: noise.handle.createView() },
      { binding: 3, resource: specks.handle.createView() },
      { binding: 4, resource: outTex.createView() },
      { binding: 5, resource: { buffer: blendParams } },
    ],
    Math.ceil(w / 8),
    Math.ceil(h / 8),
  );
  return new Tensor(ctx, outTex, shape);
}

export async function grime(tensor, shape, time, speed) {
  const ctx = tensor.ctx;
  if (ctx && ctx.device && shape[2] >= 3) {
    try {
      return await grimeWebGPU(tensor, shape, time, speed);
    } catch (e) {
      console.warn('WebGPU grime fallback to CPU', e);
      return grimeCPU(tensor, shape, time, speed);
    }
  }
  return grimeCPU(tensor, shape, time, speed);
}
register("grime", grime, {});

export async function watermark(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const valueShape = [h, w, 1];
  let mask = await values(240, valueShape, {
    ctx,
    time,
    speed,
    splineOrder: InterpolationType.constant,
    distrib: ValueDistribution.ones,
    mask: ValueMask.alphanum_numeric,
  });
  mask = await crt(mask, valueShape, time, speed);
  mask = await warp(mask, valueShape, time, speed, [2, 4], 1, 0.5);
  const noiseTensor = await values(2, valueShape, { ctx, time, speed });
  const [maskDataMaybe, noiseDataMaybe] = await Promise.all([
    mask.read(),
    noiseTensor.read(),
  ]);
  const maskData =
    maskDataMaybe instanceof Float32Array
      ? maskDataMaybe.slice()
      : Float32Array.from(maskDataMaybe ?? []);
  const noiseData =
    noiseDataMaybe instanceof Float32Array
      ? noiseDataMaybe
      : new Float32Array(noiseDataMaybe ?? []);
  for (let i = 0; i < maskData.length; i++) {
    const n = noiseData[i] || 0;
    maskData[i] = Math.fround(maskData[i] * n * n);
  }
  mask = Tensor.fromArray(ctx, maskData, valueShape);
  let brightness = await values(16, valueShape, { ctx, time, speed });
  if (c > 1) {
    mask = await expandChannels(mask, c);
    brightness = await expandChannels(brightness, c);
  }
  const maskFinalMaybe = mask.read();
  const maskFinal =
    maskFinalMaybe && typeof maskFinalMaybe.then === "function"
      ? await maskFinalMaybe
      : maskFinalMaybe;
  const maskScaled =
    maskFinal instanceof Float32Array
      ? maskFinal.slice()
      : Float32Array.from(maskFinal ?? []);
  for (let i = 0; i < maskScaled.length; i++) {
    maskScaled[i] = Math.fround(maskScaled[i] * 0.125);
  }
  const maskTensor = Tensor.fromArray(ctx, maskScaled, [h, w, c]);
  return await blend(tensor, brightness, maskTensor);
}
register("watermark", watermark, {});

export async function onScreenDisplay(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const glyphCount = randomInt(3, 6);
  const masks = [
    ValueMask.bank_ocr,
    ValueMask.alphanum_hex,
    ValueMask.alphanum_numeric,
  ];
  const mask = masks[randomInt(0, masks.length - 1)];
  const gShape = maskShape(mask);
  let baseWidth = Math.floor(w / 24);
  baseWidth = gShape[1] * Math.floor(baseWidth / gShape[1]);
  if (baseWidth <= 0) {
    return tensor;
  }
  const tileCount = baseWidth / gShape[1];
  const rowHeight = gShape[0] * tileCount;
  const totalWidth = baseWidth * glyphCount;
  const freq = [gShape[0], gShape[1] * glyphCount];
  let rowMask = await values(freq, [rowHeight, totalWidth, c], {
    ctx: tensor.ctx,
    corners: true,
    mask,
    splineOrder: InterpolationType.constant,
    distrib: ValueDistribution.ones,
    time,
    speed,
  });
  const rowDataMaybe = rowMask.read();
  const rowData =
    rowDataMaybe && typeof rowDataMaybe.then === "function"
      ? await rowDataMaybe
      : rowDataMaybe;
  const pad = new Float32Array(h * w * c);
  const yOff = 25;
  const xOff = w - totalWidth - 25;
  for (let y = 0; y < rowHeight; y++) {
    const yy = yOff + y;
    if (yy < 0 || yy >= h) continue;
    for (let x = 0; x < totalWidth; x++) {
      const xx = xOff + x;
      if (xx < 0 || xx >= w) continue;
      const srcBase = (y * totalWidth + x) * c;
      const dstBase = (yy * w + xx) * c;
      for (let k = 0; k < c; k++) {
        pad[dstBase + k] = rowData[srcBase + k];
      }
    }
  }
  const rendered = Tensor.fromArray(tensor.ctx, pad, shape);
  const alpha = 0.5 + random() * 0.25;
  const [renderedDataMaybe, tensorDataMaybe] = await Promise.all([
    rendered.read(),
    tensor.read(),
  ]);
  const renderedData =
    renderedDataMaybe && typeof renderedDataMaybe.then === "function"
      ? await renderedDataMaybe
      : renderedDataMaybe;
  const tensorData =
    tensorDataMaybe && typeof tensorDataMaybe.then === "function"
      ? await tensorDataMaybe
      : tensorDataMaybe;
  for (let i = 0; i < renderedData.length; i++) {
    renderedData[i] = Math.max(renderedData[i], tensorData[i]);
  }
  const maxTensor = Tensor.fromArray(tensor.ctx, renderedData, shape);
  return blend(tensor, maxTensor, alpha);
}
register("on_screen_display", onScreenDisplay, {});

export async function spookyTicker(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  if (random() > 0.75) {
    tensor = await onScreenDisplay(tensor, shape, time, speed);
  }
  const glyphs = [
    ValueMask.lcd_0,
    ValueMask.lcd_1,
    ValueMask.lcd_2,
    ValueMask.lcd_3,
    ValueMask.lcd_4,
    ValueMask.lcd_5,
    ValueMask.lcd_6,
    ValueMask.lcd_7,
    ValueMask.lcd_8,
    ValueMask.lcd_9,
  ];
  let rendered = Tensor.fromArray(null, new Float32Array(h * w), [h, w, 1]);
  let bottom = 2;
  const rows = randomInt(1, 3);
  for (let i = 0; i < rows; i++) {
    const gShape = maskShape(glyphs[0]);
    const rh = gShape[0];
    const rowMask = randomGlyphMask([rh, w, 1], glyphs);
    const rData = rowMask.read();
    const base = rendered.read();
    for (let y = 0; y < rh; y++) {
      const yy = h - bottom - rh + y;
      if (yy < 0 || yy >= h) continue;
      for (let x = 0; x < w; x++) {
        const idx = yy * w + x;
        base[idx] = Math.max(base[idx], rData[y * w + x]);
      }
    }
    rendered = Tensor.fromArray(null, base, [h, w, 1]);
    bottom += rh + 2;
  }
  const alpha = 0.5 + random() * 0.25;
  const offsetMask = await offsetTensor(rendered, -1, -1);
  const tData = await tensor.read();
  const oData = await offsetMask.read();
  const diff = new Float32Array(tData.length);
  for (let i = 0; i < tData.length; i++)
    diff[i] = tData[i] - oData[i % (h * w)];
  const diffTensor = Tensor.fromArray(tensor.ctx, diff, shape);
  tensor = blend(tensor, diffTensor, alpha * 0.333);
  let renderedC = rendered;
  if (c > 1) renderedC = await expandChannels(rendered, c);
  const maxData = renderedC.read();
  const tData2 = await tensor.read();
  for (let i = 0; i < maxData.length; i++)
    maxData[i] = Math.max(maxData[i], tData2[i]);
  const maxTensor = Tensor.fromArray(tensor.ctx, maxData, shape);
  return blend(tensor, maxTensor, alpha);
}
register("spooky_ticker", spookyTicker, {});
register("spooky_ticker", spookyTicker, {});

export function skew(tensor, shape, time, speed, angle = 0, range = 1) {
  // Placeholder implementation; performs no skewing.
  return tensor;
}
register("skew", skew, { angle: 0, range: 1 });

export { voronoiCPU, voronoiWebGPU, erosionWormsWebGPU, wormsCPU, wormsWebGPU };
