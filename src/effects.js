import { Tensor } from "./tensor.js";
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
  refract,
  convolution,
  fxaa,
} from "./value.js";
import { PALETTES } from "./palettes.js";
import { register } from "./effectsRegistry.js";
import { random as simplexRandom } from "./simplex.js";
import { maskValues, maskShape } from "./masks.js";
import { loadGlyphs } from "./glyphs.js";
import { random, randomInt } from "./util.js";
import { pointCloud } from "./points.js";
import {
  InterpolationType,
  DistanceMetric,
  ValueMask,
  ValueDistribution,
  PointDistribution,
  VoronoiDiagramType,
  WormBehavior,
} from "./constants.js";

export function warp(
  tensor,
  shape,
  time,
  speed,
  freq = 2,
  octaves = 5,
  displacement = 1,
) {
  let out = tensor;
  for (let octave = 0; octave < octaves; octave++) {
    const mult = 2 ** octave;
    const f = freq * mult;
    const flowX = values(f, [shape[0], shape[1], 1], {
      seed: 100 + octave,
      time,
    });
    const flowY = values(f, [shape[0], shape[1], 1], {
      seed: 200 + octave,
      time,
    });
    const dx = flowX.read();
    const dy = flowY.read();
    const flowData = new Float32Array(shape[0] * shape[1] * 2);
    for (let i = 0; i < shape[0] * shape[1]; i++) {
      flowData[i * 2] = dx[i] * 2 - 1;
      flowData[i * 2 + 1] = dy[i] * 2 - 1;
    }
    const flow = Tensor.fromArray(tensor.ctx, flowData, [
      shape[0],
      shape[1],
      2,
    ]);
    out = warpOp(out, flow, displacement / mult);
  }
  return out;
}
register("warp", warp, { freq: 2, octaves: 5, displacement: 1 });

export function shadow(tensor, shape, time, speed, alpha = 1) {
  const shade = normalize(sobel(tensor));
  return blend(tensor, shade, alpha);
}
register("shadow", shadow, { alpha: 1 });

export function bloom(tensor, shape, time, speed, alpha = 0.5) {
  const [h, w, c] = shape;
  const src = tensor.read();
  const data = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    let v = src[i] * 2 - 1;
    if (v < 0) v = 0;
    if (v > 1) v = 1;
    data[i] = v;
  }
  let blurred = Tensor.fromArray(tensor.ctx, data, shape);
  const targetH = Math.max(1, Math.floor(h / 100));
  const factor = Math.max(1, Math.floor(h / targetH));
  blurred = downsample(blurred, factor);
  const bData = blurred.read();
  for (let i = 0; i < bData.length; i++) bData[i] *= 4;
  blurred = Tensor.fromArray(tensor.ctx, bData, blurred.shape);
  blurred = upsample(blurred, factor);
  const xOff = Math.floor(w * -0.05);
  const yOff = Math.floor(h * -0.05);
  blurred = offsetTensor(blurred, xOff, yOff);
  const blurRead = blurred.read();
  for (let i = 0; i < blurRead.length; i++) {
    blurRead[i] += 0.25;
  }
  let mean = 0;
  for (let i = 0; i < blurRead.length; i++) mean += blurRead[i];
  mean /= blurRead.length;
  for (let i = 0; i < blurRead.length; i++) {
    blurRead[i] = (blurRead[i] - mean) * 1.5 + mean;
  }
  blurred = Tensor.fromArray(tensor.ctx, blurRead, shape);
  const mixData = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++)
    mixData[i] = (src[i] + blurRead[i]) * 0.5;
  const mixed = clamp01(Tensor.fromArray(tensor.ctx, mixData, shape));
  const clamped = clamp01(tensor);
  return blend(clamped, mixed, alpha);
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
  let out = sobel(tensor);
  if (withNormalize) out = normalize(out);
  if (alpha === 1) return out;
  return blend(tensor, out, alpha);
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
  const blurred = blur(tensor, shape, time, speed);
  let out = sobel(blurred);
  out = normalize(out);
  const data = out.read();
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.abs(data[i] * 2 - 1);
  }
  return Tensor.fromArray(tensor.ctx, data, shape);
}
register("sobel", sobelOperator, {
  distMetric: DistanceMetric.euclidean,
});

export function outline(
  tensor,
  shape,
  time,
  speed,
  sobelMetric = DistanceMetric.euclidean,
  invert = false,
) {
  const [h, w, c] = shape;
  const src = tensor.read();
  const valueData = new Float32Array(h * w);
  for (let i = 0; i < h * w; i++) {
    if (c === 1) {
      valueData[i] = src[i];
    } else {
      const base = i * c;
      const r = src[base];
      const g = src[base + 1] || 0;
      const b = src[base + 2] || 0;
      valueData[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }
  }
  let edges = sobelOperator(
    Tensor.fromArray(tensor.ctx, valueData, [h, w, 1]),
    [h, w, 1],
    time,
    speed,
    sobelMetric,
  ).read();
  if (invert) {
    for (let i = 0; i < edges.length; i++) edges[i] = 1 - edges[i];
  }
  const out = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    const e = edges[i];
    for (let k = 0; k < c; k++) {
      out[i * c + k] = src[i * c + k] * e;
    }
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register("outline", outline, {
  sobelMetric: DistanceMetric.euclidean,
  invert: false,
});

export function glowingEdges(
  tensor,
  shape,
  time,
  speed,
  sobelMetric = DistanceMetric.chebyshev,
  alpha = 1,
) {
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
  const levels = randomInt(3, 5);
  let edges = Tensor.fromArray(tensor.ctx, gray, [h, w, 1]);
  edges = posterize(edges, [h, w, 1], time, speed, levels);
  const blurTensor = maskValues(ValueMask.conv2d_blur)[0];
  const [bh, bw] = maskShape(ValueMask.conv2d_blur);
  const blurFlat = blurTensor.read();
  const blurArr = [];
  for (let y = 0; y < bh; y++) {
    const row = [];
    for (let x = 0; x < bw; x++) row.push(blurFlat[y * bw + x]);
    blurArr.push(row);
  }
  edges = convolution(edges, blurArr);
  const sxArr = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
  ];
  const syArr = [
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1],
  ];
  const gx = convolution(edges, sxArr, { normalize: false });
  const gy = convolution(edges, syArr, { normalize: false });
  const gxData = gx.read();
  const gyData = gy.read();
  const distData = new Float32Array(gxData.length);
  for (let i = 0; i < gxData.length; i++) {
    distData[i] = distance(
      Math.abs(gxData[i]),
      Math.abs(gyData[i]),
      sobelMetric,
    );
  }
  edges = normalize(Tensor.fromArray(tensor.ctx, distData, [h, w, 1]));
  let eData = edges.read();
  for (let i = 0; i < eData.length; i++) eData[i] = 1 - eData[i];
  const mult = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    const e = Math.min(eData[i] * 8, 1);
    for (let k = 0; k < c; k++) {
      const tVal = Math.min(src[i * c + k] * 1.25, 1);
      mult[i * c + k] = e * tVal;
    }
  }
  edges = Tensor.fromArray(tensor.ctx, mult, shape);
  edges = bloom(edges, shape, time, speed, 0.5);
  const kTensor = maskValues(ValueMask.conv2d_blur)[0];
  const [kh, kw] = maskShape(ValueMask.conv2d_blur);
  const kFlat = kTensor.read();
  const kArr = [];
  for (let y = 0; y < kh; y++) {
    const row = [];
    for (let x = 0; x < kw; x++) row.push(kFlat[y * kw + x]);
    kArr.push(row);
  }
  let blurred = convolution(edges, kArr);
  const eData2 = edges.read();
  const bData = blurred.read();
  const sum = new Float32Array(eData2.length);
  for (let i = 0; i < eData2.length; i++) sum[i] = eData2[i] + bData[i];
  edges = normalize(Tensor.fromArray(tensor.ctx, sum, shape));
  const edgesData = edges.read();
  const final = new Float32Array(edgesData.length);
  for (let i = 0; i < edgesData.length; i++) {
    final[i] = 1 - (1 - edgesData[i]) * (1 - src[i]);
  }
  let result = Tensor.fromArray(tensor.ctx, final, shape);
  if (alpha < 1) result = blend(tensor, result, alpha);
  return result;
}
register("glowingEdges", glowingEdges, {
  sobelMetric: DistanceMetric.chebyshev,
  alpha: 1,
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
      let sx = 0,
        sy = 0,
        idx = 0;
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
register("normalMap", normalMap, {});

export function voronoi(
  tensor,
  shape,
  time,
  speed,
  diagramType = VoronoiDiagramType.range,
  nth = 0,
  distMetric = DistanceMetric.euclidean,
  alpha = 1,
  pointFreq = 3,
  pointGenerations = 1,
  pointDistrib = PointDistribution.random,
  pointDrift = 0,
  pointCorners = false,
  xy = null,
) {
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
  }
  if (count === 0) return tensor;
  const distMap = new Float32Array(h * w);
  const indexMap = new Int32Array(h * w);
  let maxDist = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let best = Infinity;
      let bestIdx = 0;
      for (let i = 0; i < count; i++) {
        let dx = Math.abs(x - xPts[i]);
        dx = Math.min(dx, w - dx) / w;
        let dy = Math.abs(y - yPts[i]);
        dy = Math.min(dy, h - dy) / h;
        const d = distance(dx, dy, distMetric);
        if (d < best) {
          best = d;
          bestIdx = i;
        }
      }
      const idx = y * w + x;
      distMap[idx] = best;
      indexMap[idx] = bestIdx;
      if (best > maxDist) maxDist = best;
    }
  }
  let outTensor;
  if (diagramType === VoronoiDiagramType.range) {
    const data = new Float32Array(h * w);
    for (let i = 0; i < h * w; i++) {
      data[i] = Math.sqrt(distMap[i] / maxDist);
    }
    outTensor = Tensor.fromArray(tensor ? tensor.ctx : null, data, [h, w, 1]);
  } else if (diagramType === VoronoiDiagramType.regions) {
    const data = new Float32Array(h * w);
    for (let i = 0; i < h * w; i++) data[i] = indexMap[i] / count;
    outTensor = Tensor.fromArray(tensor ? tensor.ctx : null, data, [h, w, 1]);
  } else if (diagramType === VoronoiDiagramType.color_regions && tensor) {
    const src = tensor.read();
    const colors = new Float32Array(count * c);
    for (let i = 0; i < count; i++) {
      const px = Math.floor(yPts[i]) % h;
      const py = Math.floor(xPts[i]) % w;
      const base = (px * w + py) * c;
      for (let k = 0; k < c; k++) colors[i * c + k] = src[base + k];
    }
    const out = new Float32Array(h * w * c);
    for (let i = 0; i < h * w; i++) {
      const region = indexMap[i];
      for (let k = 0; k < c; k++) {
        out[i * c + k] = colors[region * c + k];
      }
    }
    outTensor = Tensor.fromArray(tensor.ctx, out, shape);
  } else {
    return tensor;
  }
  if (tensor && diagramType !== VoronoiDiagramType.color_regions) {
    return blend(tensor, outTensor, alpha);
  }
  return outTensor;
}
register("voronoi", voronoi, {
  diagramType: VoronoiDiagramType.range,
  nth: 0,
  distMetric: DistanceMetric.euclidean,
  alpha: 1,
  pointFreq: 3,
  pointGenerations: 1,
  pointDistrib: PointDistribution.random,
  pointDrift: 0,
  pointCorners: false,
  xy: null,
});

export function singularity(
  tensor,
  shape,
  time,
  speed,
  diagramType = VoronoiDiagramType.range,
  distMetric = DistanceMetric.euclidean,
) {
  const [x, y] = pointCloud(1, { distrib: PointDistribution.square, shape });
  return voronoi(
    tensor,
    shape,
    time,
    speed,
    diagramType,
    0,
    distMetric,
    1,
    1,
    1,
    PointDistribution.square,
    0,
    false,
    [x, y, 1],
  );
}
register("singularity", singularity, {
  diagramType: VoronoiDiagramType.range,
  distMetric: DistanceMetric.euclidean,
});

export function lowpoly(
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
  const distance = voronoi(
    null,
    shape,
    time,
    speed,
    VoronoiDiagramType.range,
    1,
    distMetric,
    1,
    1,
    1,
    PointDistribution.square,
    0,
    false,
    xy,
  );
  const color = voronoi(
    tensor,
    shape,
    time,
    speed,
    VoronoiDiagramType.color_regions,
    0,
    distMetric,
    1,
    1,
    1,
    PointDistribution.square,
    0,
    false,
    xy,
  );
  return normalize(blend(distance, color, 0.5));
}
register("lowpoly", lowpoly, {
  distrib: PointDistribution.random,
  freq: 10,
  distMetric: DistanceMetric.euclidean,
});

export function kaleido(
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
  const valueShape = [h, w, 1];
  const xyArg = xy ? [xy[0], xy[1], xy[0].length] : null;
  const r = voronoi(
    null,
    valueShape,
    time,
    speed,
    VoronoiDiagramType.range,
    0,
    DistanceMetric.euclidean,
    1,
    pointFreq,
    pointGenerations,
    pointDistrib,
    pointDrift,
    pointCorners,
    xyArg,
  ).read();
  const fader = blendEdges
    ? (() => {
        const f = singularity(
          null,
          valueShape,
          time,
          speed,
          VoronoiDiagramType.range,
          DistanceMetric.chebyshev,
        ).read();
        for (let i = 0; i < f.length; i++) f[i] = Math.pow(f[i], 5);
        return f;
      })()
    : null;
  const src = tensor.read();
  const out = new Float32Array(h * w * c);
  const step = (Math.PI * 2) / sides;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const xi = x / (w - 1) - 0.5;
      const yi = y / (h - 1) - 0.5;
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
      nx = ((Math.floor(nx) % w) + w) % w;
      ny = ((Math.floor(ny) % h) + h) % h;
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

export function texture(tensor, shape, time, speed) {
  const valueShape = [shape[0], shape[1], 1];
  let noise = values(64, valueShape, { ctx: tensor.ctx, time, speed });
  noise = warp(noise, valueShape, time, speed, 2, 8, 1);
  noise = ridge(noise);
  const shade = shadow(noise, valueShape, time, speed, 1).read();
  const src = tensor.read();
  const [h, w, c] = shape;
  const out = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    const m = 0.9 + shade[i] * 0.1;
    for (let k = 0; k < c; k++) out[i * c + k] = src[i * c + k] * m;
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register("texture", texture, {});

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
register("densityMap", densityMap, {});

export function jpegDecimate(tensor, shape, time, speed, iterations = 25) {
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
register("jpegDecimate", jpegDecimate, { iterations: 25 });

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
  alpha = 0.5,
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
  const combinedTensor = Tensor.fromArray(
    convolved.ctx,
    combined,
    convolved.shape,
  );
  const resampled = upsample(combinedTensor, 2);
  return blend(tensor, resampled, alpha);
}
register("convFeedback", convFeedback, { iterations: 50, alpha: 0.5 });

export function posterize(tensor, shape, time, speed, levels = 9) {
  if (levels <= 0) return tensor;
  const src = tensor.read();
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    out[i] = Math.floor(src[i] * levels + (1 / levels) * 0.5) / levels;
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register("posterize", posterize, { levels: 9 });

export function smoothstep(tensor, shape, time, speed, a = 0, b = 1) {
  const src = tensor.read();
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
  if (alpha < 1) out = blend(tensor, out, alpha);
  return out;
}
register("convolve", convolve, {
  kernel: ValueMask.conv2d_blur,
  withNormalize: true,
  alpha: 1,
});

export function fbm(
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
register("fbm", fbm, { freq: 4, octaves: 4, lacunarity: 2, gain: 0.5 });

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
    out[i * 3] =
      p.offset[0] +
      p.amp[0] * Math.cos(TAU * (p.freq[0] * t * 0.875 + 0.0625 + p.phase[0]));
    out[i * 3 + 1] =
      p.offset[1] +
      p.amp[1] * Math.cos(TAU * (p.freq[1] * t * 0.875 + 0.0625 + p.phase[1]));
    out[i * 3 + 2] =
      p.offset[2] +
      p.amp[2] * Math.cos(TAU * (p.freq[2] * t * 0.875 + 0.0625 + p.phase[2]));
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, 3]);
}
register("palette", palette, { name: null });

export function invert(tensor, shape, time, speed) {
  const src = tensor.read();
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    out[i] = 1 - src[i];
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register("invert", invert, {});

export function vortex(tensor, shape, time, speed, displacement = 64) {
  const [h, w, c] = shape;
  const centerX = w / 2;
  const centerY = h / 2;
  const xArr = new Float32Array(h * w);
  const yArr = new Float32Array(h * w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = x - centerX;
      const dy = y - centerY;
      const dist = Math.sqrt(dx * dx + dy * dy) + 1e-6;
      const fade = 1 - Math.max(Math.abs(dx) / centerX, Math.abs(dy) / centerY);
      const nx = (-dy / dist) * fade;
      const ny = (dx / dist) * fade;
      const idx = y * w + x;
      xArr[idx] = nx * 0.5 + 0.5;
      yArr[idx] = ny * 0.5 + 0.5;
    }
  }
  const xTensor = Tensor.fromArray(tensor.ctx, xArr, [h, w, 1]);
  const yTensor = Tensor.fromArray(tensor.ctx, yArr, [h, w, 1]);
  const disp = simplexRandom(time, undefined, speed) * 100 * displacement;
  return refract(tensor, xTensor, yTensor, disp);
}
register("vortex", vortex, { displacement: 64 });

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
register("aberration", aberration, { displacement: 0.005 });

export function glitch(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const base = values(4, [h, w, 1], { time, speed: speed * 50 });
  const noise = base.read();
  const src = tensor.read();
  const out = new Float32Array(h * w * c);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const shiftAmt = Math.floor(noise[idx] * 4);
      for (let k = 0; k < c; k++) {
        let sx = x;
        if (k === 0) sx = (x + shiftAmt) % w;
        else if (k === 2) sx = (x - shiftAmt + w) % w;
        const srcBase = (y * w + sx) * c + k;
        out[idx * c + k] = src[srcBase];
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register("glitch", glitch, {});

export function vhs(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const scanNoise = values(Math.floor(h * 0.5) + 1, [h, w, 1], {
    time,
    speed: speed * 100,
  }).read();
  const gradNoise = values(5, [h, w, 1], { time, speed }).read();
  const src = tensor.read();
  const blended = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    let g = gradNoise[i] - 0.5;
    if (g < 0) g = 0;
    g = Math.min(g * 2, 1);
    const noise = scanNoise[i];
    for (let k = 0; k < c; k++) {
      blended[i * c + k] = src[i * c + k] * (1 - g) + noise * g;
    }
  }
  const out = new Float32Array(h * w * c);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      let g = gradNoise[idx] - 0.5;
      if (g < 0) g = 0;
      g = Math.min(g * 2, 1);
      const xOff = Math.floor(scanNoise[idx] * w * g * g);
      const srcX = (x - xOff + w) % w;
      for (let k = 0; k < c; k++) {
        out[idx * c + k] = blended[(y * w + srcX) * c + k];
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register("vhs", vhs, {});

export function lensWarp(tensor, shape, time, speed, displacement = 0.0625) {
  const valueShape = [shape[0], shape[1], 1];
  const mask = singularity(null, valueShape, time, speed).read();
  for (let i = 0; i < mask.length; i++) mask[i] = mask[i] ** 5;
  const noise = values(2, valueShape, {
    ctx: tensor.ctx,
    time,
    speed,
    splineOrder: 2,
  }).read();
  for (let i = 0; i < noise.length; i++) {
    noise[i] = (noise[i] * 2 - 1) * mask[i];
  }
  const distortion = Tensor.fromArray(tensor.ctx, noise, valueShape);
  return refract(tensor, distortion, null, displacement);
}
register("lens_warp", lensWarp, { displacement: 0.0625 });

export function lensDistortion(tensor, shape, time, speed, displacement = 1) {
  const [h, w, c] = shape;
  const src = tensor.read();
  const out = new Float32Array(h * w * c);
  const maxDist = Math.sqrt(0.5 * 0.5 + 0.5 * 0.5) || 1;
  const zoom = displacement < 0 ? displacement * -0.25 : 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const xIndex = x / w;
      const yIndex = y / h;
      const xDist = xIndex - 0.5;
      const yDist = yIndex - 0.5;
      const centerDist = 1 - distance(xDist, yDist) / maxDist;
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
      const xi = ((Math.floor(xOff) % w) + w) % w;
      const yi = ((Math.floor(yOff) % h) + h) % h;
      const srcIdx = (yi * w + xi) * c;
      const dstIdx = (y * w + x) * c;
      for (let k = 0; k < c; k++) {
        out[dstIdx + k] = src[srcIdx + k] || 0;
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register("lens_distortion", lensDistortion, { displacement: 1 });

export function degauss(tensor, shape, time, speed, displacement = 0.0625) {
  const [h, w, c] = shape;
  const channelShape = [h, w, 1];
  const src = tensor.read();
  const out = new Float32Array(h * w * c);
  const channels = Math.min(3, c);
  for (let k = 0; k < channels; k++) {
    const channelData = new Float32Array(h * w);
    for (let i = 0; i < h * w; i++) channelData[i] = src[i * c + k] || 0;
    const channelTensor = Tensor.fromArray(
      tensor.ctx,
      channelData,
      channelShape,
    );
    const warped = lensWarp(
      channelTensor,
      channelShape,
      time,
      speed,
      displacement,
    ).read();
    for (let i = 0; i < h * w; i++) out[i * c + k] = warped[i];
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register("degauss", degauss, { displacement: 0.0625 });

export function scanlineError(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const errorFreq = Math.floor(h * 0.5) || 1;
  let errorLine = values(errorFreq, [h, w, 1], {
    time,
    speed: speed * 10,
    distrib: ValueDistribution.exp,
  }).read();
  let errorSwerve = values(Math.floor(h * 0.01) || 1, [h, w, 1], {
    time,
    speed,
    distrib: ValueDistribution.exp,
  }).read();
  const whiteNoise = values(errorFreq, [h, w, 1], {
    time,
    speed: speed * 100,
  }).read();
  const src = tensor.read();
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
register("scanlineError", scanlineError, {});

export function crt(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const scan = values(Math.floor(h * 0.5) || 1, [h, w, 1], {
    time,
    speed: speed * 0.1,
  }).read();
  const src = tensor.read();
  const blended = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    const s = scan[i];
    for (let k = 0; k < c; k++) {
      const t = src[i * c + k];
      blended[i * c + k] = t * 0.95 + (t + s) * s * 0.05;
    }
  }
  let outTensor = clamp01(Tensor.fromArray(tensor.ctx, blended, shape));
  if (c === 3) {
    outTensor = aberration(
      outTensor,
      shape,
      time,
      speed,
      0.0125 + random() * 0.00625,
    );
    outTensor = randomHue(outTensor, shape, time, speed, 0.125);
    outTensor = saturation(outTensor, shape, time, speed, 1.125);
  }
  const vigAlpha = random() * 0.175;
  const vigData = outTensor.read();
  const cx = (w - 1) / 2;
  const cy = (h - 1) / 2;
  const maxDist = Math.sqrt(cx * cx + cy * cy) || 1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy) / maxDist;
      const vig = dist * dist * vigAlpha;
      for (let k = 0; k < c; k++) {
        const idx = (y * w + x) * c + k;
        vigData[idx] = vigData[idx] * (1 - vig);
      }
    }
  }
  outTensor = Tensor.fromArray(outTensor.ctx, vigData, shape);
  const data = outTensor.read();
  let mean = 0;
  for (let i = 0; i < data.length; i++) mean += data[i];
  mean /= data.length;
  for (let i = 0; i < data.length; i++) {
    data[i] = (data[i] - mean) * 1.25 + mean;
  }
  return Tensor.fromArray(outTensor.ctx, data, shape);
}
register("crt", crt, {});

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
    gl.uniform1i(gl.getUniformLocation(prog, "u_tex"), 0);
    gl.uniform1f(gl.getUniformLocation(prog, "u_disp"), displacement);
    gl.uniform1f(gl.getUniformLocation(prog, "u_mod"), Math.min(h, w));
    gl.uniform1f(gl.getUniformLocation(prog, "u_channels"), c);
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
register("reindex", reindex, { displacement: 0.5 });

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
    const refTensor =
      reference || values(freq, [h, w, 1], { ctx, time, speed, splineOrder });
    const refTex =
      refTensor.ctx === ctx
        ? refTensor
        : Tensor.fromArray(ctx, refTensor.read(), refTensor.shape);
    const gl = ctx.gl;
    const rand = simplexRandom(time, undefined, speed);
    const fs = `#version 300 es\nprecision highp float;\nuniform sampler2D u_tex;\nuniform sampler2D u_ref;\nuniform float u_disp;\nuniform float u_kink;\nuniform float u_rand;\nout vec4 outColor;\nvoid main(){\n vec2 res = vec2(${w}.0, ${h}.0);\n vec2 uv = gl_FragCoord.xy / res;\n float ref = texture(u_ref, uv).r;\n float ang = ref * ${TAU} * u_kink * u_rand;\n vec2 offset = vec2(cos(ang), sin(ang)) * u_disp;\n vec2 uv2 = fract(uv + offset);\n outColor = texture(u_tex, uv2);\n}`;
    const prog = ctx.createProgram(FULLSCREEN_VS, fs);
    const pp = ctx.pingPong(w, h);
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tensor.handle);
    gl.uniform1i(gl.getUniformLocation(prog, "u_tex"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, refTex.handle);
    gl.uniform1i(gl.getUniformLocation(prog, "u_ref"), 1);
    gl.uniform1f(gl.getUniformLocation(prog, "u_disp"), displacement);
    gl.uniform1f(gl.getUniformLocation(prog, "u_kink"), kink);
    gl.uniform1f(gl.getUniformLocation(prog, "u_rand"), rand);
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
    gl.uniform1i(gl.getUniformLocation(prog, "u_tex"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, clut.handle);
    gl.uniform1i(gl.getUniformLocation(prog, "u_clut"), 1);
    gl.uniform1f(gl.getUniformLocation(prog, "u_disp"), displacement);
    gl.uniform1f(
      gl.getUniformLocation(prog, "u_horizontal"),
      horizontal ? 1 : 0,
    );
    gl.uniform1f(gl.getUniformLocation(prog, "u_channels"), c);
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
      const sx = Math.floor((xi * cw) / w);
      const sy = Math.floor((yi * ch) / h);
      const srcIdx = (sy * cw + sx) * cc;
      const outIdx = (y * w + x) * cc;
      for (let k = 0; k < cc; k++) {
        out[outIdx + k] = clutData[srcIdx + k];
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, cc]);
}
register("colorMap", colorMap, {
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
register("falseColor", falseColor, { horizontal: false, displacement: 0.5 });

export function tint(tensor, shape, time, speed, alpha = 0.5) {
  const [h, w, c] = shape;
  if (c < 3) return tensor;
  // Consume similar noise to maintain randomness parity with Python impl
  values(3, shape, { ctx: tensor.ctx, time, speed, corners: true });

  const src = tensor.read();
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

  const rand1 = random() * 0.333;
  const rand2 = random();
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
  const hsv = rgbToHsv(baseTensor).read();
  const hsvMix = new Float32Array(h * w * 3);
  for (let i = 0; i < h * w; i++) {
    hsvMix[i * 3] = colorData[i * 3];
    hsvMix[i * 3 + 1] = colorData[i * 3 + 1];
    hsvMix[i * 3 + 2] = hsv[i * 3 + 2];
  }
  const colorized = hsvToRgb(Tensor.fromArray(tensor.ctx, hsvMix, [h, w, 3]));
  let out = blend(baseTensor, colorized, alpha);

  if (c === 4) {
    const outData = out.read();
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

export function valueRefract(
  tensor,
  shape,
  time,
  speed,
  freq = 4,
  distrib = ValueDistribution.center_circle,
  displacement = 0.125,
) {
  const valueShape = [shape[0], shape[1], 1];
  const blendValues = values(freq, valueShape, {
    ctx: tensor.ctx,
    distrib,
    time,
    speed,
  });
  return refract(tensor, blendValues, null, displacement);
}
register("valueRefract", valueRefract, {
  freq: 4,
  distrib: ValueDistribution.center_circle,
  displacement: 0.125,
});

export function refractEffect(
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
      const src = tensor.read();
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
    rx = convolution(gray, kx, { normalize: true });
    ry = convolution(gray, ky, { normalize: true });
  } else if (warpFreq !== null && warpFreq !== undefined) {
    rx = values(warpFreq, valueShape, {
      ctx: tensor.ctx,
      distrib: ValueDistribution.uniform,
      time,
      speed,
      splineOrder,
    });
    ry = values(warpFreq, valueShape, {
      ctx: tensor.ctx,
      distrib: ValueDistribution.uniform,
      time,
      speed,
      splineOrder,
    });
  } else {
    if (!rx) rx = tensor;
    if (!ry) {
      if (yFromOffset) {
        ry = offsetTensor(rx, Math.floor(w * 0.5), Math.floor(h * 0.5));
      } else {
        const rData = rx.read();
        const cx = new Float32Array(rData.length);
        const cy = new Float32Array(rData.length);
        for (let i = 0; i < rData.length; i++) {
          const ang = rData[i] * TAU;
          cx[i] = Math.cos(ang);
          cy[i] = Math.sin(ang);
        }
        rx = Tensor.fromArray(tensor.ctx, cx, valueShape);
        ry = Tensor.fromArray(tensor.ctx, cy, valueShape);
      }
    }
  }
  const src = tensor.read();
  const rxData = rx.read();
  const ryData = ry.read();
  const out = new Float32Array(h * w * c);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      let vx = rxData[idx];
      let vy = ryData[idx];
      if (signedRange && !fromDerivative) {
        vx = vx * 2 - 1;
        vy = vy * 2 - 1;
      } else {
        vx *= 2;
        vy *= 2;
      }
      const xOff = x + vx * displacement * w;
      const yOff = y + vy * displacement * h;
      const x0 = Math.floor(xOff);
      const y0 = Math.floor(yOff);
      const x1 = x0 + 1;
      const y1 = y0 + 1;
      const fx = xOff - x0;
      const fy = yOff - y0;
      for (let k = 0; k < c; k++) {
        const s00 =
          src[((((y0 % h) + h) % h) * w + (((x0 % w) + w) % w)) * c + k];
        const s10 =
          src[((((y0 % h) + h) % h) * w + (((x1 % w) + w) % w)) * c + k];
        const s01 =
          src[((((y1 % h) + h) % h) * w + (((x0 % w) + w) % w)) * c + k];
        const s11 =
          src[((((y1 % h) + h) % h) * w + (((x1 % w) + w) % w)) * c + k];
        const x_y0 = s00 * (1 - fx) + s10 * fx;
        const x_y1 = s01 * (1 - fx) + s11 * fx;
        out[(y * w + x) * c + k] = x_y0 * (1 - fy) + x_y1 * fy;
      }
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [h, w, c]);
}
register("refractEffect", refractEffect, {
  displacement: 0.5,
  referenceX: null,
  referenceY: null,
  warpFreq: null,
  splineOrder: InterpolationType.bicubic,
  fromDerivative: false,
  signedRange: true,
  yFromOffset: false,
});

export function fxaaEffect(tensor, shape, time, speed) {
  return fxaa(tensor);
}
register("fxaaEffect", fxaaEffect, {});

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

function offsetTensor(tensor, xOff, yOff) {
  const [h, w, c] = tensor.shape;
  if (xOff === 0 && yOff === 0) return tensor;
  const src = tensor.read();
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

function centerMask(
  center,
  edges,
  shape,
  power = 2,
  distMetric = DistanceMetric.chebyshev,
) {
  const [h, w] = shape;
  const maskVal = Math.pow(0.5, 2 / power);
  const maskData = new Float32Array(h * w);
  maskData.fill(maskVal);
  const mask = Tensor.fromArray(center.ctx, maskData, [h, w, 1]);
  return blend(center, edges, mask);
}

function voronoiColorRegions(tensor, shape, xPts, yPts) {
  const [h, w, c] = shape;
  const count = xPts.length;
  const h2 = Math.max(1, Math.floor(h * 0.5));
  const w2 = Math.max(1, Math.floor(w * 0.5));
  const index = new Int32Array(h2 * w2);
  for (let y = 0; y < h2; y++) {
    for (let x = 0; x < w2; x++) {
      const ox = (x + 0.5) * (w / w2) - 0.5;
      const oy = (y + 0.5) * (h / h2) - 0.5;
      let best = 0;
      let bestDist = Infinity;
      for (let i = 0; i < count; i++) {
        let dx = Math.abs(ox - xPts[i]);
        dx = Math.min(dx, w - dx);
        let dy = Math.abs(oy - yPts[i]);
        dy = Math.min(dy, h - dy);
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < bestDist) {
          bestDist = dist;
          best = i;
        }
      }
      index[y * w2 + x] = best;
    }
  }
  const upIndex = new Int32Array(h * w);
  for (let y = 0; y < h; y++) {
    const sy = Math.floor((y * h2) / h);
    for (let x = 0; x < w; x++) {
      const sx = Math.floor((x * w2) / w);
      upIndex[y * w + x] = index[sy * w2 + sx];
    }
  }
  const src = tensor.read();
  const out = new Float32Array(h * w * c);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = upIndex[y * w + x];
      const px = xPts[idx];
      const py = yPts[idx];
      const srcIdx = (py * w + px) * c;
      const dstIdx = (y * w + x) * c;
      for (let k = 0; k < c; k++) out[dstIdx + k] = src[srcIdx + k];
    }
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
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
  xyBlend = 0,
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
    const exposure =
      iterations > 1 ? 1 - Math.abs(1 - (iter / (iterations - 1)) * 2) : 1;
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
    tensor = blend(
      shadow(tensor, shape, time, speed),
      reindex(tensor, shape, time, speed, 1),
      mask,
    );
  }
  return blend(tensor, outTensor, alpha);
}
register("erosionWorms", erosionWorms, {
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
  colors = null,
) {
  const [h, w, c] = shape;
  const count = Math.floor(Math.max(w, h) * density);
  const wormsY = new Float32Array(count);
  const wormsX = new Float32Array(count);
  const wormsStride = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    wormsY[i] = random() * (h - 1);
    wormsX[i] = random() * (w - 1);
    wormsStride[i] =
      randomNormal(stride, strideDeviation) * (Math.max(w, h) / 1024.0);
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
      for (let i = 0; i < n; i++)
        rot[i] = periodicValue(time * speed, random());
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
  outTensor = normalize(outTensor);
  const d = outTensor.read();
  for (let i = 0; i < d.length; i++) d[i] = Math.sqrt(d[i]);
  outTensor = Tensor.fromArray(outTensor.ctx, d, shape);
  return blend(tensor, outTensor, alpha);
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

export function wormhole(
  tensor,
  shape,
  time,
  speed,
  kink = 1.0,
  inputStride = 1.0,
  alpha = 1.0,
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
      xArr[idx] = Math.floor(x + xo) % w;
      yArr[idx] = Math.floor(y + yo) % h;
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
register("wormhole", wormhole, { kink: 1.0, inputStride: 1.0, alpha: 1.0 });

export function vignette(
  tensor,
  shape,
  time,
  speed,
  brightness = 0.0,
  alpha = 1.0,
) {
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
    gl.uniform1i(gl.getUniformLocation(prog, "u_tex"), 0);
    gl.uniform1f(gl.getUniformLocation(prog, "u_brightness"), brightness);
    gl.uniform1f(gl.getUniformLocation(prog, "u_alpha"), alpha);
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
register("vignette", vignette, { brightness: 0.0, alpha: 1.0 });

export function vaseline(tensor, shape, time, speed, alpha = 1.0) {
  const blurred = bloom(tensor, shape, time, speed, 1.0);
  const masked = centerMask(tensor, blurred, shape);
  return blend(tensor, masked, alpha);
}
register("vaseline", vaseline, { alpha: 1.0 });

export function lightLeak(tensor, shape, time, speed, alpha = 0.25) {
  const gridMembers = [
    PointDistribution.square,
    PointDistribution.waffle,
    PointDistribution.chess,
    PointDistribution.h_hex,
    PointDistribution.v_hex,
  ];
  const distrib = gridMembers[randomInt(0, gridMembers.length)];
  const [xPts, yPts] = pointCloud(6, {
    distrib,
    drift: 0.05,
    shape,
    time,
    speed,
  });
  let leak = voronoiColorRegions(tensor, shape, xPts, yPts);
  leak = wormhole(leak, shape, time, speed, 1.0, 0.25, 1.0);
  leak = bloom(leak, shape, time, speed, 1.0);
  const src = tensor.read();
  const leakData = leak.read();
  const screened = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    screened[i] = 1 - (1 - src[i]) * (1 - leakData[i]);
  }
  leak = Tensor.fromArray(tensor.ctx, screened, shape);
  leak = centerMask(tensor, leak, shape, 4);
  const blended = blend(tensor, leak, alpha);
  return vaseline(blended, shape, time, speed, alpha);
}
register("lightLeak", lightLeak, { alpha: 0.25 });

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
    gl.uniform1i(gl.getUniformLocation(prog, "u_tex"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, noise.handle);
    gl.uniform1i(gl.getUniformLocation(prog, "u_noise"), 1);
    gl.uniform1f(gl.getUniformLocation(prog, "u_levels"), levels);
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
register("dither", dither, { levels: 2 });

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
register("grain", grain, { alpha: 0.25 });

export function snow(tensor, shape, time, speed, alpha = 0.25) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const staticNoise = values(Math.max(h, w), shape, {
    ctx,
    time,
    speed: speed * 100,
    splineOrder: 0,
  });
  let limiter = values(Math.max(h, w), shape, {
    ctx,
    time,
    speed: speed * 100,
    distrib: ValueDistribution.exp,
    splineOrder: 0,
  });
  const lData = limiter.read();
  for (let i = 0; i < lData.length; i++) lData[i] *= alpha;
  limiter = Tensor.fromArray(ctx, lData, shape);
  return blend(tensor, staticNoise, limiter);
}
register("snow", snow, { alpha: 0.25 });

export function saturation(tensor, shape, time, speed, amount = 0.75) {
  if (shape[2] !== 3) return tensor;
  const hsv = rgbToHsv(tensor);
  const data = hsv.read();
  for (let i = 0; i < shape[0] * shape[1]; i++) {
    data[i * 3 + 1] = Math.min(1, Math.max(0, data[i * 3 + 1] * amount));
  }
  return hsvToRgb(Tensor.fromArray(tensor.ctx, data, hsv.shape));
}
register("saturation", saturation, { amount: 0.75 });

export function randomHue(tensor, shape, time, speed, range = 0.05) {
  const shift = random() * range * 2 - range;
  return adjustHue(tensor, shift);
}
register("randomHue", randomHue, { range: 0.05 });

export function normalizeEffect(tensor, shape, time, speed) {
  return normalize(tensor);
}
register("normalize", normalizeEffect, {});

export function adjustBrightness(tensor, shape, time, speed, amount = 0) {
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
    gl.uniform1i(gl.getUniformLocation(prog, "u_tex"), 0);
    gl.uniform1f(gl.getUniformLocation(prog, "u_amount"), amount);
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
register("adjustBrightness", adjustBrightness, { amount: 0 });

export function adjustContrast(tensor, shape, time, speed, amount = 1) {
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
    gl.uniform1i(gl.getUniformLocation(prog, "u_tex"), 0);
    gl.uniform1f(gl.getUniformLocation(prog, "u_amount"), amount);
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
register("adjustContrast", adjustContrast, { amount: 1 });

export function adjustHueEffect(tensor, shape, time, speed, amount = 0.25) {
  if (shape[2] !== 3 || amount === 0 || amount === 1 || amount === null)
    return tensor;
  return adjustHue(tensor, amount);
}
register("adjustHue", adjustHueEffect, { amount: 0.25 });

export function ridgeEffect(tensor, shape, time, speed) {
  return ridge(tensor);
}
register("ridge", ridgeEffect, {});

export function sine(tensor, shape, time, speed, amount = 1.0, rgb = false) {
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
register("sine", sine, { amount: 1.0, rgb: false });

export function blur(
  tensor,
  shape,
  time,
  speed,
  amount = 10.0,
  splineOrder = InterpolationType.bicubic,
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
register("blur", blur, {
  amount: 10.0,
  splineOrder: InterpolationType.bicubic,
});

export function wobble(tensor, shape, time, speed) {
  const xOffset = Math.floor(
    simplexRandom(time, undefined, speed * 0.5) * shape[1],
  );
  const yOffset = Math.floor(
    simplexRandom(time, undefined, speed * 0.5) * shape[0],
  );
  return offsetTensor(tensor, xOffset, yOffset);
}
register("wobble", wobble, {});

export function reverb(
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
  const reference = ridges ? ridge(tensor) : tensor;
  const base = reference.read();
  const outData = base.slice();
  for (let i = 0; i < iterations; i++) {
    for (let octave = 1; octave <= octaves; octave++) {
      const mult = 2 ** octave;
      const nh = Math.floor(h / mult);
      const nw = Math.floor(w / mult);
      if (nh === 0 || nw === 0) break;
      let layer = downsample(reference, mult);
      layer = upsample(layer, mult);
      const layerData = layer.read();
      for (let j = 0; j < outData.length; j++) {
        outData[j] += layerData[j] / mult;
      }
    }
  }
  const outTensor = Tensor.fromArray(tensor.ctx, outData, shape);
  return normalize(outTensor);
}
register("reverb", reverb, { octaves: 2, iterations: 1, ridges: true });

function expandTile(tensor, inputShape, outputShape) {
  const [ih, iw, c] = inputShape;
  const [oh, ow] = outputShape;
  const src = tensor.read();
  const out = new Float32Array(oh * ow * c);
  const xOff = Math.floor(iw / 2);
  const yOff = Math.floor(ih / 2);
  for (let y = 0; y < oh; y++) {
    const sy = (yOff + y) % ih;
    for (let x = 0; x < ow; x++) {
      const sx = (xOff + x) % iw;
      const srcBase = (sy * iw + sx) * c;
      const dstBase = (y * ow + x) * c;
      for (let k = 0; k < c; k++) out[dstBase + k] = src[srcBase + k];
    }
  }
  return Tensor.fromArray(tensor.ctx, out, [oh, ow, c]);
}

function resizeWithCropOrPad(tensor, shape, size) {
  const [h, w, c] = shape;
  const src = tensor.read();
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

function rotate2D(tensor, shape, angle) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  if (ctx && !ctx.isCPU) {
    const gl = ctx.gl;
    const fs = `#version 300 es\nprecision highp float;\nuniform sampler2D u_tex;\nuniform float u_angle;\nout vec4 outColor;\nvoid main(){\n vec2 uv = gl_FragCoord.xy / vec2(${w}.0, ${h}.0);\n uv -= 0.5;\n float c = cos(u_angle);\n float s = sin(u_angle);\n uv = vec2(c * uv.x + s * uv.y, -s * uv.x + c * uv.y) + 0.5;\n uv = fract(uv);\n outColor = texture(u_tex, uv);\n}`;
    const prog = ctx.createProgram(FULLSCREEN_VS, fs);
    const pp = ctx.pingPong(w, h);
    gl.useProgram(prog);
    ctx.bindTexture(prog, "u_tex", tensor.tex);
    gl.uniform1f(gl.getUniformLocation(prog, "u_angle"), angle);
    ctx.bindFramebuffer(pp.writeFbo, w, h);
    ctx.drawQuad();
    return new Tensor(ctx, pp.writeTex, [h, w, c]);
  }
  const src = tensor.read();
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

function cropTensor(tensor, inputShape, outputShape) {
  const [H, W, c] = inputShape;
  const [h, w] = outputShape;
  const src = tensor.read();
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

function resizeBilinear(tensor, size) {
  const [h, w, c] = tensor.shape;
  const src = tensor.read();
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

export function squareCropAndResize(tensor, shape, length = 1024) {
  const [h, w, c] = shape;
  const have = Math.min(h, w);
  let out = tensor;
  if (h !== w) {
    out = cropTensor(tensor, shape, [have, have]);
  }
  if (have !== length) {
    out = resizeBilinear(out, length);
  }
  return out;
}

export function rotate(tensor, shape, time, speed, angle = null) {
  if (angle === null || angle === undefined) angle = random() * 360;
  const [h, w, c] = shape;
  const want = Math.max(h, w) * 2;
  let padded = expandTile(tensor, shape, [want, want, c]);
  padded = rotate2D(padded, [want, want, c], (angle * Math.PI) / 180);
  return cropTensor(padded, [want, want, c], shape);
}
register("rotate", rotate, { angle: 0 });

function _pixelSort(tensor, shape, angle, darkest) {
  const [h, w, c] = shape;
  let srcData = tensor.read();
  if (darkest) {
    const inv = new Float32Array(srcData.length);
    for (let i = 0; i < srcData.length; i++) inv[i] = 1 - srcData[i];
    srcData = inv;
  }
  let working = Tensor.fromArray(tensor.ctx, srcData, shape);
  const want = Math.max(h, w) * 2;
  working = resizeWithCropOrPad(working, shape, want);
  if (angle !== false) {
    working = rotate2D(working, [want, want, c], (angle * Math.PI) / 180);
  }
  const data = working.read();
  const sorted = new Float32Array(want * want * c);
  for (let y = 0; y < want; y++) {
    const brightness = new Float32Array(want);
    for (let x = 0; x < want; x++) {
      let b = 0;
      for (let k = 0; k < c; k++) b += data[(y * want + x) * c + k];
      brightness[x] = b / c;
    }
    let maxIdx = 0;
    for (let x = 1; x < want; x++)
      if (brightness[x] > brightness[maxIdx]) maxIdx = x;
    for (let k = 0; k < c; k++) {
      const channel = new Array(want);
      for (let x = 0; x < want; x++) channel[x] = data[(y * want + x) * c + k];
      channel.sort((a, b) => b - a);
      for (let x = 0; x < want; x++) {
        const xx = (x - maxIdx + want) % want;
        sorted[(y * want + x) * c + k] = channel[xx];
      }
    }
  }
  let sortedTensor = Tensor.fromArray(tensor.ctx, sorted, [want, want, c]);
  if (angle !== false) {
    sortedTensor = rotate2D(
      sortedTensor,
      [want, want, c],
      (-angle * Math.PI) / 180,
    );
  }
  sortedTensor = cropTensor(sortedTensor, [want, want, c], shape);
  const sortedData = sortedTensor.read();
  const out = new Float32Array(sortedData.length);
  for (let i = 0; i < sortedData.length; i++) {
    const v = Math.max(srcData[i], sortedData[i]);
    out[i] = darkest ? 1 - v : v;
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}

export function pixelSort(
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
register("pixelSort", pixelSort, { angled: false, darkest: false });

export function glyphMap(
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
  let glyphShape;
  let glyphs;
  if (mask === ValueMask.truetype) {
    glyphShape = [15, 15, 1];
    glyphs = loadGlyphs(glyphShape);
  } else {
    glyphShape = maskShape(mask);
    const [g] = maskValues(mask, glyphShape);
    const data = g.read();
    const gh = glyphShape[0];
    const gw = glyphShape[1];
    const gc = glyphShape[2];
    const glyph = [];
    for (let y = 0; y < gh; y++) {
      const row = [];
      for (let x = 0; x < gw; x++) {
        row.push([data[(y * gw + x) * gc]]);
      }
      glyph.push(row);
    }
    glyphs = [glyph];
  }
  if (!glyphs.length) return tensor;
  const [h, w, c] = shape;
  const gh = glyphShape[0];
  const gw = glyphShape[1];
  const uvH = Math.max(1, Math.floor(h / gh));
  const uvW = Math.max(1, Math.floor(w / gw));
  const src = tensor.read();
  const out = new Float32Array(h * w * c);
  for (let cy = 0; cy < uvH; cy++) {
    for (let cx = 0; cx < uvW; cx++) {
      const sy = Math.min(h - 1, cy * gh);
      const sx = Math.min(w - 1, cx * gw);
      let bright;
      if (c === 1) {
        bright = src[sy * w + sx];
      } else {
        const base = (sy * w + sx) * c;
        const r = src[base];
        const gVal = src[base + 1] || 0;
        const b = src[base + 2] || 0;
        bright = 0.2126 * r + 0.7152 * gVal + 0.0722 * b;
      }
      const gIndex = Math.min(
        glyphs.length - 1,
        Math.floor(bright * glyphs.length),
      );
      const glyph = glyphs[gIndex];
      for (let gy = 0; gy < gh; gy++) {
        for (let gx = 0; gx < gw; gx++) {
          const yy = cy * gh + gy;
          const xx = cx * gw + gx;
          if (yy >= h || xx >= w) continue;
          const gv = glyph[gy][gx][0];
          const outBase = (yy * w + xx) * c;
          if (!colorize) {
            for (let k = 0; k < c; k++) out[outBase + k] = gv;
          } else {
            const srcBase = (sy * w + sx) * c;
            for (let k = 0; k < c; k++)
              out[outBase + k] = gv * src[srcBase + k];
          }
        }
      }
    }
  }
  let outTensor = Tensor.fromArray(tensor.ctx, out, shape);
  if (alpha !== 1) outTensor = blend(tensor, outTensor, alpha);
  return outTensor;
}
register("glyphMap", glyphMap, {
  mask: ValueMask.truetype,
  colorize: true,
  zoom: 1,
  alpha: 1,
  splineOrder: InterpolationType.constant,
});

export function dla(
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
  const kData = kernelTensor.read();
  const kernel = [];
  for (let i = 0; i < 5; i++) {
    kernel.push(Array.from(kData.slice(i * 5, i * 5 + 5)));
  }
  const convolved = convolution(scattered, kernel);
  const convData = convolved.read();
  const tensorData = tensor.read();
  const mult = new Float32Array(convData.length);
  for (let i = 0; i < convData.length; i++) {
    mult[i] = convData[i] * tensorData[i];
  }
  const out = Tensor.fromArray(tensor.ctx, mult, shape);
  return blend(tensor, out, alpha);
}
register("dla", dla, {
  padding: 2,
  seedDensity: 0.01,
  density: 0.125,
  xy: null,
  alpha: 1,
});

export function simpleFrame(tensor, shape, time, speed, brightness = 0) {
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
  border = blend(
    Tensor.fromArray(tensor.ctx, new Float32Array(h * w), [h, w, 1]),
    border,
    0.55,
  );
  border = posterize(border, [h, w, 1], time, speed, 1);
  const maskC = new Float32Array(h * w * c);
  const bData = border.read();
  for (let i = 0; i < h * w; i++) {
    for (let k = 0; k < c; k++) maskC[i * c + k] = bData[i];
  }
  const maskTensor = Tensor.fromArray(tensor.ctx, maskC, shape);
  const bright = new Float32Array(h * w * c);
  bright.fill(brightness);
  const brightTensor = Tensor.fromArray(tensor.ctx, bright, shape);
  return blend(tensor, brightTensor, maskTensor);
}
register("simpleFrame", simpleFrame, { brightness: 0 });

export function frame(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const halfH = Math.max(1, Math.floor(h * 0.5));
  const halfW = Math.max(1, Math.floor(w * 0.5));
  const halfShape = [halfH, halfW, c];
  const noise = values(64, halfShape, { seed: 0, time });
  const nData = noise.read();
  const cx = (halfW - 1) / 2;
  const cy = (halfH - 1) / 2;
  const maskData = new Float32Array(halfH * halfW);
  for (let y = 0; y < halfH; y++) {
    for (let x = 0; x < halfW; x++) {
      const dx = Math.abs(x - cx) / (cx || 1);
      const dy = Math.abs(y - cy) / (cy || 1);
      let m = 1 - Math.max(dx, dy);
      m = Math.max(0, Math.min(1, m + nData[y * halfW + x] * 0.005));
      maskData[y * halfW + x] = Math.sqrt(m);
    }
  }
  const maskC = new Float32Array(halfH * halfW * c);
  for (let i = 0; i < halfH * halfW; i++) {
    for (let k = 0; k < c; k++) maskC[i * c + k] = maskData[i];
  }
  const maskTensor = Tensor.fromArray(tensor.ctx, maskC, halfShape);
  let faded = downsample(tensor, 2);
  faded = adjustBrightness(faded, halfShape, time, speed, 0.1);
  faded = adjustContrast(faded, halfShape, time, speed, 0.75);
  if (halfH > 1 && halfW > 1) {
    faded = lightLeak(faded, halfShape, time, speed, 0.125);
    faded = vignette(faded, halfShape, time, speed, 0.05, 0.75);
  }
  const shade = shadow(noise, [halfH, halfW, 1], time, speed, 1.0).read();
  const edgeData = new Float32Array(halfH * halfW * c);
  for (let i = 0; i < halfH * halfW; i++) {
    for (let k = 0; k < c; k++) edgeData[i * c + k] = 0.9 + shade[i] * 0.1;
  }
  const edgeTex = Tensor.fromArray(tensor.ctx, edgeData, halfShape);
  let out = blend(faded, edgeTex, maskTensor);
  out = aberration(out, halfShape, time, speed, 0.00666);
  out = upsample(out, 2);
  return out;
}
register("frame", frame, {});

export function sketch(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  let valuesTensor = tensor;
  if (c !== 1) {
    const src = tensor.read();
    const gray = new Float32Array(h * w);
    for (let i = 0; i < h * w; i++) {
      const base = i * c;
      const r = src[base];
      const g = src[base + 1] || 0;
      const b = src[base + 2] || 0;
      gray[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }
    valuesTensor = Tensor.fromArray(tensor.ctx, gray, [h, w, 1]);
  }
  valuesTensor = adjustContrast(valuesTensor, [h, w, 1], time, speed, 2.0);
  valuesTensor = clamp01(valuesTensor);
  let outline = derivative(valuesTensor, [h, w, 1], time, speed).read();
  const invValues = valuesTensor.read();
  const invData = new Float32Array(invValues.length);
  for (let i = 0; i < invValues.length; i++) invData[i] = 1 - invValues[i];
  const d2 = derivative(
    Tensor.fromArray(tensor.ctx, invData, [h, w, 1]),
    [h, w, 1],
    time,
    speed,
  ).read();
  for (let i = 0; i < outline.length; i++) {
    outline[i] = Math.min(1 - outline[i], 1 - d2[i]);
  }
  let outlineTensor = Tensor.fromArray(tensor.ctx, outline, [h, w, 1]);
  outlineTensor = adjustContrast(outlineTensor, [h, w, 1], time, speed, 0.25);
  outlineTensor = normalize(outlineTensor);
  valuesTensor = vignette(valuesTensor, [h, w, 1], time, speed, 1.0, 0.875);
  const invValTensor = Tensor.fromArray(tensor.ctx, invData, [h, w, 1]);
  let wormsOut = worms(
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
  ).read();
  for (let i = 0; i < wormsOut.length; i++) wormsOut[i] = 1 - wormsOut[i];
  let cross = Tensor.fromArray(tensor.ctx, wormsOut, [h, w, 1]);
  cross = normalize(cross);
  let combined = blend(cross, outlineTensor, 0.75);
  combined = warp(
    combined,
    [h, w, 1],
    time,
    speed,
    Math.max(1, Math.floor(Math.max(h, w) * 0.125)),
    1,
    0.0025,
  );
  const combData = combined.read();
  for (let i = 0; i < combData.length; i++) combData[i] *= combData[i];
  combined = Tensor.fromArray(tensor.ctx, combData, [h, w, 1]);
  if (c === 1) return combined;
  const out = new Float32Array(h * w * c);
  const src = combined.read();
  for (let i = 0; i < h * w; i++) {
    for (let k = 0; k < c; k++) out[i * c + k] = src[i];
  }
  return Tensor.fromArray(tensor.ctx, out, shape);
}
register("sketch", sketch, {});

export function nebula(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const valueShape = [h, w, 1];

  function simpleMultires(freq, octaves, distrib = ValueDistribution.uniform) {
    const out = new Float32Array(h * w);
    for (let octave = 1; octave <= octaves; octave++) {
      const mult = 2 ** octave;
      const baseFreq = Math.floor(freq * 0.5 * mult);
      if (baseFreq > h && baseFreq > w) break;
      let layer = values(baseFreq, valueShape, {
        ctx,
        time,
        speed,
        distrib,
        seed: octave,
      });
      layer = ridge(layer);
      const lData = layer.read();
      for (let i = 0; i < out.length; i++) out[i] += lData[i] / mult;
    }
    let min = Infinity;
    let max = -Infinity;
    for (const v of out) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min || 1;
    for (let i = 0; i < out.length; i++) out[i] = (out[i] - min) / range;
    return Tensor.fromArray(ctx, out, valueShape);
  }

  let overlay = simpleMultires(randomInt(3, 4), 6, ValueDistribution.exp);
  const subtractor = simpleMultires(randomInt(2, 4), 4);
  let oData = overlay.read();
  const sData = subtractor.read();
  for (let i = 0; i < oData.length; i++)
    oData[i] = (oData[i] - sData[i]) * 0.125;
  overlay = Tensor.fromArray(ctx, oData, valueShape);

  overlay = rotate(overlay, valueShape, time, speed, randomInt(-15, 15));
  const baseData = tensor.read();
  const ovData = overlay.read();
  for (let i = 0; i < h * w; i++) {
    const v = ovData[i];
    const mult = 1 - v;
    for (let k = 0; k < c; k++) baseData[i * c + k] *= mult;
  }

  if (c >= 3) {
    const color = values(3, shape, {
      ctx,
      time,
      speed,
      corners: true,
      seed: randomInt(0, 65536),
    }).read();
    const hsv = new Float32Array(h * w * 3);
    const off1 = random();
    const off2 = random();
    for (let i = 0; i < h * w; i++) {
      const v = Math.max(ovData[i], 0);
      hsv[i * 3] = (ovData[i] * 0.333 + off1 * 0.333 + off2) % 1;
      hsv[i * 3 + 1] = color[i * 3 + 1];
      hsv[i * 3 + 2] = v;
    }
    const rgb = hsvToRgb(Tensor.fromArray(ctx, hsv, [h, w, 3])).read();
    for (let i = 0; i < h * w; i++) {
      for (let k = 0; k < 3; k++) baseData[i * c + k] += rgb[i * 3 + k];
    }
  } else {
    for (let i = 0; i < h * w; i++) {
      const v = Math.max(ovData[i], 0);
      for (let k = 0; k < c; k++) baseData[i * c + k] += v;
    }
  }

  for (let i = 0; i < baseData.length; i++) {
    if (baseData[i] < 0) baseData[i] = 0;
    if (baseData[i] > 1) baseData[i] = 1;
  }

  return Tensor.fromArray(ctx, baseData, shape);
}
register("nebula", nebula, {});

export function spatter(tensor, shape, time, speed, color = true) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const valueShape = [h, w, 1];
  let smear = values(randomInt(3, 6), valueShape, {
    ctx,
    time,
    speed,
    distrib: ValueDistribution.exp,
  });
  smear = warp(
    smear,
    valueShape,
    time,
    speed,
    randomInt(2, 3),
    randomInt(1, 2),
    1 + random(),
  );
  let sp1 = values(randomInt(32, 64), valueShape, {
    ctx,
    time,
    speed,
    distrib: ValueDistribution.exp,
    splineOrder: InterpolationType.linear,
  });
  let d1 = sp1.read();
  for (let i = 0; i < d1.length; i++) {
    let v = d1[i] - 1.0;
    v = (v - 0.5) * 4 + 0.5;
    d1[i] = Math.min(1, Math.max(0, v));
  }
  sp1 = Tensor.fromArray(ctx, d1, valueShape);
  let smData = smear.read();
  let spData = sp1.read();
  for (let i = 0; i < smData.length; i++)
    smData[i] = Math.max(smData[i], spData[i]);
  smear = Tensor.fromArray(ctx, smData, valueShape);
  let sp2 = values(randomInt(150, 200), valueShape, {
    ctx,
    time,
    speed,
    distrib: ValueDistribution.exp,
    splineOrder: InterpolationType.linear,
  });
  let d2 = sp2.read();
  for (let i = 0; i < d2.length; i++) {
    let v = d2[i] - 1.25;
    v = (v - 0.5) * 4 + 0.5;
    d2[i] = Math.min(1, Math.max(0, v));
  }
  sp2 = Tensor.fromArray(ctx, d2, valueShape);
  smData = smear.read();
  spData = sp2.read();
  for (let i = 0; i < smData.length; i++)
    smData[i] = Math.max(smData[i], spData[i]);
  smear = Tensor.fromArray(ctx, smData, valueShape);
  const remover = values(randomInt(2, 3), valueShape, {
    ctx,
    time,
    speed,
    distrib: ValueDistribution.exp,
  });
  const remData = remover.read();
  smData = smear.read();
  for (let i = 0; i < smData.length; i++)
    smData[i] = Math.max(0, smData[i] - remData[i]);
  smear = Tensor.fromArray(ctx, smData, valueShape);
  let mask = normalize(smear);
  let maskData = mask.read();
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
      const hsv = rgbToHsv(base);
      const hsvData = hsv.read();
      const delta = random() - 0.5;
      for (let i = 0; i < h * w; i++) {
        hsvData[i * 3] = (hsvData[i * 3] + delta + 1) % 1;
      }
      overlay = hsvToRgb(Tensor.fromArray(ctx, hsvData, hsv.shape));
    }
  } else {
    overlay = values(1, shape, { ctx, distrib: ValueDistribution.ones });
  }
  return blend(tensor, overlay, alphaTensor);
}
register("spatter", spatter, { color: true });

export function clouds(tensor, shape, time, speed) {
  const [h, w] = shape;
  const ctx = tensor.ctx;
  const preH = Math.max(1, Math.floor(h * 0.25));
  const preW = Math.max(1, Math.floor(w * 0.25));
  const preShape = [preH, preW, 1];
  let control = values(randomInt(2, 4), preShape, {
    ctx,
    time,
    speed,
    distrib: ValueDistribution.exp,
    seed: randomInt(0, 1000),
  });
  control = warp(control, preShape, time, speed, 3, 2, 0.125);
  let shaded = offsetTensor(control, randomInt(-15, 15), randomInt(-15, 15));
  let shadeData = shaded.read();
  for (let i = 0; i < shadeData.length; i++)
    shadeData[i] = Math.min(1, shadeData[i] * 2.5);
  shaded = Tensor.fromArray(ctx, shadeData, preShape);
  const blurTensor = maskValues(ValueMask.conv2d_blur)[0];
  const kData = blurTensor.read();
  const kSize = blurTensor.shape[0];
  const blurKernel = [];
  for (let i = 0; i < kSize; i++) {
    blurKernel.push(Array.from(kData.slice(i * kSize, i * kSize + kSize)));
  }
  for (let i = 0; i < 3; i++) shaded = convolution(shaded, blurKernel);
  const factor = Math.max(1, Math.floor(h / preH));
  let shadedUp = upsample(shaded, factor);
  let combined = upsample(control, factor);
  let shadedDataUp = shadedUp.read();
  for (let i = 0; i < shadedDataUp.length; i++) shadedDataUp[i] *= 0.75;
  shadedUp = Tensor.fromArray(ctx, shadedDataUp, [h, w, 1]);
  const zeros = values(1, shape, { ctx, distrib: ValueDistribution.zeros });
  const ones = values(1, shape, { ctx, distrib: ValueDistribution.ones });
  let out = blend(tensor, zeros, shadedUp);
  out = blend(out, ones, combined);
  return shadow(out, shape, time, speed, 0.5);
}
register("clouds", clouds, {});

export function fibers(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const valueShape = [h, w, 1];
  let out = tensor;
  for (let i = 0; i < 4; i++) {
    let mask = values(4, valueShape, { ctx, time, speed });
    const density = 0.05 + random() * 0.00125;
    const kink = randomInt(5, 10);
    mask = worms(
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
    let maskData = mask.read();
    for (let j = 0; j < maskData.length; j++) maskData[j] *= 0.5;
    mask = Tensor.fromArray(ctx, maskData, valueShape);
    out = blend(out, brightness, mask);
  }
  return out;
}
register("fibers", fibers, {});

export function scratches(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const valueShape = [h, w, 1];
  let out = tensor;
  for (let i = 0; i < 4; i++) {
    let mask = values(randomInt(2, 4), valueShape, { ctx, time, speed });
    const behavior = [WormBehavior.obedient, WormBehavior.unruly][
      randomInt(0, 1)
    ];
    const density = 0.25 + random() * 0.25;
    const duration = 2 + random() * 2;
    const kink = 0.125 + random() * 0.125;
    mask = worms(
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
    const sub = values(randomInt(2, 4), valueShape, { ctx, time, speed });
    let maskData = mask.read();
    const subData = sub.read();
    for (let j = 0; j < maskData.length; j++) {
      maskData[j] = Math.max(maskData[j] - subData[j] * 2.0, 0);
    }
    mask = Tensor.fromArray(ctx, maskData, valueShape);
    const outData = out.read();
    maskData = mask.read();
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
register("scratches", scratches, {});

export function strayHair(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const valueShape = [h, w, 1];
  let mask = values(4, valueShape, { ctx, time, speed });
  const density = 0.0025 + random() * 0.00125;
  const duration = randomInt(8, 16);
  const kink = randomInt(5, 50);
  mask = worms(
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
  let brightness = values(32, valueShape, { ctx, time, speed });
  let bData = brightness.read();
  for (let i = 0; i < bData.length; i++) bData[i] *= 0.333;
  brightness = Tensor.fromArray(ctx, bData, valueShape);
  let mData = mask.read();
  for (let i = 0; i < mData.length; i++) mData[i] *= 0.666;
  mask = Tensor.fromArray(ctx, mData, valueShape);
  return blend(tensor, brightness, mask);
}
register("strayHair", strayHair, {});

function expandChannels(tensor, channels) {
  const [h, w, c] = tensor.shape;
  if (c === channels) return tensor;
  const data = tensor.read();
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

export function grime(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const valueShape = [h, w, 1];
  let mask = fbm(null, valueShape, time, speed, 5, 8);
  mask = refract(mask);
  mask = derivative(
    mask,
    valueShape,
    time,
    speed,
    DistanceMetric.chebyshev,
    true,
    0.125,
  );
  let gateData = mask.read();
  const gate = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    const v = gateData[i] * gateData[i] * 0.075;
    for (let k = 0; k < c; k++) gate[i * c + k] = v;
  }
  const gateTensor = Tensor.fromArray(ctx, gate, shape);
  const baseData = new Float32Array(h * w * c).fill(0.25);
  const baseTensor = Tensor.fromArray(ctx, baseData, shape);
  let dusty = blend(tensor, baseTensor, gateTensor);
  let specks = values(Math.max(1, Math.floor(h * 0.25)), valueShape, {
    ctx,
    time,
    speed,
    distrib: ValueDistribution.exp,
    splineOrder: 0,
  });
  specks = refract(specks, null, null, 0.25);
  let sData = specks.read();
  for (let i = 0; i < sData.length; i++) {
    sData[i] = Math.max(sData[i] - 0.625, 0);
  }
  specks = Tensor.fromArray(ctx, sData, valueShape);
  specks = normalize(specks);
  sData = specks.read();
  for (let i = 0; i < sData.length; i++) sData[i] = 1 - Math.sqrt(sData[i]);
  specks = Tensor.fromArray(ctx, sData, valueShape);
  let noise = values(Math.max(h, w), valueShape, {
    ctx,
    time,
    speed,
    distrib: ValueDistribution.exp,
  });
  dusty = blend(dusty, noise, 0.075);
  let dustyData = dusty.read();
  sData = specks.read();
  for (let i = 0; i < dustyData.length; i++)
    dustyData[i] *= sData[Math.floor(i / c)];
  dusty = Tensor.fromArray(ctx, dustyData, shape);
  gateData = mask.read();
  const maskScaled = new Float32Array(h * w * c);
  for (let i = 0; i < h * w; i++) {
    const v = gateData[i] * 0.75;
    for (let k = 0; k < c; k++) maskScaled[i * c + k] = v;
  }
  const maskTensor = Tensor.fromArray(ctx, maskScaled, shape);
  return blend(tensor, dusty, maskTensor);
}
register("grime", grime, {});

export function watermark(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  const ctx = tensor.ctx;
  const valueShape = [h, w, 1];
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
  let mask = randomGlyphMask(valueShape, glyphs);
  mask = warp(mask, valueShape, time, speed, 2, 1, 0.5);
  const noise = values(2, valueShape, { ctx, time, speed }).read();
  let mData = mask.read();
  for (let i = 0; i < mData.length; i++) mData[i] *= noise[i] * noise[i];
  mask = Tensor.fromArray(ctx, mData, valueShape);
  let brightness = values(16, valueShape, { ctx, time, speed });
  if (c > 1) {
    mask = expandChannels(mask, c);
    brightness = expandChannels(brightness, c);
  }
  mData = mask.read();
  for (let i = 0; i < mData.length; i++) mData[i] *= 0.125;
  mask = Tensor.fromArray(ctx, mData, shape);
  return blend(tensor, brightness, mask);
}
register("watermark", watermark, {});

export function onScreenDisplay(tensor, shape, time, speed) {
  const [h, w, c] = shape;
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
  const glyphShape = maskShape(glyphs[0]);
  const height = glyphShape[0];
  const width = Math.min(w, glyphShape[1] * randomInt(3, 6));
  const rowMask = randomGlyphMask([height, width, 1], glyphs);
  const pad = new Float32Array(h * w);
  const rData = rowMask.read();
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const yy = Math.min(h - 1, 25 + y);
      const xx = Math.min(w - 1, w - width - 25 + x);
      pad[yy * w + xx] = rData[y * width + x];
    }
  }
  let rendered = Tensor.fromArray(null, pad, [h, w, 1]);
  if (c > 1) rendered = expandChannels(rendered, c);
  const alpha = 0.5 + random() * 0.25;
  const maxData = rendered.read();
  const tData = tensor.read();
  for (let i = 0; i < maxData.length; i++)
    maxData[i] = Math.max(maxData[i], tData[i]);
  const maxTensor = Tensor.fromArray(tensor.ctx, maxData, shape);
  return blend(tensor, maxTensor, alpha);
}
register("onScreenDisplay", onScreenDisplay, {});

export function spookyTicker(tensor, shape, time, speed) {
  const [h, w, c] = shape;
  if (random() > 0.75) {
    tensor = onScreenDisplay(tensor, shape, time, speed);
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
  const offsetMask = offsetTensor(rendered, -1, -1);
  const tData = tensor.read();
  const oData = offsetMask.read();
  const diff = new Float32Array(tData.length);
  for (let i = 0; i < tData.length; i++)
    diff[i] = tData[i] - oData[i % (h * w)];
  const diffTensor = Tensor.fromArray(tensor.ctx, diff, shape);
  tensor = blend(tensor, diffTensor, alpha * 0.333);
  let renderedC = rendered;
  if (c > 1) renderedC = expandChannels(rendered, c);
  const maxData = renderedC.read();
  const tData2 = tensor.read();
  for (let i = 0; i < maxData.length; i++)
    maxData[i] = Math.max(maxData[i], tData2[i]);
  const maxTensor = Tensor.fromArray(tensor.ctx, maxData, shape);
  return blend(tensor, maxTensor, alpha);
}
register("spookyTicker", spookyTicker, {});
