struct VoronoiParams {
  size : vec4<f32>;
  config : vec4<f32>;
  extra : vec4<f32>;
};

@group(0) @binding(0) var<storage, read> pointsBuffer : array<f32>;
@group(0) @binding(2) var<storage, read_write> rangeBuffer : array<f32>;
@group(0) @binding(3) var<storage, read_write> indexBuffer : array<f32>;
@group(0) @binding(4) var<uniform> params : VoronoiParams;
@group(0) @binding(5) var<storage, read_write> flowBuffer : array<f32>;
@group(0) @binding(6) var<storage, read> pointColorBuffer : array<f32>;
@group(0) @binding(7) var<storage, read_write> colorFlowBuffer : array<f32>;

const MAX_NTH : u32 = 64u;
const EPSILON : f32 = 1e-6;
const LARGE_VALUE : f32 = 1e30;
const TAU : f32 = 6.283185307179586;
const MAX_COLOR_CHANNELS : u32 = 32u;

fn width() -> u32 {
  return max(u32(params.size.x + 0.5), 1u);
}

fn height() -> u32 {
  return max(u32(params.size.y + 0.5), 1u);
}

fn pointCount() -> u32 {
  return max(u32(params.size.z + 0.5), 0u);
}

fn inverseFlag() -> bool {
  return params.config.y > 0.5;
}

fn metricId() -> i32 {
  return i32(round(params.config.z));
}

fn nthIndex(count : u32, flowMode : bool) -> u32 {
  if (count == 0u) {
    return 0u;
  }
  if (flowMode) {
    return 0u;
  }
  let rawNth : f32 = params.config.w;
  var nthValue : i32 = i32(round(rawNth));
  if (nthValue < 0) {
    nthValue = 0;
  }
  let maxIdx : i32 = i32(count) - 1;
  if (nthValue > maxIdx) {
    nthValue = maxIdx;
  }
  var nthU : u32 = u32(nthValue);
  if (nthU >= MAX_NTH) {
    nthU = MAX_NTH - 1u;
  }
  return nthU;
}

fn sdfSides() -> f32 {
  return max(params.extra.x, 3.0);
}

fn flowMode() -> bool {
  return params.extra.y > 0.5;
}

fn flowChannels() -> u32 {
  let raw = params.extra.z;
  if (raw <= 0.0) {
    return 0u;
  }
  return max(u32(raw + 0.5), 0u);
}

fn isTriangularMetric(metric : i32) -> bool {
  return metric == 101 || metric == 102 || metric == 201;
}

fn distanceMetric(dx : f32, dy : f32, metric : i32, sdfSidesVal : f32) -> f32 {
  switch (metric) {
    case 2: { // Manhattan
      return abs(dx) + abs(dy);
    }
    case 3: { // Chebyshev
      return max(abs(dx), abs(dy));
    }
    case 4: { // Octagram
      let sum = (abs(dx) + abs(dy)) * 0.7071067811865476;
      return max(sum, max(abs(dx), abs(dy)));
    }
    case 101: { // Triangular
      return max(abs(dx) - dy * 0.5, dy);
    }
    case 102: { // Hexagram
      let a = max(abs(dx) - dy * 0.5, dy);
      let b = max(abs(dx) + dy * 0.5, -dy);
      return max(a, b);
    }
    case 201: { // Signed distance field polygon
      let angle = atan2(dx, -dy) + 3.141592653589793;
      let r = TAU / max(sdfSidesVal, 3.0);
      let k = floor(0.5 + angle / r) * r - angle;
      let base = sqrt(dx * dx + dy * dy);
      return cos(k) * base;
    }
    case 1:
    default: {
      let sum = dx * dx + dy * dy;
      return sqrt(max(sum, 0.0));
    }
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let w : u32 = width();
  let h : u32 = height();
  if (global_id.x >= w || global_id.y >= h) {
    return;
  }

  let count : u32 = pointCount();
  if (count == 0u) {
    return;
  }

  let metric : i32 = metricId();
  let triMetric : bool = isTriangularMetric(metric);
  let flow : bool = flowMode();
  let channels : u32 = flowChannels();
  let nth : u32 = nthIndex(count, flow);
  let widthF : f32 = max(params.size.x, 1.0);
  let heightF : f32 = max(params.size.y, 1.0);
  let halfW : f32 = floor(widthF * 0.5);
  let halfH : f32 = floor(heightF * 0.5);
  let ySign : f32 = if (inverseFlag() && triMetric) { -1.0 } else { 1.0 };
  let sdfSidesVal : f32 = sdfSides();

  var bestDists : array<f32, MAX_NTH>;
  var bestIdx : array<u32, MAX_NTH>;
  for (var i : u32 = 0u; i <= nth; i = i + 1u) {
    bestDists[i] = LARGE_VALUE;
    bestIdx[i] = 0u;
  }

  let pixelX : f32 = f32(global_id.x);
  let pixelY : f32 = f32(global_id.y);
  var flowAccum : f32 = 0.0;
  let hasColorFlow : bool = flow && channels > 0u;
  var colorAccum : array<f32, MAX_COLOR_CHANNELS>;
  if (hasColorFlow) {
    for (var cIdx : u32 = 0u; cIdx < MAX_COLOR_CHANNELS; cIdx = cIdx + 1u) {
      colorAccum[cIdx] = 0.0;
    }
  }

  for (var i : u32 = 0u; i < count; i = i + 1u) {
    let base : u32 = i * 2u;
    let px : f32 = pointsBuffer[base];
    let py : f32 = pointsBuffer[base + 1u];

    var dx : f32;
    var dy : f32;
    if (triMetric) {
      dx = (pixelX - px) / widthF;
      dy = ((pixelY - py) * ySign) / heightF;
    } else {
      let x0 : f32 = pixelX - px - halfW;
      let x1 : f32 = pixelX - px + halfW;
      let y0 : f32 = pixelY - py - halfH;
      let y1 : f32 = pixelY - py + halfH;
      dx = min(abs(x0), abs(x1)) / widthF;
      dy = min(abs(y0), abs(y1)) / heightF;
    }

    let dist : f32 = distanceMetric(dx, dy, metric, sdfSidesVal);
    let safeDist : f32 = max(dist, EPSILON);

    if (flow) {
      let logVal : f32 = log(safeDist);
      let clamped : f32 = clamp(logVal, -10.0, 10.0);
      flowAccum = flowAccum + clamped;
      if (hasColorFlow) {
        let channelBase : u32 = i * channels;
        let limit : u32 = min(channels, MAX_COLOR_CHANNELS);
        for (var cIdx : u32 = 0u; cIdx < limit; cIdx = cIdx + 1u) {
          colorAccum[cIdx] = colorAccum[cIdx] + clamped * pointColorBuffer[channelBase + cIdx];
        }
        if (channels > MAX_COLOR_CHANNELS) {
          for (var cIdx : u32 = MAX_COLOR_CHANNELS; cIdx < channels; cIdx = cIdx + 1u) {
            let unused = pointColorBuffer[channelBase + cIdx];
            // Prevent unused variable warnings; contribution ignored beyond MAX_COLOR_CHANNELS.
            if (unused != 0.0) {
              // no-op to keep the compiler from eliminating the read
            }
          }
        }
      }
    }

    let targetIdx : u32 = nth;
    if (dist < bestDists[targetIdx] - EPSILON || (abs(dist - bestDists[targetIdx]) <= EPSILON && i < bestIdx[targetIdx])) {
      var j : u32 = targetIdx;
      loop {
        if (j == 0u) {
          break;
        }
        let prev : u32 = j - 1u;
        if (dist >= bestDists[prev]) {
          break;
        }
        bestDists[j] = bestDists[prev];
        bestIdx[j] = bestIdx[prev];
        j = prev;
      }
      bestDists[j] = dist;
      bestIdx[j] = i;
    }
  }

  let pixelIndex : u32 = global_id.y * w + global_id.x;
  let countF : f32 = f32(count);

  if (flow) {
    rangeBuffer[pixelIndex] = bestDists[0u];
    indexBuffer[pixelIndex] = f32(bestIdx[0u]) / max(countF, 1.0);
    flowBuffer[pixelIndex] = flowAccum;
    if (hasColorFlow) {
      let limit : u32 = min(channels, MAX_COLOR_CHANNELS);
      let base : u32 = pixelIndex * channels;
      for (var cIdx : u32 = 0u; cIdx < limit; cIdx = cIdx + 1u) {
        colorFlowBuffer[base + cIdx] = colorAccum[cIdx];
      }
      if (channels > MAX_COLOR_CHANNELS) {
        let start : u32 = base + MAX_COLOR_CHANNELS;
        for (var cIdx : u32 = MAX_COLOR_CHANNELS; cIdx < channels; cIdx = cIdx + 1u) {
          colorFlowBuffer[start + (cIdx - MAX_COLOR_CHANNELS)] = 0.0;
        }
      }
    }
  } else {
    rangeBuffer[pixelIndex] = bestDists[nth];
    indexBuffer[pixelIndex] = f32(bestIdx[nth]) / max(countF, 1.0);
    // flowBuffer and colorFlowBuffer are unused in this mode.
  }
}
