export const VORONOI_WGSL = `
struct Point { x: f32, y: f32 };
struct Params {
  width: f32,
  height: f32,
  count: f32,
  blend: f32,
  alpha: f32,
  inverse: f32,
  metric: f32,
  nth: f32,
  sdfSides: f32,
  useFlow: f32,
  channels: f32,
  _pad0: f32,
};
@group(0) @binding(0) var<storage, read> points: array<Point>;
@group(0) @binding(2) var<storage, read_write> outBuffer: array<f32>;
@group(0) @binding(3) var<storage, read_write> indexBuffer: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read_write> flowBuffer: array<f32>;
@group(0) @binding(6) var<storage, read> pointColors: array<f32>;
@group(0) @binding(7) var<storage, read_write> colorFlowBuffer: array<f32>;
const MAX_NTH: u32 = 64u;
const MAX_CHANNELS: u32 = 4u;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u32(params.width) || gid.y >= u32(params.height)) { return; }
  let idx = gid.y * u32(params.width) + gid.x;
  let metric = u32(params.metric);
  let nth = u32(params.nth);
  let sdfSides = params.sdfSides;
  var best: array<f32, MAX_NTH>;
  var bestIdx: array<u32, MAX_NTH>;
  var flow: f32 = 0.0;
  var colorSum: array<f32, MAX_CHANNELS>;
  for (var j: u32 = 0u; j <= nth; j = j + 1u) {
    best[j] = 1e9;
    bestIdx[j] = 0u;
  }
  for (var k: u32 = 0u; k < u32(params.channels); k = k + 1u) {
    colorSum[k] = 0.0;
  }
  for (var i: u32 = 0u; i < u32(params.count); i = i + 1u) {
    var dx = f32(gid.x) - points[i].x;
    var dy = f32(gid.y) - points[i].y;
    if (metric == 101u || metric == 102u || metric == 201u) {
      if (params.inverse != 0.0) {
        dy = -dy;
      }
    } else {
      dx = abs(dx);
      if (dx > params.width * 0.5) { dx = params.width - dx; }
      dy = abs(dy);
      if (dy > params.height * 0.5) { dy = params.height - dy; }
    }
    var d: f32;
    if (metric == 2u) {
      d = abs(dx) + abs(dy);
    } else if (metric == 3u) {
      d = max(abs(dx), abs(dy));
    } else if (metric == 4u) {
      let adx = abs(dx);
      let ady = abs(dy);
      d = max((adx + ady) / sqrt(2.0), max(adx, ady));
    } else if (metric == 101u) {
      d = max(abs(dx) - dy * 0.5, dy);
    } else if (metric == 102u) {
      d = max(max(abs(dx) - dy * 0.5, dy), max(abs(dx) + dy * 0.5, -dy));
    } else if (metric == 201u) {
      let arctan = atan2(dx, -dy) + 3.141592653589793;
      let r = 6.283185307179586 / sdfSides;
      d = cos(floor(0.5 + arctan / r) * r - arctan) * sqrt(dx * dx + dy * dy);
    } else {
      d = sqrt(dx * dx + dy * dy);
    }
    if (d < best[nth]) {
      var k: u32 = nth;
      while (k > 0u && d < best[k - 1u]) {
        best[k] = best[k - 1u];
        bestIdx[k] = bestIdx[k - 1u];
        k = k - 1u;
      }
      best[k] = d;
      bestIdx[k] = i;
    }
    if (params.useFlow != 0.0) {
      var ld = log(max(d, 1e-9));
      if (ld < -10.0) { ld = -10.0; }
      if (ld > 10.0) { ld = 10.0; }
      flow = flow + ld;
      if (params.channels > 0.0) {
        let baseIdx = i * u32(params.channels);
        for (var c: u32 = 0u; c < u32(params.channels); c = c + 1u) {
          colorSum[c] = colorSum[c] + ld * pointColors[baseIdx + c];
        }
      }
    }
  }
  var idxNorm = f32(bestIdx[nth]) / params.count;
  outBuffer[idx] = best[nth];
  indexBuffer[idx] = idxNorm;
  if (params.useFlow != 0.0) {
    flowBuffer[idx] = flow;
    if (params.channels > 0.0) {
      for (var c: u32 = 0u; c < u32(params.channels); c = c + 1u) {
        colorFlowBuffer[idx * u32(params.channels) + c] = colorSum[c];
      }
    }
  }
}
`;

export const EROSION_WORMS_WGSL = `
struct Worm {
  x: f32,
  y: f32,
  dx: f32,
  dy: f32,
  inertia: f32,
};
struct Params {
  width: f32,
  height: f32,
  count: u32,
  iterations: u32,
  contraction: f32,
  quantize: f32,
  channels: u32,
};
@group(0) @binding(0) var<storage, read_write> worms: array<Worm>;
@group(0) @binding(1) var<storage, read> values: array<f32>;
@group(0) @binding(2) var<storage, read> startColors: array<f32>;
@group(0) @binding(3) var<storage, read_write> outBuffer: array<atomic<f32>>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.count) { return; }
  var worm = worms[idx];
  for (var it: u32 = 0u; it < params.iterations; it = it + 1u) {
    let xi = u32(floor(worm.x)) % u32(params.width);
    let yi = u32(floor(worm.y)) % u32(params.height);
    let x1 = (xi + 1u) % u32(params.width);
    let y1 = (yi + 1u) % u32(params.height);
    let baseIdx = yi * u32(params.width) + xi;
    let x1Idx = yi * u32(params.width) + x1;
    let y1Idx = y1 * u32(params.width) + xi;
    let x1y1Idx = y1 * u32(params.width) + x1;
    let base = values[baseIdx];
    let x1v = values[x1Idx];
    let y1v = values[y1Idx];
    let x1y1v = values[x1y1Idx];
    let u = fract(worm.x);
    let v = fract(worm.y);
    var gx = mix(y1v - base, x1y1v - x1v, u);
    var gy = mix(x1v - base, x1y1v - y1v, v);
    if (params.quantize != 0.0) {
      gx = floor(gx);
      gy = floor(gy);
    }
    let len = sqrt(gx * gx + gy * gy) * params.contraction;
    if (len != 0.0) {
      let invLen = 1.0 / len;
      worm.dx = worm.dx * (1.0 - worm.inertia) + gx * invLen * worm.inertia;
      worm.dy = worm.dy * (1.0 - worm.inertia) + gy * invLen * worm.inertia;
    }
    let exposure = 1.0 - abs(1.0 - f32(it) / f32(max(params.iterations - 1u, 1u)) * 2.0);
    let outBase = (yi * u32(params.width) + xi) * params.channels;
    let colorBase = idx * params.channels;
    for (var c: u32 = 0u; c < params.channels; c = c + 1u) {
      let amt = startColors[colorBase + c] * exposure;
      atomicAdd(&outBuffer[outBase + c], amt);
    }
    worm.x = worm.x + worm.dx;
    worm.y = worm.y + worm.dy;
    if (worm.x < 0.0) {
      worm.x = worm.x + params.width;
    } else if (worm.x >= params.width) {
      worm.x = worm.x - params.width;
    }
    if (worm.y < 0.0) {
      worm.y = worm.y + params.height;
    } else if (worm.y >= params.height) {
      worm.y = worm.y - params.height;
    }
  }
  worms[idx] = worm;
}
`;

export const WORMS_WGSL = `
struct WormParams {
  width: f32,
  height: f32,
  channels: f32,
  count: f32,
  iterations: f32,
  quantize: f32,
  kink: f32,
  _pad0: f32,
};
@group(0) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> strides: array<f32>;
@group(0) @binding(2) var<storage, read_write> rots: array<f32>;
@group(0) @binding(3) var<storage, read> colors: array<f32>;
@group(0) @binding(4) var<storage, read> indexBuffer: array<f32>;
@group(0) @binding(5) var<storage, read> drunk: array<f32>;
@group(0) @binding(6) var<storage, read_write> outBuffer: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params: WormParams;

fn atomicAddF32(target: ptr<storage, atomic<u32>>, value: f32) {
  var old = atomicLoad(target);
  loop {
    let f = bitcast<f32>(old) + value;
    let new = bitcast<u32>(f);
    let result = atomicCompareExchangeWeak(target, old, new);
    if (result.exchanged) { break; }
    old = result.old_value;
  }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let count = u32(params.count);
  if (idx >= count) { return; }
  let width = u32(params.width);
  let height = u32(params.height);
  let channels = u32(params.channels);
  let iterations = u32(params.iterations);

  var pos = positions[idx];
  let stride = strides[idx];
  var rot = rots[idx];

  for (var iter: u32 = 0u; iter < iterations; iter = iter + 1u) {
    rot = rot + drunk[iter * count + idx];
    let xi = u32(floor(pos.x)) % width;
    let yi = u32(floor(pos.y)) % height;
    let pix = yi * width + xi;
    var exposure: f32 = 1.0;
    if (iterations > 1u) {
      exposure = 1.0 - abs(1.0 - (f32(iter) / f32(iterations - 1u)) * 2.0);
    }
    let base = idx * channels;
    for (var c: u32 = 0u; c < channels; c = c + 1u) {
      let val = colors[base + c] * exposure;
      atomicAddF32(&outBuffer[pix * channels + c], val);
    }
    var next = indexBuffer[pix] + rot;
    if (params.quantize != 0.0) { next = round(next); }
    pos.y = pos.y + cos(next) * stride;
    pos.x = pos.x + sin(next) * stride;
    pos.y = fract(pos.y / params.height) * params.height;
    pos.x = fract(pos.x / params.width) * params.width;
  }
  positions[idx] = pos;
  rots[idx] = rot;
}
`;

