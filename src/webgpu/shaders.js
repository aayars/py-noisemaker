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
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
};
@group(0) @binding(0) var<storage, read> points: array<Point>;
@group(0) @binding(1) var<storage, read> base: array<f32>;
@group(0) @binding(2) var<storage, read_write> outBuffer: array<f32>;
@group(0) @binding(3) var<storage, read_write> indexBuffer: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
const MAX_NTH: u32 = 64u;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u32(params.width) || gid.y >= u32(params.height)) { return; }
  let idx = gid.y * u32(params.width) + gid.x;
  let metric = u32(params.metric);
  let nth = u32(params.nth);
  let sdfSides = params.sdfSides;
  var best: array<f32, MAX_NTH>;
  var bestIdx: array<u32, MAX_NTH>;
  for (var j: u32 = 0u; j <= nth; j = j + 1u) {
    best[j] = 1e9;
    bestIdx[j] = 0u;
  }
  for (var i: u32 = 0u; i < u32(params.count); i = i + 1u) {
    let dx = f32(gid.x) - points[i].x;
    let dy = f32(gid.y) - points[i].y;
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
  }
  var maxd: f32;
  if (metric == 2u) {
    maxd = params.width + params.height;
  } else if (metric == 3u) {
    maxd = max(params.width, params.height);
  } else if (metric == 4u) {
    maxd = max((params.width + params.height) / sqrt(2.0), max(params.width, params.height));
  } else {
    maxd = sqrt(params.width * params.width + params.height * params.height);
  }
  var dist = best[nth] / maxd;
  var idxNorm = f32(bestIdx[nth]) / params.count;
  if (params.inverse != 0.0) { dist = 1.0 - dist; }
  if (params.blend != 0.0) {
    let baseVal = base[idx];
    outBuffer[idx] = baseVal * (1.0 - params.alpha) + dist * params.alpha;
  } else {
    outBuffer[idx] = dist;
  }
  indexBuffer[idx] = idxNorm;
}
`;

