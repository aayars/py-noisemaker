export const VORONOI_WGSL = `
struct Point { x: f32, y: f32 };
struct Params {
  width: f32,
  height: f32,
  count: f32,
  blend: f32,
  alpha: f32,
  inverse: f32,
  padding0: f32,
  padding1: f32,
};
@group(0) @binding(0) var<storage, read> points: array<Point>;
@group(0) @binding(1) var<storage, read> base: array<f32>;
@group(0) @binding(2) var<storage, read_write> outBuffer: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u32(params.width) || gid.y >= u32(params.height)) { return; }
  let idx = gid.y * u32(params.width) + gid.x;
  var best: f32 = 1e9;
  for (var i: u32 = 0u; i < u32(params.count); i = i + 1u) {
    let dx = f32(gid.x) - points[i].x;
    let dy = f32(gid.y) - points[i].y;
    let d = dx*dx + dy*dy;
    if (d < best) { best = d; }
  }
  let maxd = sqrt(params.width * params.width + params.height * params.height);
  var dist = sqrt(best) / maxd;
  if (params.inverse != 0.0) { dist = 1.0 - dist; }
  if (params.blend != 0.0) {
    let baseVal = base[idx];
    outBuffer[idx] = baseVal * (1.0 - params.alpha) + dist * params.alpha;
  } else {
    outBuffer[idx] = dist;
  }
}
`;

