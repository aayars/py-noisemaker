export const VORONOI_WGSL = `
struct Point { x: f32, y: f32 };
struct Params {
  width: u32,
  height: u32,
  count: u32,
  diagram: u32,
  channels: u32,
};
@group(0) @binding(0) var<storage, read> points: array<Point>;
@group(0) @binding(1) var<storage, read_write> outBuffer: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  var best: f32 = 1e9;
  for (var i: u32 = 0u; i < params.count; i = i + 1u) {
    let dx = f32(gid.x) - points[i].x;
    let dy = f32(gid.y) - points[i].y;
    let d = dx*dx + dy*dy;
    if (d < best) { best = d; }
  }
  outBuffer[idx] = sqrt(best);
}
`;
