struct DerivativeParams {
  width: f32,
  height: f32,
  channels: f32,
  metric: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
  pad3: f32,
};
@group(0) @binding(0) var texDx: texture_2d<f32>;
@group(0) @binding(1) var texDy: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: DerivativeParams;

fn dist(dx: f32, dy: f32) -> f32 {
  let m = i32(params.metric);
  switch m {
    case 2: { return abs(dx) + abs(dy); }
    case 3: { return max(abs(dx), abs(dy)); }
    case 4: { return max((abs(dx) + abs(dy)) / sqrt(2.0), max(abs(dx), abs(dy))); }
    case 101: { return max(abs(dx) - dy * 0.5, dy); }
    case 102: { return max(max(abs(dx) - dy * 0.5, dy), max(abs(dx) + dy * 0.5, -dy)); }
    case 201: {
      let PI = 3.141592653589793;
      let arctan = atan2(dx, -dy) + PI;
      let r = (PI * 2.0) / 5.0;
      return cos(floor(0.5 + arctan / r) * r - arctan) * sqrt(dx * dx + dy * dy);
    }
    default: {
      return sqrt(dx * dx + dy * dy);
    }
  }
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = u32(params.width);
  let h = u32(params.height);
  if (gid.x >= w || gid.y >= h) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let dxv = textureLoad(texDx, vec2<i32>(x, y), 0);
  let dyv = textureLoad(texDy, vec2<i32>(x, y), 0);
  let ch = u32(params.channels);
  let base = (gid.y * w + gid.x) * ch;
  if (ch > 0u) { out[base] = dist(dxv.x, dyv.x); }
  if (ch > 1u) { out[base + 1u] = dist(dxv.y, dyv.y); }
  if (ch > 2u) { out[base + 2u] = dist(dxv.z, dyv.z); }
  if (ch > 3u) { out[base + 3u] = dist(dxv.w, dyv.w); }
}
