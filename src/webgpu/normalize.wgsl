struct NormalizeParams {
  width: f32,
  height: f32,
  channels: f32,
  mode: f32,
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read_write> reduce: array<f32>;
@group(0) @binding(3) var<uniform> params: NormalizeParams;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let mode = params.mode;
  let w = u32(params.width);
  let h = u32(params.height);
  let c = u32(params.channels);
  if (mode < 0.5) {
    if (gid.x == 0u && gid.y == 0u) {
      var minv = 1e20;
      var maxv = -1e20;
      for (var y: u32 = 0u; y < h; y = y + 1u) {
        for (var x: u32 = 0u; x < w; x = x + 1u) {
          let col = textureLoad(tex, vec2<i32>(i32(x), i32(y)), 0);
          if (c > 0u) { let v = col.x; minv = min(minv, v); maxv = max(maxv, v); }
          if (c > 1u) { let v = col.y; minv = min(minv, v); maxv = max(maxv, v); }
          if (c > 2u) { let v = col.z; minv = min(minv, v); maxv = max(maxv, v); }
          if (c > 3u) { let v = col.w; minv = min(minv, v); maxv = max(maxv, v); }
        }
      }
      reduce[0] = minv;
      reduce[1] = maxv;
    }
    return;
  }
  let x = gid.x;
  let y = gid.y;
  if (x >= w || y >= h) { return; }
  let minv = reduce[0];
  let maxv = reduce[1];
  let range = max(maxv - minv, 1e-6);
  let col = textureLoad(tex, vec2<i32>(i32(x), i32(y)), 0);
  let diff = col - vec4<f32>(minv);
  let val = diff / vec4<f32>(range);
  let base = (y * w + x) * c;
  if (c > 0u) { out[base] = val.x; }
  if (c > 1u) { out[base + 1u] = val.y; }
  if (c > 2u) { out[base + 2u] = val.z; }
  if (c > 3u) { out[base + 3u] = val.w; }
}
