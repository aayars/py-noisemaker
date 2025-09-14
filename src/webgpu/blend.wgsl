struct BlendParams {
  width: f32;
  height: f32;
  channels: f32;
  t: f32;
  pad0: f32;
  pad1: f32;
  pad2: f32;
  pad3: f32;
};
@group(0) @binding(0) var texA: texture_2d<f32>;
@group(0) @binding(1) var texB: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: BlendParams;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  let a = textureLoad(texA, vec2<i32>(i32(x), i32(y)), 0);
  let b = textureLoad(texB, vec2<i32>(i32(x), i32(y)), 0);
  let val = mix(a, b, params.t);
  let ch = u32(params.channels);
  let base = (y * w + x) * ch;
  if (ch > 0u) { out[base] = val.x; }
  if (ch > 1u) { out[base + 1u] = val.y; }
  if (ch > 2u) { out[base + 2u] = val.z; }
  if (ch > 3u) { out[base + 3u] = val.w; }
}
