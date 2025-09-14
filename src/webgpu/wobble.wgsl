struct WobbleParams {
  width: f32,
  height: f32,
  channels: f32,
  xOffset: f32,
  yOffset: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: WobbleParams;

fn fmod(a: f32, b: f32) -> f32 {
  return a - b * floor(a / b);
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  let sx = i32(fmod(f32(x) + params.xOffset + params.width, params.width));
  let sy = i32(fmod(f32(y) + params.yOffset + params.height, params.height));
  let val = textureLoad(tex, vec2<i32>(sx, sy), 0);
  let ch = u32(params.channels);
  let base = (y * w + x) * ch;
  if (ch > 0u) { out[base] = val.x; }
  if (ch > 1u) { out[base + 1u] = val.y; }
  if (ch > 2u) { out[base + 2u] = val.z; }
  if (ch > 3u) { out[base + 3u] = val.w; }
}
