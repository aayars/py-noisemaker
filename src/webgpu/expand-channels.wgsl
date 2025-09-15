struct ExpandParams {
  width: u32,
  height: u32,
  channels: u32,
  pad: u32,
};

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: ExpandParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = params.width;
  let h = params.height;
  if (x >= w || y >= h) {
    return;
  }
  let texel = textureLoad(tex, vec2<i32>(i32(x), i32(y)), 0);
  let value = texel.x;
  let base = (y * w + x) * params.channels;
  for (var i: u32 = 0u; i < params.channels; i = i + 1u) {
    out[base + i] = value;
  }
}
