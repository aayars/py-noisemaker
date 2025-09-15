struct GrayscaleParams {
  width: u32,
  height: u32,
  channels: u32,
  pad: u32,
};

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: GrayscaleParams;

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
  var r = texel.x;
  var g = texel.y;
  var b = texel.z;
  if (params.channels <= 1u) {
    g = r;
    b = r;
  } else if (params.channels == 2u) {
    b = g;
  }
  let gray = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  let idx = y * w + x;
  out[idx] = gray;
}
