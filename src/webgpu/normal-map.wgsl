struct NormalMapParams {
  width: f32,
  height: f32,
  channels: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
  pad3: f32,
  pad4: f32,
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: NormalMapParams;

fn sampleGray(x: i32, y: i32) -> f32 {
  let w = i32(params.width);
  let h = i32(params.height);
  let cx = clamp(x, 0, w - 1);
  let cy = clamp(y, 0, h - 1);
  let col = textureLoad(tex, vec2<i32>(cx, cy), 0);
  if (params.channels < 1.5) {
    return col.x;
  }
  var g = col.x * 0.2126 + col.y * 0.7152;
  if (params.channels > 2.5) {
    g = g + col.z * 0.0722;
  }
  return g;
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  let w = u32(params.width);
  let h = u32(params.height);
  if (gid.x >= w || gid.y >= h) { return; }
  let gxKernel = array<f32,9>(-1.0,0.0,1.0,-2.0,0.0,2.0,-1.0,0.0,1.0);
  let gyKernel = array<f32,9>(-1.0,-2.0,-1.0,0.0,0.0,0.0,1.0,2.0,1.0);
  var gx: f32 = 0.0;
  var gy: f32 = 0.0;
  var idx: u32 = 0u;
  for (var yy: i32 = -1; yy <= 1; yy = yy + 1) {
    for (var xx: i32 = -1; xx <= 1; xx = xx + 1) {
      let v = sampleGray(x + xx, y + yy);
      gx = gx + v * gxKernel[idx];
      gy = gy + v * gyKernel[idx];
      idx = idx + 1u;
    }
  }
  let n = normalize(vec3<f32>(-gx, gy, 1.0));
  let base = (u32(y) * w + u32(x)) * 3u;
  out[base] = n.x * 0.5 + 0.5;
  out[base + 1u] = n.y * 0.5 + 0.5;
  out[base + 2u] = n.z * 0.5 + 0.5;
}
