struct SobelParams {
  width: f32;
  height: f32;
  channels: f32;
  pad0: f32;
  pad1: f32;
  pad2: f32;
  pad3: f32;
  pad4: f32;
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: SobelParams;

fn sampleClamped(x: i32, y: i32) -> vec4<f32> {
  let w = i32(params.width);
  let h = i32(params.height);
  let cx = clamp(x, 0, w - 1);
  let cy = clamp(y, 0, h - 1);
  return textureLoad(tex, vec2<i32>(cx, cy), 0);
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
  var gx: vec4<f32> = vec4<f32>(0.0);
  var gy: vec4<f32> = vec4<f32>(0.0);
  var idx: u32 = 0u;
  for (var yy: i32 = -1; yy <= 1; yy = yy + 1) {
    for (var xx: i32 = -1; xx <= 1; xx = xx + 1) {
      let v = sampleClamped(x + xx, y + yy);
      gx = gx + v * gxKernel[idx];
      gy = gy + v * gyKernel[idx];
      idx = idx + 1u;
    }
  }
  let res = sqrt(gx * gx + gy * gy);
  let ch = u32(params.channels);
  let base = (u32(y) * w + u32(x)) * ch;
  if (ch > 0u) { out[base] = res.x; }
  if (ch > 1u) { out[base + 1u] = res.y; }
  if (ch > 2u) { out[base + 2u] = res.z; }
  if (ch > 3u) { out[base + 3u] = res.w; }
}
