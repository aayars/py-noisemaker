struct DownsampleParams {
  srcWidth: f32;
  srcHeight: f32;
  dstWidth: f32;
  dstHeight: f32;
  factor: f32;
  channels: f32;
  pad0: f32;
  pad1: f32;
};
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: DownsampleParams;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let nw = u32(params.dstWidth);
  let nh = u32(params.dstHeight);
  if (x >= nw || y >= nh) { return; }
  let f = u32(params.factor);
  var sum: vec4<f32> = vec4<f32>(0.0);
  for (var yy: u32 = 0u; yy < f; yy = yy + 1u) {
    for (var xx: u32 = 0u; xx < f; xx = xx + 1u) {
      let ix = i32(x * f + xx);
      let iy = i32(y * f + yy);
      let v = textureLoad(src, vec2<i32>(ix, iy), 0);
      sum = sum + v;
    }
  }
  let inv = 1.0 / (params.factor * params.factor);
  let val = sum * inv;
  let ch = u32(params.channels);
  let base = (y * nw + x) * ch;
  if (ch > 0u) { out[base] = val.x; }
  if (ch > 1u) { out[base + 1u] = val.y; }
  if (ch > 2u) { out[base + 2u] = val.z; }
  if (ch > 3u) { out[base + 3u] = val.w; }
}
