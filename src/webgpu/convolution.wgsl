struct ConvolutionParams {
  width: f32,
  height: f32,
  channels: f32,
  kWidth: f32,
  kHeight: f32,
  normalize: f32,
  alpha: f32,
  kernelSum: f32,
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> kernel: array<f32>;
@group(0) @binding(3) var<uniform> params: ConvolutionParams;

fn wrapCoord(v: i32, size: i32) -> i32 {
  var m = v % size;
  if (m < 0) { m = m + size; }
  return m;
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  let w = i32(params.width);
  let h = i32(params.height);
  if (gid.x >= u32(w) || gid.y >= u32(h)) { return; }
  let kw = i32(params.kWidth);
  let kh = i32(params.kHeight);
  let halfW = kw / 2;
  let halfH = kh / 2;
  var sum: vec4<f32> = vec4<f32>(0.0);
  var idx: u32 = 0u;
  for (var j: i32 = 0; j < kh; j = j + 1) {
    for (var i: i32 = 0; i < kw; i = i + 1) {
      let xx = wrapCoord(x + i - halfW, w);
      let yy = wrapCoord(y + j - halfH, h);
      let v = textureLoad(tex, vec2<i32>(xx, yy), 0);
      let kVal = kernel[idx];
      sum = sum + v * kVal;
      idx = idx + 1u;
    }
  }
  if (params.normalize > 0.5 && params.kernelSum != 0.0) {
    sum = sum / params.kernelSum;
  }
  if (params.alpha < 1.0) {
    let orig = textureLoad(tex, vec2<i32>(x, y), 0);
    sum = mix(orig, sum, params.alpha);
  }
  let ch = u32(params.channels);
  let base = (u32(y) * u32(w) + u32(x)) * ch;
  if (ch > 0u) { out[base] = sum.x; }
  if (ch > 1u) { out[base + 1u] = sum.y; }
  if (ch > 2u) { out[base + 2u] = sum.z; }
  if (ch > 3u) { out[base + 3u] = sum.w; }
}
