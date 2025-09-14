struct WarpParams {
  width: f32,
  height: f32,
  channels: f32,
  displacement: f32,
  signed: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var rx: texture_2d<f32>;
@group(0) @binding(2) var ry: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: WarpParams;

fn fmod(a: f32, b: f32) -> f32 {
  return a - b * floor(a / b);
}

fn fmod2(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return a - b * floor(a / b);
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  var vx = textureLoad(rx, vec2<i32>(i32(x), i32(y)), 0).x;
  var vy = textureLoad(ry, vec2<i32>(i32(x), i32(y)), 0).x;
  if (params.signed > 0.5) {
    vx = vx * 2.0 - 1.0;
    vy = vy * 2.0 - 1.0;
  } else {
    vx = vx * 2.0;
    vy = vy * 2.0;
  }
  var samplePos = vec2<f32>(f32(x), f32(y)) + vec2<f32>(vx * params.displacement * params.width, vy * params.displacement * params.height);
  samplePos = fmod2(samplePos + vec2<f32>(params.width, params.height), vec2<f32>(params.width, params.height));
  let c0 = floor(samplePos);
  let f = samplePos - c0;
  let c1 = fmod2(c0 + 1.0, vec2<f32>(params.width, params.height));
  let i00 = vec2<i32>(i32(c0.x), i32(c0.y));
  let i10 = vec2<i32>(i32(c1.x), i32(c0.y));
  let i01 = vec2<i32>(i32(c0.x), i32(c1.y));
  let i11 = vec2<i32>(i32(c1.x), i32(c1.y));
  let sx = f.x;
  let sy = f.y;
  let s00 = textureLoad(tex, i00, 0);
  let s10 = textureLoad(tex, i10, 0);
  let s01 = textureLoad(tex, i01, 0);
  let s11 = textureLoad(tex, i11, 0);
  let mx0 = mix(s00, s10, sx);
  let mx1 = mix(s01, s11, sx);
  let val = mix(mx0, mx1, sy);
  let ch = u32(params.channels);
  let base = (y * w + x) * ch;
  if (ch > 0u) { out[base] = val.x; }
  if (ch > 1u) { out[base + 1u] = val.y; }
  if (ch > 2u) { out[base + 2u] = val.z; }
  if (ch > 3u) { out[base + 3u] = val.w; }
}
