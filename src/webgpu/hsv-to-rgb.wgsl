struct HSVToRGBParams {
  width: f32,
  height: f32,
  channels: f32,
  pad0: f32,
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: HSVToRGBParams;

fn fmod(a: f32, b: f32) -> f32 { return a - b * floor(a / b); }

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  let col = textureLoad(tex, vec2<i32>(i32(x), i32(y)), 0);
  let H = col.x;
  let S = col.y;
  let V = col.z;
  let C = V * S;
  let hPrime = fmod(H * 6.0, 6.0);
  let X = C * (1.0 - abs(fmod(hPrime, 2.0) - 1.0));
  var r1: f32 = 0.0;
  var g1: f32 = 0.0;
  var b1: f32 = 0.0;
  if (hPrime < 1.0) {
    r1 = C; g1 = X;
  } else if (hPrime < 2.0) {
    r1 = X; g1 = C;
  } else if (hPrime < 3.0) {
    g1 = C; b1 = X;
  } else if (hPrime < 4.0) {
    g1 = X; b1 = C;
  } else if (hPrime < 5.0) {
    r1 = X; b1 = C;
  } else {
    r1 = C; b1 = X;
  }
  let m = V - C;
  let base = (y * w + x) * 3u;
  out[base] = r1 + m;
  out[base + 1u] = g1 + m;
  out[base + 2u] = b1 + m;
}
