struct TintParams {
  width: f32,
  height: f32,
  channels: f32,
  alpha: f32,
  rand1: f32,
  rand2: f32,
  pad0: f32,
  pad1: f32,
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: TintParams;

fn fmod(a: f32, b: f32) -> f32 { return a - b * floor(a / b); }

fn rgb_to_hsv(c: vec3<f32>) -> vec3<f32> {
  let r = c.x;
  let g = c.y;
  let b = c.z;
  let maxv = max(r, max(g, b));
  let minv = min(r, min(g, b));
  let d = maxv - minv;
  var h: f32 = 0.0;
  if (d != 0.0) {
    if (maxv == r) {
      h = (g - b) / d;
      if (h < 0.0) { h = h + 6.0; }
    } else if (maxv == g) {
      h = (b - r) / d + 2.0;
    } else {
      h = (r - g) / d + 4.0;
    }
    h = h / 6.0;
  }
  let s = select(0.0, d / maxv, maxv != 0.0);
  return vec3<f32>(h, s, maxv);
}

fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
  let H = hsv.x;
  let S = hsv.y;
  let V = hsv.z;
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
  return vec3<f32>(r1 + m, g1 + m, b1 + m);
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  let col = textureLoad(tex, vec2<i32>(i32(x), i32(y)), 0);
  let rgb = col.xyz;
  let alpha_chan = col.w;
  let hsv = rgb_to_hsv(rgb);
  let newHue = fmod(rgb.x * 0.333 + params.rand1 + params.rand2, 1.0);
  let newSat = rgb.y;
  let newHSV = vec3<f32>(newHue, newSat, hsv.z);
  let colorized = hsv_to_rgb(newHSV);
  let blended = mix(rgb, colorized, params.alpha);
  let ch = u32(params.channels);
  let base = (y * w + x) * ch;
  if (ch > 0u) { out[base] = blended.x; }
  if (ch > 1u) { out[base + 1u] = blended.y; }
  if (ch > 2u) { out[base + 2u] = blended.z; }
  if (ch > 3u) { out[base + 3u] = alpha_chan; }
}
