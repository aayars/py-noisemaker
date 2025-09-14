struct CRTParams {
  width: f32,
  height: f32,
  channels: f32,
  disp: f32,
  hueShift: f32,
  randHue: f32,
  sat: f32,
  vigAlpha: f32,
  pad0: f32,
  pad1: f32,
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var scanTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: CRTParams;

fn fmod(a: f32, b: f32) -> f32 { return a - b * floor(a / b); }

fn rgb2hsv(col: vec3<f32>) -> vec3<f32> {
  let r = col.x;
  let g = col.y;
  let b = col.z;
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

fn hsv2rgb(col: vec3<f32>) -> vec3<f32> {
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
  return vec3<f32>(r1 + m, g1 + m, b1 + m);
}

fn getColor(ix: i32, iy: i32) -> vec3<f32> {
  let col = textureLoad(tex, vec2<i32>(ix, iy), 0);
  let scan = textureLoad(scanTex, vec2<i32>(ix, iy), 0).x;
  var c = col.xyz;
  c = c * 0.95 + (c + vec3<f32>(scan)) * scan * 0.05;
  var hsv = rgb2hsv(c);
  hsv.x = hsv.x + params.hueShift;
  hsv.x = hsv.x - floor(hsv.x);
  c = hsv2rgb(hsv);
  return c;
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  let ch = u32(params.channels);
  let cx = (params.width - 1.0) * 0.5;
  let cy = (params.height - 1.0) * 0.5;
  let dx = abs(f32(x) - cx);
  let dy = abs(f32(y) - cy);
  let dist = sqrt(dx*dx + dy*dy);
  let maxd = sqrt(cx*cx + cy*cy);
  let m = pow(select(0.0, dist / maxd, maxd != 0.0), 3.0);
  let g = select(0.0, f32(x) / (params.width - 1.0), params.width > 1.0);
  let disp = params.disp;
  var rX = min(params.width - 1.0, f32(x) + disp);
  rX = mix(rX, f32(x), g);
  rX = mix(f32(x), rX, m);
  rX = round(rX);
  var bX = max(0.0, f32(x) - disp);
  bX = mix(f32(x), bX, g);
  bX = mix(f32(x), bX, m);
  bX = round(bX);
  let rCol = getColor(i32(rX), i32(y));
  let gCol = getColor(i32(x), i32(y));
  let bCol = getColor(i32(bX), i32(y));
  var color = vec3<f32>(rCol.x, gCol.y, bCol.z);
  var hsv = rgb2hsv(color);
  hsv.x = hsv.x - params.hueShift + params.randHue;
  hsv.x = hsv.x - floor(hsv.x);
  hsv.y = hsv.y * params.sat;
  color = hsv2rgb(hsv);
  let uv = vec2<f32>(f32(x)/params.width, f32(y)/params.height);
  let distUv = distance(uv, vec2<f32>(0.5, 0.5));
  let vignetted = mix(color, vec3<f32>(0.0,0.0,0.0), distUv*distUv);
  color = mix(color, vignetted, params.vigAlpha);
  color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
  let base = (y * w + x) * ch;
  if (ch > 0u) { out[base] = color.x; }
  if (ch > 1u) { out[base + 1u] = color.y; }
  if (ch > 2u) { out[base + 2u] = color.z; }
  if (ch > 3u) {
    let baseCol = textureLoad(tex, vec2<i32>(i32(x), i32(y)), 0);
    out[base + 3u] = baseCol.w;
  }
}
