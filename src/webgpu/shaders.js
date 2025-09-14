import { ValueDistribution, InterpolationType } from '../constants.js';

const loadShader = async (name) => (await fetch(new URL(name, import.meta.url))).text();

export const VORONOI_WGSL = await loadShader('./voronoi.wgsl');
export const EROSION_WORMS_WGSL = await loadShader('./erosion-worms.wgsl');
export const WORMS_WGSL = await loadShader('./worms.wgsl');
export const RESAMPLE_WGSL = await loadShader('./resample.wgsl');
export const UPSAMPLE_WGSL = RESAMPLE_WGSL;
export const DOWNSAMPLE_WGSL = await loadShader('./downsample.wgsl');
export const BLEND_WGSL = await loadShader('./blend.wgsl');
export const SOBEL_WGSL = await loadShader('./sobel.wgsl');
export const REFRACT_WGSL = await loadShader('./refract.wgsl');
export const VALUE_WGSL = /* wgsl */ `
struct ValueParams {
  width: f32;
  height: f32;
  freqX: f32;
  freqY: f32;
  seed: f32;
  time: f32;
  speed: f32;
  corners: f32;
  interp: f32;
  distrib: f32;
  useMask: f32;
  maskWidth: f32;
  maskHeight: f32;
  pad1: f32;
  pad2: f32;
  pad3: f32;
};
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: ValueParams;
@group(0) @binding(2) var<storage, read> mask: array<f32>;

fn mod289_2(x: vec2<f32>) -> vec2<f32> { return x - floor(x * (1.0 / 289.0)) * 289.0; }
fn mod289_3(x: vec3<f32>) -> vec3<f32> { return x - floor(x * (1.0 / 289.0)) * 289.0; }
fn mod289_4(x: vec4<f32>) -> vec4<f32> { return x - floor(x * (1.0 / 289.0)) * 289.0; }
fn permute3(x: vec3<f32>) -> vec3<f32> { return mod289_3(((x * 34.0) + 1.0) * x); }
fn permute4(x: vec4<f32>) -> vec4<f32> { return mod289_4(((x * 34.0) + 1.0) * x); }
fn taylorInvSqrt(r: vec4<f32>) -> vec4<f32> { return 1.79284291400159 - 0.85373472095314 * r; }
fn simplex2(v: vec2<f32>) -> f32 {
  let C = vec4<f32>(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
  var i = floor(v + dot(v, C.yy));
  var x0 = v - i + dot(i, C.xx);
  var i1 = select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), x0.x > x0.y);
  var x12 = x0.xyxy + C.xxzz;
  x12.xy = x12.xy - i1;
  i = mod289_2(i);
  var p = permute3(permute3(i.y + vec3<f32>(0.0, i1.y, 1.0)) + i.x + vec3<f32>(0.0, i1.x, 1.0));
  var m = max(0.5 - vec3<f32>(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), vec3<f32>(0.0));
  m = m * m;
  m = m * m;
  var x = 2.0 * fract(p * C.www) - 1.0;
  var h = abs(x) - 0.5;
  var ox = floor(x + 0.5);
  var a0 = x - ox;
  m = m * (1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h));
  var g = vec3<f32>(0.0);
  g.x = a0.x * x0.x + h.x * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}
fn simplex3(v: vec3<f32>) -> f32 {
  let C = vec2<f32>(1.0 / 6.0, 1.0 / 3.0);
  let D = vec4<f32>(0.0, 0.5, 1.0, 2.0);
  var i = floor(v + dot(v, C.yyy));
  var x0 = v - i + dot(i, C.xxx);
  let g = step(x0.yzx, x0.xyz);
  let l = 1.0 - g;
  let i1 = min(g.xyz, l.zxy);
  let i2 = max(g.xyz, l.zxy);
  let x1 = x0 - i1 + C.xxx;
  let x2 = x0 - i2 + C.yyy;
  let x3 = x0 - D.yyy;
  i = mod289_3(i);
  var p = permute4(permute4(permute4(i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0)) + i.y + vec4<f32>(0.0, i1.y, i2.y, 1.0)) + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0));
  let n_ = 0.142857142857;
  let ns = n_ * D.wyz - D.xzx;
  var j = p - 49.0 * floor(p * ns.z * ns.z);
  var x_ = floor(j * ns.z);
  var y_ = floor(j - 7.0 * x_);
  var x = x_ * ns.x + ns.yyyy;
  var y = y_ * ns.x + ns.yyyy;
  var h = 1.0 - abs(x) - abs(y);
  var b0 = vec4<f32>(x.xy, y.xy);
  var b1 = vec4<f32>(x.zw, y.zw);
  var s0 = floor(b0) * 2.0 + 1.0;
  var s1 = floor(b1) * 2.0 + 1.0;
  var sh = -step(h, vec4<f32>(0.0));
  var a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  var a1 = b1.xzyw + s1.xzyw * sh.zzww;
  var p0 = vec3<f32>(a0.xy, h.x);
  var p1 = vec3<f32>(a0.zw, h.y);
  var p2 = vec3<f32>(a1.xy, h.z);
  var p3 = vec3<f32>(a1.zw, h.w);
  var norm = taylorInvSqrt(vec4<f32>(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;
  var m = max(0.6 - vec4<f32>(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4<f32>(0.0));
  m = m * m;
  return 42.0 * dot(m * m, vec4<f32>(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

fn rand2D(x: f32, y: f32, seed: f32, time: f32, speed: f32) -> f32 {
  let sx = floor(x);
  let sy = floor(y);
  let s = sx * 12.9898 + sy * 78.233 + (seed % 65536.0) * 37.719 + time * speed * 0.1;
  return fract(sin(s) * 43758.5453);
}

fn sdfPolygon(dx: f32, dy: f32, sides: f32) -> f32 {
  let an = atan2(dx, -dy) + 3.141592653589793;
  let r = 6.283185307179586 / sides;
  return cos(floor(0.5 + an / r) * r - an) * sqrt(dx * dx + dy * dy);
}

fn interpFunc(t: f32) -> f32 {
  let i = u32(params.interp);
  if (i == ${InterpolationType.linear}u) { return t; }
  if (i == ${InterpolationType.cosine}u) { return 0.5 - cos(t * 3.141592653589793) * 0.5; }
  return t * t * (3.0 - 2.0 * t);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = u32(params.width);
  let height = u32(params.height);
  let idx = gid.x;
  if (idx >= width * height) { return; }
  let x = f32(idx % width);
  let y = f32(idx / width);
  var val: f32 = 0.0;
  let dx = (x + 0.5) / params.width - 0.5;
  let dy = (y + 0.5) / params.height - 0.5;
  let distrib = u32(params.distrib);

  if (distrib == ${ValueDistribution.exp}u) {
    let r = rand2D(x, y, params.seed, params.time, params.speed);
    val = pow(r, 3.0);
  } else if (distrib == ${ValueDistribution.simplex}u) {
    let fx = max(1.0, floor(params.freqX));
    let fy = max(1.0, floor(params.freqY));
    var xx = x / params.width * fx;
    var yy = y / params.height * fy;
    if ((params.corners < 0.5 && mod(fx, 2.0) == 0.0) || (params.corners > 0.5 && mod(fx, 2.0) == 1.0)) {
      xx = mod(xx + 0.5, fx);
    }
    if ((params.corners < 0.5 && mod(fy, 2.0) == 0.0) || (params.corners > 0.5 && mod(fy, 2.0) == 1.0)) {
      yy = mod(yy + 0.5, fy);
    }
    let ang = 6.283185307179586 * params.time;
    let z = cos(ang) * params.speed;
    let s = params.seed % 65536.0;
    val = (simplex3(vec3<f32>(xx + s, yy + s, z)) + 1.0) * 0.5;
  } else if (distrib == ${ValueDistribution.column_index}u) {
    let fy = max(1.0, floor(params.freqY));
    var yy = y / params.height * fy;
    if ((params.corners < 0.5 && mod(fy, 2.0) == 0.0) || (params.corners > 0.5 && mod(fy, 2.0) == 1.0)) {
      yy = mod(yy + 0.5, fy);
    }
    val = fy <= 1.0 ? 0.0 : floor(yy) / (fy - 1.0);
  } else if (distrib == ${ValueDistribution.row_index}u) {
    let fx = max(1.0, floor(params.freqX));
    var xx = x / params.width * fx;
    if ((params.corners < 0.5 && mod(fx, 2.0) == 0.0) || (params.corners > 0.5 && mod(fx, 2.0) == 1.0)) {
      xx = mod(xx + 0.5, fx);
    }
    val = fx <= 1.0 ? 0.0 : floor(xx) / (fx - 1.0);
  } else if (distrib == ${ValueDistribution.ones}u) {
    val = 1.0;
  } else if (distrib == ${ValueDistribution.mids}u) {
    val = 0.5;
  } else if (distrib == ${ValueDistribution.zeros}u) {
    val = 0.0;
  } else if (distrib == ${ValueDistribution.center_circle}u) {
    let d = sqrt(dx * dx + dy * dy);
    val = max(0.0, 1.0 - d * 2.0);
  } else if (distrib == ${ValueDistribution.center_triangle}u) {
    let d = max(abs(dx) - dy * 0.5, dy);
    val = max(0.0, 1.0 - d * 2.0);
  } else if (distrib == ${ValueDistribution.center_diamond}u) {
    let d = abs(dx) + abs(dy);
    val = max(0.0, 1.0 - d * 2.0);
  } else if (distrib == ${ValueDistribution.center_square}u) {
    let d = max(abs(dx), abs(dy));
    val = max(0.0, 1.0 - d * 2.0);
  } else if (distrib == ${ValueDistribution.center_pentagon}u) {
    let d = sdfPolygon(dx, dy, 5.0);
    val = max(0.0, 1.0 - d * 2.0);
  } else if (distrib == ${ValueDistribution.center_hexagon}u) {
    let d = max(max(abs(dx) - dy * 0.5, dy), max(abs(dx) + dy * 0.5, -dy));
    val = max(0.0, 1.0 - d * 2.0);
  } else if (distrib == ${ValueDistribution.center_heptagon}u) {
    let d = sdfPolygon(dx, dy, 7.0);
    val = max(0.0, 1.0 - d * 2.0);
  } else if (distrib == ${ValueDistribution.center_octagon}u) {
    let d = max((abs(dx) + abs(dy)) / 1.4142135623730951, max(abs(dx), abs(dy)));
    val = max(0.0, 1.0 - d * 2.0);
  } else if (distrib == ${ValueDistribution.center_nonagon}u) {
    let d = sdfPolygon(dx, dy, 9.0);
    val = max(0.0, 1.0 - d * 2.0);
  } else if (distrib == ${ValueDistribution.center_decagon}u) {
    let d = sdfPolygon(dx, dy, 10.0);
    val = max(0.0, 1.0 - d * 2.0);
  } else if (distrib == ${ValueDistribution.center_hendecagon}u) {
    let d = sdfPolygon(dx, dy, 11.0);
    val = max(0.0, 1.0 - d * 2.0);
  } else if (distrib == ${ValueDistribution.center_dodecagon}u) {
    let d = sdfPolygon(dx, dy, 12.0);
    val = max(0.0, 1.0 - d * 2.0);
  } else {
    let u = x / params.width * params.freqX;
    let v = y / params.height * params.freqY;
    let x0 = floor(u);
    let y0 = floor(v);
    let xf = fract(u);
    let yf = fract(v);
    let r00 = rand2D(x0, y0, params.seed, params.time, params.speed);
    let r10 = rand2D(x0 + 1.0, y0, params.seed, params.time, params.speed);
    let r01 = rand2D(x0, y0 + 1.0, params.seed, params.time, params.speed);
    let r11 = rand2D(x0 + 1.0, y0 + 1.0, params.seed, params.time, params.speed);
    let sx = interpFunc(xf);
    let sy = interpFunc(yf);
    let nx0 = mix(r00, r10, sx);
    let nx1 = mix(r01, r11, sx);
    val = mix(nx0, nx1, sy);
  }

  if (params.useMask > 0.5) {
    let mw = u32(params.maskWidth);
    let mh = u32(params.maskHeight);
    let mu = x / params.width * params.maskWidth;
    let mv = y / params.height * params.maskHeight;
    let mx0 = u32(floor(mu));
    let my0 = u32(floor(mv));
    let mxf = fract(mu);
    let myf = fract(mv);
    let idx00 = my0 * mw + mx0;
    let idx10 = my0 * mw + min(mx0 + 1u, mw - 1u);
    let idx01 = min(my0 + 1u, mh - 1u) * mw + mx0;
    let idx11 = min(my0 + 1u, mh - 1u) * mw + min(mx0 + 1u, mw - 1u);
    let m00 = mask[idx00];
    let m10 = mask[idx10];
    let m01 = mask[idx01];
    let m11 = mask[idx11];
    var mval: f32;
    if (u32(params.interp) == ${InterpolationType.constant}u) {
      mval = m00;
    } else {
      let sx = interpFunc(mxf);
      let sy = interpFunc(myf);
      let mx0v = mix(m00, m10, sx);
      let mx1v = mix(m01, m11, sx);
      mval = mix(mx0v, mx1v, sy);
    }
    val = val * mval;
  }
  out[idx] = val;
}
`;

export const REINDEX_WGSL = /* wgsl */ `
struct ReindexParams {
  width: f32;
  height: f32;
  channels: f32;
  displacement: f32;
  mod: f32;
  pad0: f32;
  pad1: f32;
  pad2: f32;
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: ReindexParams;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  let col = textureLoad(tex, vec2<i32>(i32(x), i32(y)), 0);
  var lum = col.x;
  if (params.channels > 1.5) {
    lum = dot(col.xyz, vec3(0.2126,0.7152,0.0722));
  }
  let off = lum * params.displacement * params.mod + lum;
  let offi = u32(off);
  let xo = offi % w;
  let yo = offi % h;
  let val = textureLoad(tex, vec2<i32>(i32(xo), i32(yo)), 0);
  let ch = u32(params.channels);
  let base = (y * w + x) * ch;
  if (ch > 0u) { out[base] = val.x; }
  if (ch > 1u) { out[base + 1u] = val.y; }
  if (ch > 2u) { out[base + 2u] = val.z; }
  if (ch > 3u) { out[base + 3u] = val.w; }
}`;
export const RIPPLE_WGSL = /* wgsl */ `
struct RippleParams {
  width: f32;
  height: f32;
  channels: f32;
  displacement: f32;
  kink: f32;
  rand: f32;
  pad0: f32;
  pad1: f32;
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var refTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: RippleParams;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  let ref = textureLoad(refTex, vec2<i32>(i32(x), i32(y)), 0).x;
  let ang = ref * 6.283185307179586 * params.kink * params.rand;
  let offset = vec2<f32>(cos(ang), sin(ang)) * params.displacement;
  var samplePos = vec2<f32>(f32(x), f32(y)) + offset * vec2<f32>(params.width, params.height);
  samplePos = mod(samplePos, vec2<f32>(params.width, params.height));
  let c0 = floor(samplePos);
  let f = samplePos - c0;
  let c1 = mod(c0 + 1.0, vec2<f32>(params.width, params.height));
  let i00 = vec2<i32>(i32(c0.x), i32(c0.y));
  let i10 = vec2<i32>(i32(c1.x), i32(c0.y));
  let i01 = vec2<i32>(i32(c0.x), i32(c1.y));
  let i11 = vec2<i32>(i32(c1.x), i32(c1.y));
  let s00 = textureLoad(tex, i00, 0);
  let s10 = textureLoad(tex, i10, 0);
  let s01 = textureLoad(tex, i01, 0);
  let s11 = textureLoad(tex, i11, 0);
  let sx = f.x;
  let sy = f.y;
  let mx0 = mix(s00, s10, sx);
  let mx1 = mix(s01, s11, sx);
  let val = mix(mx0, mx1, sy);
  let ch = u32(params.channels);
  let base = (y * w + x) * ch;
  if (ch > 0u) { out[base] = val.x; }
  if (ch > 1u) { out[base + 1u] = val.y; }
  if (ch > 2u) { out[base + 2u] = val.z; }
  if (ch > 3u) { out[base + 3u] = val.w; }
}`;
export const COLOR_MAP_WGSL = /* wgsl */ `
struct ColorMapParams {
  width: f32;
  height: f32;
  channels: f32;
  displacement: f32;
  horizontal: f32;
  clutWidth: f32;
  clutHeight: f32;
  clutChannels: f32;
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var clut: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: ColorMapParams;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  let col = textureLoad(tex, vec2<i32>(i32(x), i32(y)), 0);
  var lum = col.x;
  if (params.channels > 1.5) {
    lum = dot(col.xyz, vec3(0.2126,0.7152,0.0722));
  }
  let ref = lum * params.displacement;
  let xi = (i32(x) + i32(floor(ref * (params.width - 1.0)))) % i32(w);
  let yi = params.horizontal > 0.5
    ? i32(y)
    : (i32(y) + i32(floor(ref * (params.height - 1.0)))) % i32(h);
  let sx = u32(floor(f32(xi) * params.clutWidth / params.width));
  let sy = u32(floor(f32(yi) * params.clutHeight / params.height));
  let c = textureLoad(clut, vec2<i32>(i32(sx), i32(sy)), 0);
  let cc = u32(params.clutChannels);
  let base = (y * w + x) * cc;
  if (cc > 0u) { out[base] = c.x; }
  if (cc > 1u) { out[base + 1u] = c.y; }
  if (cc > 2u) { out[base + 2u] = c.z; }
  if (cc > 3u) { out[base + 3u] = c.w; }
}`;
export const VIGNETTE_WGSL = /* wgsl */ `
struct VignetteParams {
  width: f32;
  height: f32;
  channels: f32;
  brightness: f32;
  alpha: f32;
  pad0: f32;
  pad1: f32;
  pad2: f32;
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: VignetteParams;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  let col = textureLoad(tex, vec2<i32>(i32(x), i32(y)), 0);
  let uv = vec2<f32>(f32(x)/params.width, f32(y)/params.height);
  let dist = distance(uv, vec2<f32>(0.5,0.5)) / length(vec2<f32>(0.5,0.5));
  let vignetted = mix(col, vec4<f32>(params.brightness), dist*dist);
  let val = mix(col, vignetted, params.alpha);
  let ch = u32(params.channels);
  let base = (y * w + x) * ch;
  if (ch > 0u) { out[base] = val.x; }
  if (ch > 1u) { out[base + 1u] = val.y; }
  if (ch > 2u) { out[base + 2u] = val.z; }
  if (ch > 3u) { out[base + 3u] = val.w; }
}`;
export const DITHER_WGSL = /* wgsl */ `
struct DitherParams {
  width: f32;
  height: f32;
  channels: f32;
  levels: f32;
  pad0: f32;
  pad1: f32;
  pad2: f32;
  pad3: f32;
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var noise: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: DitherParams;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  var c = textureLoad(tex, vec2<i32>(i32(x), i32(y)), 0);
  let n = textureLoad(noise, vec2<i32>(i32(x), i32(y)), 0).x - 0.5;
  let inv = 1.0 / params.levels;
  c = floor(clamp(c + n * inv, 0.0, 1.0) * params.levels) * inv;
  let ch = u32(params.channels);
  let base = (y * w + x) * ch;
  if (ch > 0u) { out[base] = c.x; }
  if (ch > 1u) { out[base + 1u] = c.y; }
  if (ch > 2u) { out[base + 2u] = c.z; }
  if (ch > 3u) { out[base + 3u] = c.w; }
}`;
export const ADJUST_BRIGHTNESS_WGSL = /* wgsl */ `
struct BrightnessParams {
  width: f32;
  height: f32;
  channels: f32;
  amount: f32;
  pad0: f32;
  pad1: f32;
  pad2: f32;
  pad3: f32;
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: BrightnessParams;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  let col = textureLoad(tex, vec2<i32>(i32(x), i32(y)), 0) + params.amount;
  let val = clamp(col, -1.0, 1.0);
  let ch = u32(params.channels);
  let base = (y * w + x) * ch;
  if (ch > 0u) { out[base] = val.x; }
  if (ch > 1u) { out[base + 1u] = val.y; }
  if (ch > 2u) { out[base + 2u] = val.z; }
  if (ch > 3u) { out[base + 3u] = val.w; }
}`;
export const ADJUST_CONTRAST_WGSL = /* wgsl */ `
struct ContrastParams {
  width: f32;
  height: f32;
  channels: f32;
  amount: f32;
  mean0: f32;
  mean1: f32;
  mean2: f32;
  pad0: f32;
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: ContrastParams;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  let col = textureLoad(tex, vec2<i32>(i32(x), i32(y)), 0);
  var val = col;
  if (params.channels > 0.5) {
    val.x = clamp((col.x - params.mean0) * params.amount + params.mean0, 0.0, 1.0);
  }
  if (params.channels > 1.5) {
    val.y = clamp((col.y - params.mean1) * params.amount + params.mean1, 0.0, 1.0);
  }
  if (params.channels > 2.5) {
    val.z = clamp((col.z - params.mean2) * params.amount + params.mean2, 0.0, 1.0);
  }
  let ch = u32(params.channels);
  let base = (y * w + x) * ch;
  if (ch > 0u) { out[base] = val.x; }
  if (ch > 1u) { out[base + 1u] = val.y; }
  if (ch > 2u) { out[base + 2u] = val.z; }
  if (ch > 3u) { out[base + 3u] = val.w; }
}`;
export const ROTATE_WGSL = /* wgsl */ `
struct RotateParams {
  width: f32;
  height: f32;
  channels: f32;
  angle: f32;
  pad0: f32;
  pad1: f32;
  pad2: f32;
  pad3: f32;
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: RotateParams;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  var uv = vec2<f32>(f32(x)/params.width, f32(y)/params.height);
  uv = uv - vec2<f32>(0.5, 0.5);
  let c = cos(params.angle);
  let s = sin(params.angle);
  uv = vec2<f32>(c * uv.x + s * uv.y, -s * uv.x + c * uv.y) + vec2<f32>(0.5, 0.5);
  uv = fract(uv);
  var samplePos = uv * vec2<f32>(params.width, params.height);
  let c0 = floor(samplePos);
  let f = samplePos - c0;
  let c1 = mod(c0 + 1.0, vec2<f32>(params.width, params.height));
  let i00 = vec2<i32>(i32(c0.x), i32(c0.y));
  let i10 = vec2<i32>(i32(c1.x), i32(c0.y));
  let i01 = vec2<i32>(i32(c0.x), i32(c1.y));
  let i11 = vec2<i32>(i32(c1.x), i32(c1.y));
  let s00 = textureLoad(tex, i00, 0);
  let s10 = textureLoad(tex, i10, 0);
  let s01 = textureLoad(tex, i01, 0);
  let s11 = textureLoad(tex, i11, 0);
  let sx = f.x;
  let sy = f.y;
  let mx0 = mix(s00, s10, sx);
  let mx1 = mix(s01, s11, sx);
  let val = mix(mx0, mx1, sy);
  let ch = u32(params.channels);
  let base = (y * w + x) * ch;
  if (ch > 0u) { out[base] = val.x; }
  if (ch > 1u) { out[base + 1u] = val.y; }
  if (ch > 2u) { out[base + 2u] = val.z; }
  if (ch > 3u) { out[base + 3u] = val.w; }
}`;
