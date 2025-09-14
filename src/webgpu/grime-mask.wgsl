struct GrimeMaskParams {
  width: f32,
  height: f32,
  time: f32,
  speed: f32,
};
@group(0) @binding(0) var outTex: texture_storage_2d<r32float, write>;
@group(0) @binding(1) var<uniform> params: GrimeMaskParams;

fn mod289_3(x: vec3<f32>) -> vec3<f32> { return x - floor(x * (1.0 / 289.0)) * 289.0; }
fn mod289_4(x: vec4<f32>) -> vec4<f32> { return x - floor(x * (1.0 / 289.0)) * 289.0; }
fn permute3(x: vec3<f32>) -> vec3<f32> { return mod289_3(((x * 34.0) + 1.0) * x); }
fn permute4(x: vec4<f32>) -> vec4<f32> { return mod289_4(((x * 34.0) + 1.0) * x); }
fn taylorInvSqrt(r: vec4<f32>) -> vec4<f32> { return 1.79284291400159 - 0.85373472095314 * r; }
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
  var norm = taylorInvSqrt(vec4<f32>(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;
  var m = max(0.6 - vec4<f32>(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), vec4<f32>(0.0));
  m = m * m;
  return 42.0 * dot(m*m, vec4<f32>(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  var freq = 5.0;
  var amp = 1.0;
  var total = 0.0;
  var val = 0.0;
  for (var o = 0; o < 8; o = o + 1) {
    let fx = freq;
    let fy = freq;
    let xx = f32(x) / params.width * fx;
    let yy = f32(y) / params.height * fy;
    let ang = 6.283185307179586 * params.time;
    let z = cos(ang) * params.speed;
    let s = f32(o);
    let n = (simplex3(vec3<f32>(xx + s, yy + s, z)) + 1.0) * 0.5;
    val = val + n * amp;
    total = total + amp;
    amp = amp * 0.5;
    freq = freq * 2.0;
  }
  val = val / total;
  textureStore(outTex, vec2<i32>(i32(x), i32(y)), vec4<f32>(val, 0.0, 0.0, 1.0));
}
