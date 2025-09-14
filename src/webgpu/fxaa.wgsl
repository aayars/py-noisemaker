struct FxaaParams {
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
@group(0) @binding(2) var<uniform> params: FxaaParams;

fn reflectCoord(i: i32, n: i32) -> i32 {
  if (n == 1) { return 0; }
  let m = 2 * n - 2;
  var ii = i % m;
  if (ii < 0) { ii = ii + m; }
  return select(ii, m - ii, ii >= n);
}

fn sampleReflect(x: i32, y: i32) -> vec4<f32> {
  let rx = reflectCoord(x, i32(params.width));
  let ry = reflectCoord(y, i32(params.height));
  return textureLoad(tex, vec2<i32>(rx, ry), 0);
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  let w = u32(params.width);
  let h = u32(params.height);
  if (gid.x >= w || gid.y >= h) { return; }
  let ch = u32(params.channels);
  let base = (u32(y) * w + u32(x)) * ch;
  if (ch == 1u) {
    let c = sampleReflect(x, y).x;
    let n = sampleReflect(x, y - 1).x;
    let s = sampleReflect(x, y + 1).x;
    let wv = sampleReflect(x - 1, y).x;
    let e = sampleReflect(x + 1, y).x;
    let wC = 1.0;
    let wN = exp(-abs(c - n));
    let wS = exp(-abs(c - s));
    let wW = exp(-abs(c - wv));
    let wE = exp(-abs(c - e));
    let sum = wC + wN + wS + wW + wE + 1e-10;
    out[base] = (c * wC + n * wN + s * wS + wv * wW + e * wE) / sum;
  } else if (ch == 2u) {
    let c = sampleReflect(x, y);
    let lum = c.x;
    let alpha = c.y;
    let lN = sampleReflect(x, y - 1).x;
    let lS = sampleReflect(x, y + 1).x;
    let lW = sampleReflect(x - 1, y).x;
    let lE = sampleReflect(x + 1, y).x;
    let wC = 1.0;
    let wN = exp(-abs(lum - lN));
    let wS = exp(-abs(lum - lS));
    let wW = exp(-abs(lum - lW));
    let wE = exp(-abs(lum - lE));
    let sum = wC + wN + wS + wW + wE + 1e-10;
    out[base] = (lum * wC + lN * wN + lS * wS + lW * wW + lE * wE) / sum;
    out[base + 1u] = alpha;
  } else {
    let lumWeights = vec3<f32>(0.299, 0.587, 0.114);
    let c = sampleReflect(x, y);
    let n = sampleReflect(x, y - 1);
    let s = sampleReflect(x, y + 1);
    let wv = sampleReflect(x - 1, y);
    let e = sampleReflect(x + 1, y);
    let lC = dot(c.xyz, lumWeights);
    let lN = dot(n.xyz, lumWeights);
    let lS = dot(s.xyz, lumWeights);
    let lW = dot(wv.xyz, lumWeights);
    let lE = dot(e.xyz, lumWeights);
    let wC = 1.0;
    let wN = exp(-abs(lC - lN));
    let wS = exp(-abs(lC - lS));
    let wW = exp(-abs(lC - lW));
    let wE = exp(-abs(lC - lE));
    let sum = wC + wN + wS + wW + wE + 1e-10;
    let col = (c.xyz * wC + n.xyz * wN + s.xyz * wS + wv.xyz * wW + e.xyz * wE) / sum;
    out[base] = col.x;
    out[base + 1u] = col.y;
    out[base + 2u] = col.z;
    if (ch > 3u) { out[base + 3u] = c.w; }
  }
}
