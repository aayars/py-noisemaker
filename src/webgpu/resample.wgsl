struct ResampleParams {
  srcWidth: f32;
  srcHeight: f32;
  dstWidth: f32;
  dstHeight: f32;
  channels: f32;
  interp: f32;
  pad0: f32;
  pad1: f32;
};
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: ResampleParams;

fn sampleWrapped(ix: i32, iy: i32) -> vec4<f32> {
  let w = i32(params.srcWidth);
  let h = i32(params.srcHeight);
  let x = ((ix % w) + w) % w;
  let y = ((iy % h) + h) % h;
  return textureLoad(src, vec2<i32>(x, y), 0);
}

fn cubic(a: vec4<f32>, b: vec4<f32>, c: vec4<f32>, d: vec4<f32>, t: f32) -> vec4<f32> {
  let t2 = t * t;
  let t3 = t2 * t;
  let a0 = d - c - a + b;
  let a1 = a - b - a0;
  let a2 = c - a;
  let a3 = b;
  return a0 * t3 + a1 * t2 + a2 * t + a3;
}

fn interpFunc(t: f32) -> f32 {
  let i = u32(params.interp);
  if (i == 1u) { return t; }
  if (i == 2u) { return 0.5 - cos(t * 3.141592653589793) * 0.5; }
  return t * t * (3.0 - 2.0 * t);
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let nw = u32(params.dstWidth);
  let nh = u32(params.dstHeight);
  if (x >= nw || y >= nh) { return; }
  let gx = f32(x) * params.srcWidth / params.dstWidth;
  let gy = f32(y) * params.srcHeight / params.dstHeight;
  let x0 = i32(floor(gx));
  let y0 = i32(floor(gy));
  let xf = gx - floor(gx);
  let yf = gy - floor(gy);
  var val: vec4<f32>;
  let interp = u32(params.interp);
  if (interp == 0u) {
    val = sampleWrapped(i32(round(gx)), i32(round(gy)));
  } else if (interp == 1u || interp == 2u) {
    let v00 = sampleWrapped(x0, y0);
    let v10 = sampleWrapped(x0 + 1, y0);
    let v01 = sampleWrapped(x0, y0 + 1);
    let v11 = sampleWrapped(x0 + 1, y0 + 1);
    let sx = interpFunc(xf);
    let sy = interpFunc(yf);
    let mx0 = mix(v00, v10, sx);
    let mx1 = mix(v01, v11, sx);
    val = mix(mx0, mx1, sy);
  } else {
    var rows: array<vec4<f32>, 4>;
    for (var m: i32 = -1; m < 3; m = m + 1) {
      let r0 = sampleWrapped(x0 - 1, y0 + m);
      let r1 = sampleWrapped(x0, y0 + m);
      let r2 = sampleWrapped(x0 + 1, y0 + m);
      let r3 = sampleWrapped(x0 + 2, y0 + m);
      rows[(m + 1)] = cubic(r0, r1, r2, r3, xf);
    }
    val = cubic(rows[0], rows[1], rows[2], rows[3], yf);
  }
  let ch = u32(params.channels);
  let base = (y * nw + x) * ch;
  if (ch > 0u) { out[base] = val.x; }
  if (ch > 1u) { out[base + 1u] = val.y; }
  if (ch > 2u) { out[base + 2u] = val.z; }
  if (ch > 3u) { out[base + 3u] = val.w; }
}
