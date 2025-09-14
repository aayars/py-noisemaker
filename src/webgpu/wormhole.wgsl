struct WormholeParams {
  width: f32,
  height: f32,
  channels: f32,
  stride: f32,
  kink: f32,
  xOff: f32,
  yOff: f32,
  pad0: f32,
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> lum: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: WormholeParams;

fn atomicAddF32(target: ptr<storage, atomic<u32>>, value: f32) {
  var old = atomicLoad(target);
  loop {
    let f = bitcast<f32>(old) + value;
    let new = bitcast<u32>(f);
    let result = atomicCompareExchangeWeak(target, old, new);
    if (result.exchanged) { break; }
    old = result.old_value;
  }
}

fn fmod(a: f32, b: f32) -> f32 { return a - b * floor(a / b); }

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  let idx = y * w + x;
  let lumVal = lum[idx];
  let deg = lumVal * 6.283185307179586 * params.kink;
  let xo = (cos(deg) + 1.0) * params.stride;
  let yo = (sin(deg) + 1.0) * params.stride;
  var xi = u32(floor(f32(x) + xo));
  var yi = u32(floor(f32(y) + yo));
  xi = u32(fmod(f32(xi), params.width));
  yi = u32(fmod(f32(yi), params.height));
  xi = (xi + u32(params.xOff)) % w;
  yi = (yi + u32(params.yOff)) % h;
  let src = textureLoad(tex, vec2<i32>(i32(x), i32(y)), 0);
  let l2 = lumVal * lumVal;
  let ch = u32(params.channels);
  let dest = (yi * w + xi) * ch;
  if (ch > 0u) { atomicAddF32(&out[dest], src.x * l2); }
  if (ch > 1u) { atomicAddF32(&out[dest + 1u], src.y * l2); }
  if (ch > 2u) { atomicAddF32(&out[dest + 2u], src.z * l2); }
  if (ch > 3u) { atomicAddF32(&out[dest + 3u], src.w * l2); }
}
