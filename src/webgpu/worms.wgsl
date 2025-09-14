struct WormParams {
  width: f32,
  height: f32,
  channels: f32,
  count: f32,
  iterations: f32,
  quantize: f32,
  kink: f32,
  drunkenness: f32,
};
@group(0) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> strides: array<f32>;
@group(0) @binding(2) var<storage, read_write> rots: array<f32>;
@group(0) @binding(3) var<storage, read> colors: array<f32>;
@group(0) @binding(4) var<storage, read> indexBuffer: array<f32>;
@group(0) @binding(5) var<storage, read> drunk: array<f32>;
@group(0) @binding(6) var<storage, read_write> outBuffer: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params: WormParams;

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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let count = u32(params.count);
  if (idx >= count) { return; }
  let width = u32(params.width);
  let height = u32(params.height);
  let channels = u32(params.channels);
  let iterations = u32(params.iterations);

  var pos = positions[idx];
  let stride = strides[idx];
  var rot = rots[idx];

  for (var iter: u32 = 0u; iter < iterations; iter = iter + 1u) {
    rot = rot + drunk[iter * count + idx] * params.drunkenness;
    let xi = u32(floor(pos.x)) % width;
    let yi = u32(floor(pos.y)) % height;
    let pix = yi * width + xi;
    var exposure: f32 = 1.0;
    if (iterations > 1u) {
      exposure = 1.0 - abs(1.0 - (f32(iter) / f32(iterations - 1u)) * 2.0);
    }
    let base = idx * channels;
    for (var c: u32 = 0u; c < channels; c = c + 1u) {
      let val = colors[base + c] * exposure;
      atomicAddF32(&outBuffer[pix * channels + c], val);
    }
    var next = indexBuffer[pix] * params.kink + rot;
    if (params.quantize != 0.0) { next = round(next); }
    pos.y = pos.y + cos(next) * stride;
    pos.x = pos.x + sin(next) * stride;
    pos.y = fract(pos.y / params.height) * params.height;
    pos.x = fract(pos.x / params.width) * params.width;
  }
  positions[idx] = pos;
  rots[idx] = rot;
}
