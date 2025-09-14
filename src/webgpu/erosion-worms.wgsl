struct Worm {
  x: f32,
  y: f32,
  dx: f32,
  dy: f32,
  inertia: f32,
};
struct Params {
  width: f32,
  height: f32,
  count: u32,
  iterations: u32,
  contraction: f32,
  quantize: f32,
  channels: u32,
};
@group(0) @binding(0) var<storage, read_write> worms: array<Worm>;
@group(0) @binding(1) var<storage, read> values: array<f32>;
@group(0) @binding(2) var<storage, read> startColors: array<f32>;
@group(0) @binding(3) var<storage, read_write> outBuffer: array<atomic<f32>>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.count) { return; }
  var worm = worms[idx];
  for (var it: u32 = 0u; it < params.iterations; it = it + 1u) {
    let xi = u32(floor(worm.x)) % u32(params.width);
    let yi = u32(floor(worm.y)) % u32(params.height);
    let x1 = (xi + 1u) % u32(params.width);
    let y1 = (yi + 1u) % u32(params.height);
    let baseIdx = yi * u32(params.width) + xi;
    let x1Idx = yi * u32(params.width) + x1;
    let y1Idx = y1 * u32(params.width) + xi;
    let x1y1Idx = y1 * u32(params.width) + x1;
    let base = values[baseIdx];
    let x1v = values[x1Idx];
    let y1v = values[y1Idx];
    let x1y1v = values[x1y1Idx];
    let u = fract(worm.x);
    let v = fract(worm.y);
    var gx = mix(y1v - base, x1y1v - x1v, u);
    var gy = mix(x1v - base, x1y1v - y1v, v);
    if (params.quantize != 0.0) {
      gx = floor(gx);
      gy = floor(gy);
    }
    let len = sqrt(gx * gx + gy * gy) * params.contraction;
    if (len != 0.0) {
      let invLen = 1.0 / len;
      worm.dx = worm.dx * (1.0 - worm.inertia) + gx * invLen * worm.inertia;
      worm.dy = worm.dy * (1.0 - worm.inertia) + gy * invLen * worm.inertia;
    }
    let exposure = 1.0 - abs(1.0 - f32(it) / f32(max(params.iterations - 1u, 1u)) * 2.0);
    let outBase = (yi * u32(params.width) + xi) * params.channels;
    let colorBase = idx * params.channels;
    for (var c: u32 = 0u; c < params.channels; c = c + 1u) {
      let amt = startColors[colorBase + c] * exposure;
      atomicAdd(&outBuffer[outBase + c], amt);
    }
    worm.x = worm.x + worm.dx;
    worm.y = worm.y + worm.dy;
    if (worm.x < 0.0) {
      worm.x = worm.x + params.width;
    } else if (worm.x >= params.width) {
      worm.x = worm.x - params.width;
    }
    if (worm.y < 0.0) {
      worm.y = worm.y + params.height;
    } else if (worm.y >= params.height) {
      worm.y = worm.y - params.height;
    }
  }
  worms[idx] = worm;
}
