struct KaleidoParams {
  width: f32,
  height: f32,
  channels: f32,
  sides: f32,
  blendEdges: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> radius: array<f32>;
@group(0) @binding(3) var<storage, read> fader: array<f32>;
@group(0) @binding(4) var<uniform> params: KaleidoParams;

fn fmod(a: f32, b: f32) -> f32 {
  return a - b * floor(a / b);
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  let idx = y * w + x;
  let xi = f32(x) / (params.width - 1.0) - 0.5;
  let yi = f32(y) / (params.height - 1.0) - 0.5;
  let radiusVal = radius[idx];
  let step = (2.0 * 3.141592653589793) / params.sides;
  var a = atan2(yi, xi) + 3.141592653589793 / 2.0;
  a = fmod(a, step);
  if (a < 0.0) { a = a + step; }
  a = abs(a - step / 2.0);
  var nx = radiusVal * params.width * sin(a);
  var ny = radiusVal * params.height * cos(a);
  if (params.blendEdges > 0.5) {
    let fade = fader[idx];
    nx = nx * (1.0 - fade) + f32(x) * fade;
    ny = ny * (1.0 - fade) + f32(y) * fade;
  }
  var sx = i32(floor(nx));
  var sy = i32(floor(ny));
  sx = ((sx % i32(w)) + i32(w)) % i32(w);
  sy = ((sy % i32(h)) + i32(h)) % i32(h);
  let col = textureLoad(tex, vec2<i32>(sx, sy), 0);
  let ch = u32(params.channels);
  let base = idx * ch;
  if (ch > 0u) { out[base] = col.x; }
  if (ch > 1u) { out[base + 1u] = col.y; }
  if (ch > 2u) { out[base + 2u] = col.z; }
  if (ch > 3u) { out[base + 3u] = col.w; }
}
