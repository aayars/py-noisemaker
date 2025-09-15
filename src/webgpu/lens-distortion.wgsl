struct LensParams {
  width: f32,
  height: f32,
  channels: f32,
  displacement: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
  pad3: f32,
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: LensParams;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  let uv = vec2<f32>(f32(x)/params.width, f32(y)/params.height);
  let xDist = uv.x - 0.5;
  let yDist = uv.y - 0.5;
  let maxDist = sqrt(0.5 * 0.5 + 0.5 * 0.5);
  let centerDist = 1.0 - length(vec2<f32>(xDist, yDist)) / maxDist;
  let zoom = select(0.0, params.displacement * -0.25, params.displacement < 0.0);
  let xOff = (uv.x - xDist * zoom - xDist * centerDist * centerDist * params.displacement) * params.width;
  let yOff = (uv.y - yDist * zoom - yDist * centerDist * centerDist * params.displacement) * params.height;
  var sx = i32(floor(xOff));
  var sy = i32(floor(yOff));
  let wi = i32(w);
  let hi = i32(h);
  sx = ((sx % wi) + wi) % wi;
  sy = ((sy % hi) + hi) % hi;
  let col = textureLoad(tex, vec2<i32>(sx, sy), 0);
  let ch = u32(params.channels);
  let base = (y * w + x) * ch;
  if (ch > 0u) { out[base] = col.x; }
  if (ch > 1u) { out[base + 1u] = col.y; }
  if (ch > 2u) { out[base + 2u] = col.z; }
  if (ch > 3u) { out[base + 3u] = col.w; }
}
