struct RGBToHSVParams {
  width: f32,
  height: f32,
  channels: f32,
  pad0: f32,
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: RGBToHSVParams;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  let col = textureLoad(tex, vec2<i32>(i32(x), i32(y)), 0);
  let r = col.x;
  let g = col.y;
  let b = col.z;
  let maxv = max(r, max(g, b));
  let minv = min(r, min(g, b));
  let d = maxv - minv;
  var hVal: f32 = 0.0;
  if (d != 0.0) {
    if (maxv == r) {
      hVal = (g - b) / d;
      if (hVal < 0.0) { hVal = hVal + 6.0; }
    } else if (maxv == g) {
      hVal = (b - r) / d + 2.0;
    } else {
      hVal = (r - g) / d + 4.0;
    }
    hVal = hVal / 6.0;
  }
  let sVal = select(0.0, d / maxv, maxv != 0.0);
  let base = (y * w + x) * 3u;
  out[base] = hVal;
  out[base + 1u] = sVal;
  out[base + 2u] = maxv;
}
