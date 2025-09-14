struct GlyphMapParams {
  width: f32,
  height: f32,
  channels: f32,
  glyphWidth: f32,
  glyphHeight: f32,
  glyphCount: f32,
  colorize: f32,
  zoom: f32,
  alpha: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var glyphs: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: GlyphMapParams;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  let gw = u32(params.glyphWidth);
  let gh = u32(params.glyphHeight);
  let zoom = params.zoom;
  let scaledX = u32(floor(f32(x) / zoom));
  let scaledY = u32(floor(f32(y) / zoom));
  let cellX = scaledX / gw;
  let cellY = scaledY / gh;
  let sx = min(u32(params.width) - 1u, cellX * gw);
  let sy = min(u32(params.height) - 1u, cellY * gh);
  let sample = textureLoad(tex, vec2<i32>(i32(sx), i32(sy)), 0);
  var bright = sample.x;
  if (params.channels > 1.5) {
    bright = 0.2126 * sample.x + 0.7152 * sample.y + 0.0722 * sample.z;
  }
  let gCount = u32(params.glyphCount);
  let gIndex = min(gCount - 1u, u32(floor(bright * f32(gCount))));
  let localX = scaledX - sx;
  let localY = scaledY - sy;
  let glyphVal = textureLoad(
    glyphs,
    vec2<i32>(i32(localX), i32(localY + gIndex * gh)),
    0,
  ).x;
  var col: vec4<f32>;
  if (params.colorize > 0.5) {
    col = vec4<f32>(glyphVal) * sample;
  } else {
    col = vec4<f32>(glyphVal);
  }
  let orig = textureLoad(tex, vec2<i32>(i32(x), i32(y)), 0);
  let final = mix(orig, col, params.alpha);
  let ch = u32(params.channels);
  let base = (y * w + x) * ch;
  if (ch > 0u) { out[base] = final.x; }
  if (ch > 1u) { out[base + 1u] = final.y; }
  if (ch > 2u) { out[base + 2u] = final.z; }
  if (ch > 3u) { out[base + 3u] = final.w; }
}
