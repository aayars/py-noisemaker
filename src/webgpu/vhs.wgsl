struct VHSParams {
  width: f32,
  height: f32,
  channels: f32,
  pad0: f32,
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var scanTex: texture_2d<f32>;
@group(0) @binding(2) var gradTex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: VHSParams;

fn gradTransform(g: f32) -> f32 {
  var v = g - 0.5;
  if (v < 0.0) { v = 0.0; }
  v = min(v * 2.0, 1.0);
  return v;
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  let ch = u32(params.channels);

  let n1 = textureLoad(scanTex, vec2<i32>(i32(x), i32(y)), 0).x;
  let g1 = gradTransform(textureLoad(gradTex, vec2<i32>(i32(x), i32(y)), 0).x);
  let xOff = i32(floor(n1 * params.width * g1 * g1));
  let srcX = (i32(w) + i32(x) - xOff) % i32(w);
  let n2 = textureLoad(scanTex, vec2<i32>(srcX, i32(y)), 0).x;
  let g2 = gradTransform(textureLoad(gradTex, vec2<i32>(srcX, i32(y)), 0).x);
  let srcCol = textureLoad(tex, vec2<i32>(srcX, i32(y)), 0);
  let noiseCol = vec4<f32>(n2, n2, n2, n2);
  let blended = srcCol * (1.0 - g2) + noiseCol * g2;

  let base = (y * w + x) * ch;
  if (ch > 0u) { out[base] = blended.x; }
  if (ch > 1u) { out[base + 1u] = blended.y; }
  if (ch > 2u) { out[base + 2u] = blended.z; }
  if (ch > 3u) { out[base + 3u] = blended.w; }
}
