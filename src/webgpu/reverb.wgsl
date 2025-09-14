struct ReverbParams {
  dstWidth: f32,
  dstHeight: f32,
  srcWidth: f32,
  srcHeight: f32,
  channels: f32,
  weight: f32,
  pad0: f32,
  pad1: f32,
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: ReverbParams;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.dstWidth);
  let h = u32(params.dstHeight);
  if (x >= w || y >= h) { return; }
  let sw = i32(params.srcWidth);
  let sh = i32(params.srcHeight);
  let xOff = sw / 2;
  let yOff = sh / 2;
  let sx = ((i32(x) + xOff) % sw + sw) % sw;
  let sy = ((i32(y) + yOff) % sh + sh) % sh;
  let val = textureLoad(tex, vec2<i32>(sx, sy), 0);
  let ch = u32(params.channels);
  let base = (y * w + x) * ch;
  let wt = params.weight;
  if (ch > 0u) { out[base] = out[base] + val.x * wt; }
  if (ch > 1u) { out[base + 1u] = out[base + 1u] + val.y * wt; }
  if (ch > 2u) { out[base + 2u] = out[base + 2u] + val.z * wt; }
  if (ch > 3u) { out[base + 3u] = out[base + 3u] + val.w * wt; }
}
