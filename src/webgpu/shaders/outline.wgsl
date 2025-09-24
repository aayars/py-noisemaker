struct OutlineParams {
  width : f32,
  height : f32,
  channels : f32,
  invert : f32,
  _pad0 : f32,
  _pad1 : f32,
  _pad2 : f32,
  _pad3 : f32,
};

@group(0) @binding(0) var base_tex : texture_2d<f32>;
@group(0) @binding(1) var edge_tex : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(3) var<uniform> params : OutlineParams;

fn as_u32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn clamp01(value : f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

fn fetch_component(texel : vec4<f32>, index : u32) -> f32 {
  switch index {
    case 0u: { return texel.x; }
    case 1u: { return texel.y; }
    case 2u: { return texel.z; }
    default: { return texel.w; }
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = as_u32(params.width);
  let height : u32 = as_u32(params.height);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channel_count : u32 = max(as_u32(params.channels + 0.5), 1u);
  let coords = vec2<i32>(i32(gid.x), i32(gid.y));

  let src_texel : vec4<f32> = textureLoad(base_tex, coords, 0);
  let edge_texel : vec4<f32> = textureLoad(edge_tex, coords, 0);
  let invert_flag : f32 = params.invert;
  let raw_mask : f32 = clamp01(edge_texel.x);
  let mask : f32 = select(raw_mask, 1.0 - raw_mask, invert_flag >= 0.5);

  let pixel_index : u32 = gid.y * width + gid.x;
  let base_index : u32 = pixel_index * channel_count;

  if (channel_count > 0u) {
    let value : f32 = clamp01(fetch_component(src_texel, 0u) * mask);
    output_buffer[base_index] = value;
  }
  if (channel_count > 1u) {
    let value : f32 = clamp01(fetch_component(src_texel, 1u) * mask);
    output_buffer[base_index + 1u] = value;
  }
  if (channel_count > 2u) {
    let value : f32 = clamp01(fetch_component(src_texel, 2u) * mask);
    output_buffer[base_index + 2u] = value;
  }
  if (channel_count > 3u) {
    let value : f32 = clamp01(fetch_component(src_texel, 3u) * mask);
    output_buffer[base_index + 3u] = value;
    if (channel_count > 4u) {
      var extra_channel : u32 = 4u;
      loop {
        if (extra_channel >= channel_count) {
          break;
        }
        output_buffer[base_index + extra_channel] = value;
        extra_channel = extra_channel + 1u;
      }
    }
  }
}
