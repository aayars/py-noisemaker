// Copies the input texture into a larger tile by wrapping coordinates with an optional offset.
// Mirrors the CPU expand_tile effect used by the reverb pipeline and other helpers.

struct ExpandTileParams {
  dims0 : vec4<i32>;  // inWidth, inHeight, outWidth, outHeight
  dims1 : vec4<i32>;  // channels, xOffset, yOffset, unused
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ExpandTileParams;

fn clamp_channel_index(index : i32, channels : i32) -> u32 {
  let count = max(channels, 1);
  if (index <= 0) {
    return 0u;
  }
  if (index >= count) {
    return u32(count - 1);
  }
  return u32(index);
}

fn sample_channel(value : vec4<f32>, channel : i32, channel_count : i32) -> f32 {
  let idx = clamp_channel_index(channel, channel_count);
  switch idx {
    case 0u: { return value.x; }
    case 1u: { return value.y; }
    case 2u: { return value.z; }
    default: { return value.w; }
  }
}

fn wrap_coord(coord : i32, size : i32) -> i32 {
  if (size <= 0) {
    return 0;
  }
  var wrapped = coord % size;
  if (wrapped < 0) {
    wrapped = wrapped + size;
  }
  return wrapped;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let out_width = max(params.dims0.z, 1);
  let out_height = max(params.dims0.w, 1);
  if (i32(global_id.x) >= out_width || i32(global_id.y) >= out_height) {
    return;
  }

  let in_width = max(params.dims0.x, 1);
  let in_height = max(params.dims0.y, 1);
  let channel_count = max(min(params.dims1.x, 4), 1);
  let x_offset = params.dims1.y;
  let y_offset = params.dims1.z;

  let src_x = wrap_coord(i32(global_id.x) + x_offset, in_width);
  let src_y = wrap_coord(i32(global_id.y) + y_offset, in_height);

  let coords = vec2<i32>(src_x, src_y);
  let texel = textureLoad(input_texture, coords, 0);

  let channels_u = u32(channel_count);
  let width_u = u32(out_width);
  let base_index = (global_id.y * width_u + global_id.x) * channels_u;

  for (var channel : u32 = 0u; channel < channels_u; channel = channel + 1u) {
    let value = sample_channel(texel, i32(channel), channel_count);
    output_buffer[base_index + channel] = value;
  }
}
