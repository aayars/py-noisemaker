// Adds a tiled downsampled layer into an accumulation buffer for the reverb effect.
// The input layer is wrapped with a half-width/height offset to match the CPU
// expand_tile behaviour before being blended into the output.

struct ReverbParams {
  width : f32;
  height : f32;
  sample_width : f32;
  sample_height : f32;
  channels : f32;
  scale : f32;
  pad0 : f32;
  pad1 : f32;
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ReverbParams;

fn clamp_channel_index(index : i32, channel_count : i32) -> u32 {
  let count = max(channel_count, 1);
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
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width = max(i32(params.width), 1);
  let height = max(i32(params.height), 1);
  if (i32(gid.x) >= width || i32(gid.y) >= height) {
    return;
  }

  let sample_width = max(i32(params.sample_width), 1);
  let sample_height = max(i32(params.sample_height), 1);
  let channel_count = max(i32(params.channels), 1);
  let channel_count_u = u32(channel_count);

  let width_u = u32(width);
  let base_index = (gid.y * width_u + gid.x) * channel_count_u;

  let x_offset = sample_width / 2;
  let y_offset = sample_height / 2;

  let src_x = wrap_coord(i32(gid.x) + x_offset, sample_width);
  let src_y = wrap_coord(i32(gid.y) + y_offset, sample_height);

  let texel = textureLoad(input_texture, vec2<i32>(src_x, src_y), 0);
  let weight = params.scale;

  for (var channel : u32 = 0u; channel < channel_count_u; channel = channel + 1u) {
    let sample_value = sample_channel(texel, i32(channel), channel_count);
    let idx = base_index + channel;
    let current = output_buffer[idx];
    output_buffer[idx] = current + sample_value * weight;
  }
}
