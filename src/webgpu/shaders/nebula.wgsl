struct NebulaParams {
  width : f32,
  height : f32,
  channels : f32,
  _pad : f32,
};

@group(0) @binding(0) var base_texture : texture_2d<f32>;
@group(0) @binding(1) var tint_texture : texture_2d<f32>;
@group(0) @binding(2) var mask_texture : texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(4) var<uniform> params : NebulaParams;

fn as_u32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn sample_component(value : vec4<f32>, index : u32, max_channels : u32) -> f32 {
  switch index {
    case 0u: {
      return value.x;
    }
    case 1u: {
      return select(value.x, value.y, max_channels > 1u);
    }
    case 2u: {
      return select(value.x, value.z, max_channels > 2u);
    }
    default: {
      return select(value.x, value.w, max_channels > 3u);
    }
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let width = as_u32(params.width + 0.5);
  let height = as_u32(params.height + 0.5);
  if (global_id.x >= width || global_id.y >= height) {
    return;
  }

  let channel_count = as_u32(params.channels + 0.5);
  if (channel_count == 0u) {
    return;
  }

  let coords = vec2<i32>(i32(global_id.x), i32(global_id.y));
  let base_sample = textureLoad(base_texture, coords, 0);
  let tint_sample = textureLoad(tint_texture, coords, 0);
  let mask_sample = textureLoad(mask_texture, coords, 0);

  let mask_value = mask_sample.x;
  let multiplier = 1.0 - mask_value;

  let pixel_index = (global_id.y * width + global_id.x) * channel_count;
  for (var channel : u32 = 0u; channel < channel_count; channel = channel + 1u) {
    let base_val = sample_component(base_sample, channel, channel_count);
    let tint_val = sample_component(tint_sample, channel, channel_count);
    let combined = base_val * multiplier + tint_val;
    output_buffer[pixel_index + channel] = combined;
  }
}
