struct SnowParams {
  base_width : f32,
  base_height : f32,
  base_channels : f32,
  static_width : f32,
  static_height : f32,
  static_channels : f32,
  limiter_width : f32,
  limiter_height : f32,
  limiter_channels : f32,
  alpha_width : f32,
  alpha_height : f32,
  alpha_channels : f32,
  alpha_value : f32,
  alpha_is_texture : f32,
  _pad0 : f32,
  _pad1 : f32,
};

@group(0) @binding(0) var base_texture : texture_2d<f32>;
@group(0) @binding(1) var static_texture : texture_2d<f32>;
@group(0) @binding(2) var limiter_texture : texture_2d<f32>;
@group(0) @binding(3) var alpha_texture : texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(5) var<uniform> params : SnowParams;

fn as_u32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn wrap_coord(coord : u32, limit : u32) -> i32 {
  if (limit == 0u) {
    return 0;
  }
  return i32(coord % limit);
}

fn get_component(value : vec4<f32>, index : u32) -> f32 {
  switch index {
    case 0u: {
      return value.x;
    }
    case 1u: {
      return value.y;
    }
    case 2u: {
      return value.z;
    }
    default: {
      return value.w;
    }
  }
}

fn sample_channel(value : vec4<f32>, channel : u32, channel_count : u32) -> f32 {
  let safe_index = select(0u, channel, channel < channel_count);
  return get_component(value, safe_index);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let width = as_u32(params.base_width);
  let height = as_u32(params.base_height);
  if (global_id.x >= width || global_id.y >= height) {
    return;
  }

  let base_channels = max(as_u32(params.base_channels + 0.5), 1u);
  let static_channels = max(as_u32(params.static_channels + 0.5), 1u);
  let limiter_channels = max(as_u32(params.limiter_channels + 0.5), 1u);
  let alpha_channels = max(as_u32(params.alpha_channels + 0.5), 1u);

  let base_coords = vec2<i32>(i32(global_id.x), i32(global_id.y));

  let static_width = max(as_u32(params.static_width), 1u);
  let static_height = max(as_u32(params.static_height), 1u);
  let static_coords = vec2<i32>(
    wrap_coord(global_id.x, static_width),
    wrap_coord(global_id.y, static_height),
  );

  let limiter_width = max(as_u32(params.limiter_width), 1u);
  let limiter_height = max(as_u32(params.limiter_height), 1u);
  let limiter_coords = vec2<i32>(
    wrap_coord(global_id.x, limiter_width),
    wrap_coord(global_id.y, limiter_height),
  );

  let base_sample = textureLoad(base_texture, base_coords, 0);
  let static_sample = textureLoad(static_texture, static_coords, 0);
  let limiter_sample = textureLoad(limiter_texture, limiter_coords, 0);

  let limiter_value = sample_channel(limiter_sample, 0u, limiter_channels);

  var alpha_value = params.alpha_value;
  if (params.alpha_is_texture > 0.5) {
    let alpha_width = max(as_u32(params.alpha_width), 1u);
    let alpha_height = max(as_u32(params.alpha_height), 1u);
    let alpha_coords = vec2<i32>(
      wrap_coord(global_id.x, alpha_width),
      wrap_coord(global_id.y, alpha_height),
    );
    let alpha_sample = textureLoad(alpha_texture, alpha_coords, 0);
    alpha_value = sample_channel(alpha_sample, 0u, alpha_channels);
  }

  let mask = limiter_value * alpha_value;
  let base_index = (global_id.y * width + global_id.x) * base_channels;

  for (var channel : u32 = 0u; channel < base_channels; channel = channel + 1u) {
    let base_val = sample_channel(base_sample, channel, base_channels);
    let noise_val = sample_channel(static_sample, channel, static_channels);
    let result = base_val * (1.0 - mask) + noise_val * mask;
    output_buffer[base_index + channel] = result;
  }
}
