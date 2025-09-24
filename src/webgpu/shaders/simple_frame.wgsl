struct SimpleFrameParams {
  width : f32,
  height : f32,
  channels : f32,
  brightness : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : SimpleFrameParams;

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

fn axis_min_max(size : u32, size_f : f32) -> vec2<f32> {
  if (size <= 1u) {
    return vec2<f32>(0.5, 0.5);
  }
  if ((size & 1u) == 0u) {
    return vec2<f32>(0.0, 0.5);
  }
  let half_floor = f32(size / 2u);
  let min_val = 0.5 / size_f;
  let max_val = (half_floor - 0.5) / size_f;
  return vec2<f32>(min_val, max_val);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let width_u = max(u32(params.width + 0.5), 1u);
  let height_u = max(u32(params.height + 0.5), 1u);
  if (global_id.x >= width_u || global_id.y >= height_u) {
    return;
  }

  let channels = max(u32(params.channels + 0.5), 1u);
  let width_f = max(f32(width_u), 1.0);
  let height_f = max(f32(height_u), 1.0);
  let half_width_u = width_u / 2u;
  let half_height_u = height_u / 2u;
  let half_width_f = f32(half_width_u);
  let half_height_f = f32(half_height_u);
  let center_x = width_f * 0.5;
  let center_y = height_f * 0.5;

  let sample_x = (global_id.x + half_width_u) % width_u;
  let sample_y = (global_id.y + half_height_u) % height_u;
  let x_f = f32(sample_x);
  let y_f = f32(sample_y);

  let x0 = x_f - center_x - half_width_f;
  let x1 = x_f - center_x + half_width_f;
  let y0 = y_f - center_y - half_height_f;
  let y1 = y_f - center_y + half_height_f;

  let dx = min(abs(x0), abs(x1)) / width_f;
  let dy = min(abs(y0), abs(y1)) / height_f;

  let axis_x = axis_min_max(width_u, width_f);
  let axis_y = axis_min_max(height_u, height_f);
  let min_dist = max(axis_x.x, axis_y.x);
  let max_dist = max(axis_x.y, axis_y.y);
  let dist = max(dx, dy);
  let delta = max_dist - min_dist;

  var value = 0.0;
  if (delta > 0.0) {
    let normalized = clamp((dist - min_dist) / delta, 0.0, 1.0);
    value = sqrt(normalized);
  } else {
    value = sqrt(max(dist, 0.0));
  }

  let scaled = value * 0.55;
  var mask = floor(scaled + 0.5);
  mask = clamp(mask, 0.0, 1.0);

  let pixel_index = global_id.y * width_u + global_id.x;
  let base_index = pixel_index * channels;
  let sample = textureLoad(
    input_texture,
    vec2<i32>(i32(global_id.x), i32(global_id.y)),
    0,
  );

  for (var channel : u32 = 0u; channel < channels; channel = channel + 1u) {
    let src = get_component(sample, channel);
    let blended = src * (1.0 - mask) + params.brightness * mask;
    output_buffer[base_index + channel] = blended;
  }
}
