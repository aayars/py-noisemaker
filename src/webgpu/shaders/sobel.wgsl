struct SobelParams {
  width : f32,
  height : f32,
  channels : f32,
  _padding : f32,
  _padding2 : vec4<f32>,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : SobelParams;

const SOBEL_X : array<f32, 9> = array<f32, 9>(
  -1.0, 0.0, 1.0,
  -2.0, 0.0, 2.0,
  -1.0, 0.0, 1.0,
);

const SOBEL_Y : array<f32, 9> = array<f32, 9>(
  -1.0, -2.0, -1.0,
  0.0, 0.0, 0.0,
  1.0, 2.0, 1.0,
);

fn clamp_coord(value : i32, limit : i32) -> i32 {
  if (value < 0) {
    return 0;
  }
  if (value >= limit) {
    return limit - 1;
  }
  return value;
}

fn get_component(v : vec4<f32>, index : u32) -> f32 {
  switch index {
    case 0u: {
      return v.x;
    }
    case 1u: {
      return v.y;
    }
    case 2u: {
      return v.z;
    }
    default: {
      return v.w;
    }
  }
}

fn as_u32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let width : u32 = as_u32(params.width);
  let height : u32 = as_u32(params.height);
  if (global_id.x >= width || global_id.y >= height) {
    return;
  }

  let channel_count : u32 = max(as_u32(params.channels + 0.5), 1u);
  let pixel_index : u32 = global_id.y * width + global_id.x;
  let base_index : u32 = pixel_index * channel_count;

  let width_i : i32 = i32(width);
  let height_i : i32 = i32(height);
  let origin_x : i32 = i32(global_id.x);
  let origin_y : i32 = i32(global_id.y);

  var accum_x : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var accum_y : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var kernel_index : u32 = 0u;

  for (var ky : i32 = -1; ky <= 1; ky = ky + 1) {
    let sample_y : i32 = clamp_coord(origin_y + ky, height_i);
    for (var kx : i32 = -1; kx <= 1; kx = kx + 1) {
      let sample_x : i32 = clamp_coord(origin_x + kx, width_i);
      let sample : vec4<f32> = textureLoad(input_texture, vec2<i32>(sample_x, sample_y), 0);
      let weight_x : f32 = SOBEL_X[kernel_index];
      let weight_y : f32 = SOBEL_Y[kernel_index];
      accum_x = accum_x + sample * weight_x;
      accum_y = accum_y + sample * weight_y;
      kernel_index = kernel_index + 1u;
    }
  }

  var channel : u32 = 0u;
  loop {
    if (channel >= channel_count) {
      break;
    }
    let dx : f32 = get_component(accum_x, channel);
    let dy : f32 = get_component(accum_y, channel);
    let magnitude : f32 = sqrt(dx * dx + dy * dy);
    output_buffer[base_index + channel] = magnitude;
    channel = channel + 1u;
  }
}
