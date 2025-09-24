struct SobelFinalizeParams {
  width : f32,
  height : f32,
  channels : f32,
  offset_x : f32,
  offset_y : f32,
  _pad0 : f32,
  _pad1 : f32,
  _pad2 : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : SobelFinalizeParams;

fn as_u32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn wrap_coord(value : i32, limit : i32) -> i32 {
  if (limit == 0) {
    return 0;
  }
  var result : i32 = value % limit;
  if (result < 0) {
    result = result + limit;
  }
  return result;
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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let width : u32 = as_u32(params.width);
  let height : u32 = as_u32(params.height);
  if (global_id.x >= width || global_id.y >= height) {
    return;
  }

  let channels : u32 = max(as_u32(params.channels + 0.5), 1u);
  let pixel_index : u32 = global_id.y * width + global_id.x;
  let base_index : u32 = pixel_index * channels;

  let offset_x : i32 = i32(params.offset_x);
  let offset_y : i32 = i32(params.offset_y);
  let sample_x : i32 = wrap_coord(i32(global_id.x) + offset_x, i32(width));
  let sample_y : i32 = wrap_coord(i32(global_id.y) + offset_y, i32(height));
  let sample : vec4<f32> = textureLoad(input_texture, vec2<i32>(sample_x, sample_y), 0);

  var channel : u32 = 0u;
  loop {
    if (channel >= channels) {
      break;
    }
    let value : f32 = get_component(sample, channel);
    let remapped : f32 = abs(value * 2.0 - 1.0);
    output_buffer[base_index + channel] = remapped;
    channel = channel + 1u;
  }
}
