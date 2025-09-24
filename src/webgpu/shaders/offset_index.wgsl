struct OffsetIndexParams {
  width : f32,
  height : f32,
  x_offset : f32,
  y_offset : f32,
};

@group(0) @binding(0) var y_index_texture : texture_2d<f32>;
@group(0) @binding(1) var x_index_texture : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(3) var<uniform> params : OffsetIndexParams;

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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let width : u32 = as_u32(params.width);
  let height : u32 = as_u32(params.height);
  if (global_id.x >= width || global_id.y >= height) {
    return;
  }

  let coords : vec2<i32> = vec2<i32>(i32(global_id.x), i32(global_id.y));
  let y_texel : vec4<f32> = textureLoad(y_index_texture, coords, 0);
  let x_texel : vec4<f32> = textureLoad(x_index_texture, coords, 0);
  let base_y : i32 = i32(floor(y_texel.x));
  let base_x : i32 = i32(floor(x_texel.x));
  let offset_y : i32 = i32(params.y_offset);
  let offset_x : i32 = i32(params.x_offset);
  let wrapped_y : i32 = wrap_coord(base_y + offset_y, i32(height));
  let wrapped_x : i32 = wrap_coord(base_x + offset_x, i32(width));

  let pixel_index : u32 = global_id.y * width + global_id.x;
  let out_index : u32 = pixel_index * 2u;
  output_buffer[out_index] = f32(wrapped_y);
  output_buffer[out_index + 1u] = f32(wrapped_x);
}
