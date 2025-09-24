struct SmoothstepParams {
  width : f32,
  height : f32,
  channels : f32,
  min_edge : f32,
  inv_range : f32,
  padding0 : f32,
  padding1 : f32,
  padding2 : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : SmoothstepParams;

fn as_u32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn clamp01(value : f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

fn smoothstep_value(value : f32, min_edge : f32, inv_range : f32) -> f32 {
  var t : f32 = (value - min_edge) * inv_range;
  t = clamp01(t);
  return t * t * (3.0 - 2.0 * t);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = as_u32(params.width);
  let height : u32 = as_u32(params.height);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channel_count : u32 = as_u32(params.channels);
  if (channel_count == 0u) {
    return;
  }

  let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let texel : vec4<f32> = textureLoad(input_texture, coords, 0);
  let pixel_index : u32 = gid.y * width + gid.x;
  let base_index : u32 = pixel_index * channel_count;
  let min_edge : f32 = params.min_edge;
  let inv_range : f32 = params.inv_range;

  if (channel_count > 0u) {
    output_buffer[base_index] = smoothstep_value(texel.x, min_edge, inv_range);
  }
  if (channel_count > 1u) {
    output_buffer[base_index + 1u] = smoothstep_value(texel.y, min_edge, inv_range);
  }
  if (channel_count > 2u) {
    output_buffer[base_index + 2u] = smoothstep_value(texel.z, min_edge, inv_range);
  }
  if (channel_count > 3u) {
    output_buffer[base_index + 3u] = smoothstep_value(texel.w, min_edge, inv_range);
  }

  if (channel_count > 4u) {
    var ch : u32 = 4u;
    let fallback : f32 = smoothstep_value(texel.w, min_edge, inv_range);
    loop {
      if (ch >= channel_count) {
        break;
      }
      output_buffer[base_index + ch] = fallback;
      ch = ch + 1u;
    }
  }
}
