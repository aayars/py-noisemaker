// Texture shading compute shader.
//
// Applies a scalar light mask to an input image by multiplying each channel
// with a factor derived from a precomputed shade map. The shader mirrors the
// binding model used by other effects: binding(0) reads the source texture,
// binding(1) samples the single-channel shade texture, binding(2) is the
// write-only storage texture for results, and binding(3) carries packed
// dimensions plus scaling factors.
struct TextureParams {
  size_and_channels : vec4<f32>;
  factors : vec4<f32>;
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var shade_texture : texture_2d<f32>;
@group(0) @binding(2) var output_texture : texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> params : TextureParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let width : u32 = u32(round(params.size_and_channels.x));
  let height : u32 = u32(round(params.size_and_channels.y));
  if (global_id.x >= width || global_id.y >= height) {
    return;
  }

  let coord : vec2<i32> = vec2<i32>(i32(global_id.x), i32(global_id.y));
  let base_sample : vec4<f32> = textureLoad(input_texture, coord, 0);
  let shade_sample : vec4<f32> = textureLoad(shade_texture, coord, 0);
  let factor : f32 = params.factors.x + shade_sample.x * params.factors.y;
  let channels : u32 = clamp(u32(round(params.size_and_channels.z)), 0u, 4u);

  var result : vec4<f32> = base_sample;
  if (channels > 0u) {
    result.x = base_sample.x * factor;
  }
  if (channels > 1u) {
    result.y = base_sample.y * factor;
  }
  if (channels > 2u) {
    result.z = base_sample.z * factor;
  }
  if (channels > 3u) {
    result.w = base_sample.w * factor;
  }

  textureStore(output_texture, coord, result);
}
