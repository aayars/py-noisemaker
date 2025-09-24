struct AberrationParams {
  dims0 : vec4<f32>;
  dims1 : vec4<f32>;
};

@group(0) @binding(0) var inputTexture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read> maskBuffer : array<f32>;
@group(0) @binding(2) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(3) var<uniform> params : AberrationParams;

const PI : f32 = 3.141592653589793;

fn blend_linear(a : f32, b : f32, t : f32) -> f32 {
  return mix(a, b, clamp(t, 0.0, 1.0));
}

fn blend_cosine(a : f32, b : f32, g : f32) -> f32 {
  let clamped : f32 = clamp(g, 0.0, 1.0);
  let weight : f32 = (1.0 - cos(clamped * PI)) * 0.5;
  return mix(a, b, weight);
}

fn clamp_index(value : f32, max_index : f32) -> u32 {
  if (value <= 0.0) {
    return 0u;
  }
  if (value >= max_index) {
    return u32(max_index);
  }
  return u32(floor(value));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = u32(params.dims0.x);
  let height : u32 = u32(params.dims0.y);
  let channels : u32 = u32(params.dims0.z);
  let displacement : f32 = params.dims0.w;
  let width_minus_one : f32 = params.dims1.x;

  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let x : u32 = gid.x;
  let y : u32 = gid.y;
  let index : u32 = y * width + x;
  let x_float : f32 = f32(x);
  var gradient : f32 = 0.0;
  if (width > 1u && width_minus_one > 0.0) {
    gradient = x_float / width_minus_one;
  }
  let mask : f32 = maskBuffer[index];

  var red_offset : f32 = min(width_minus_one, x_float + displacement);
  red_offset = blend_linear(red_offset, x_float, gradient);
  red_offset = blend_cosine(x_float, red_offset, mask);
  let red_x : u32 = clamp_index(red_offset, width_minus_one);

  var blue_offset : f32 = max(0.0, x_float - displacement);
  blue_offset = blend_linear(x_float, blue_offset, gradient);
  blue_offset = blend_cosine(x_float, blue_offset, mask);
  let blue_x : u32 = clamp_index(blue_offset, width_minus_one);

  let green_offset : f32 = blend_cosine(x_float, x_float, mask);
  let green_x : u32 = clamp_index(green_offset, width_minus_one);

  let red_texel : vec4<f32> = textureLoad(inputTexture, vec2<i32>(i32(red_x), i32(y)), 0);
  let green_texel : vec4<f32> = textureLoad(inputTexture, vec2<i32>(i32(green_x), i32(y)), 0);
  let blue_texel : vec4<f32> = textureLoad(inputTexture, vec2<i32>(i32(blue_x), i32(y)), 0);

  let base : u32 = index * channels;
  if (channels > 0u) {
    outputBuffer[base] = red_texel.x;
  }
  if (channels > 1u) {
    outputBuffer[base + 1u] = green_texel.y;
  }
  if (channels > 2u) {
    outputBuffer[base + 2u] = blue_texel.z;
  }
  if (channels > 3u) {
    outputBuffer[base + 3u] = blue_texel.w;
  }
}
