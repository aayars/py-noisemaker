struct NormalMapParams {
  width : f32,
  height : f32,
  channels : f32,
  _pad0 : f32,
  _pad1 : f32,
  _pad2 : f32,
  _pad3 : f32,
  _pad4 : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : NormalMapParams;

fn srgb_to_linear(v : f32) -> f32 {
  if (v <= 0.04045) {
    return v / 12.92;
  }
  return pow((v + 0.055) / 1.055, 2.4);
}

fn cbrt(v : f32) -> f32 {
  if (v == 0.0) {
    return 0.0;
  }
  let sign_v = select(-1.0, 1.0, v >= 0.0);
  return sign_v * pow(abs(v), 1.0 / 3.0);
}

fn oklab_l(rgb : vec3<f32>) -> f32 {
  let r_lin = srgb_to_linear(clamp(rgb.x, 0.0, 1.0));
  let g_lin = srgb_to_linear(clamp(rgb.y, 0.0, 1.0));
  let b_lin = srgb_to_linear(clamp(rgb.z, 0.0, 1.0));
  let l = 0.4121656120 * r_lin + 0.5362752080 * g_lin + 0.0514575653 * b_lin;
  let m = 0.2118591070 * r_lin + 0.6807189584 * g_lin + 0.1074065790 * b_lin;
  let s = 0.0883097947 * r_lin + 0.2818474174 * g_lin + 0.6302613616 * b_lin;
  let l_c = cbrt(l);
  let m_c = cbrt(m);
  let s_c = cbrt(s);
  return clamp(0.2104542553 * l_c + 0.7936177850 * m_c - 0.0040720468 * s_c, 0.0, 1.0);
}

fn as_u32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn as_i32(value : f32) -> i32 {
  return i32(max(value, 0.0));
}

fn wrap_coord(v : i32, limit : i32) -> i32 {
  if (limit == 0) {
    return 0;
  }
  var result : i32 = v % limit;
  if (result < 0) {
    result = result + limit;
  }
  return result;
}

fn compute_luminance(texel : vec4<f32>, channel_count : u32) -> f32 {
  if (channel_count <= 1u) {
    return clamp(texel.x, 0.0, 1.0);
  }
  if (channel_count == 2u) {
    return clamp(texel.x, 0.0, 1.0);
  }
  let rgb = vec3<f32>(texel.x, texel.y, texel.z);
  return oklab_l(rgb);
}

fn sample_luminance(
  x : i32,
  y : i32,
  width : i32,
  height : i32,
  channel_count : u32,
) -> f32 {
  let sx : i32 = wrap_coord(x, width);
  let sy : i32 = wrap_coord(y, height);
  let texel : vec4<f32> = textureLoad(input_texture, vec2<i32>(sx, sy), 0);
  return compute_luminance(texel, channel_count);
}

fn sobel_gradient(
  x : i32,
  y : i32,
  width : i32,
  height : i32,
  channel_count : u32,
) -> vec2<f32> {
  let top_left = sample_luminance(x - 1, y - 1, width, height, channel_count);
  let top = sample_luminance(x, y - 1, width, height, channel_count);
  let top_right = sample_luminance(x + 1, y - 1, width, height, channel_count);
  let left = sample_luminance(x - 1, y, width, height, channel_count);
  let right = sample_luminance(x + 1, y, width, height, channel_count);
  let bottom_left = sample_luminance(x - 1, y + 1, width, height, channel_count);
  let bottom = sample_luminance(x, y + 1, width, height, channel_count);
  let bottom_right = sample_luminance(x + 1, y + 1, width, height, channel_count);

  let gx = top_left + 2.0 * left + bottom_left - top_right - 2.0 * right - bottom_right;
  let gy = top_left + 2.0 * top + top_right - bottom_left - 2.0 * bottom - bottom_right;
  return vec2<f32>(gx, gy);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width_u : u32 = as_u32(params.width + 0.5);
  let height_u : u32 = as_u32(params.height + 0.5);
  if (gid.x >= width_u || gid.y >= height_u) {
    return;
  }

  let channel_count : u32 = max(as_u32(params.channels + 0.5), 1u);
  let width_i : i32 = as_i32(params.width + 0.5);
  let height_i : i32 = as_i32(params.height + 0.5);

  let coords = vec2<i32>(i32(gid.x), i32(gid.y));
  let gradient = sobel_gradient(coords.x, coords.y, width_i, height_i, channel_count);

  let sobel_x_norm = clamp((gradient.x + 4.0) * 0.125, 0.0, 1.0);
  let sobel_y_norm = clamp((gradient.y + 4.0) * 0.125, 0.0, 1.0);

  let x_val = clamp(1.0 - sobel_x_norm, 0.0, 1.0);
  let y_val = sobel_y_norm;

  let magnitude = sqrt(x_val * x_val + y_val * y_val);
  let z_norm = clamp(magnitude * (1.0 / sqrt(2.0)), 0.0, 1.0);
  let two_z = z_norm * 2.0 - 1.0;
  let z_val = 1.0 - abs(two_z) * 0.5 + 0.5;

  let pixel_index : u32 = gid.y * width_u + gid.x;
  let base_index : u32 = pixel_index * 3u;
  output_buffer[base_index] = x_val;
  output_buffer[base_index + 1u] = y_val;
  output_buffer[base_index + 2u] = z_val;
}
