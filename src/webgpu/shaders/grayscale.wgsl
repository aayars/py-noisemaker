struct GrayscaleParams {
  width : u32,
  height : u32,
  channels : u32,
  _pad : u32,
};

@group(0) @binding(0) var input_tex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : GrayscaleParams;

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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = params.width;
  let height : u32 = params.height;
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channel_count : u32 = max(params.channels, 1u);
  let coords = vec2<i32>(i32(gid.x), i32(gid.y));
  let texel = textureLoad(input_tex, coords, 0);
  let luminance = compute_luminance(texel, channel_count);

  let index : u32 = gid.y * width + gid.x;
  output_buffer[index] = clamp(luminance, 0.0, 1.0);
}
