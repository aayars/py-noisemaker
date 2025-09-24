struct PaletteParams {
  width : f32;
  height : f32;
  channels : f32;
  blend : f32;
  pad0 : f32;
  pad1 : f32;
  pad2 : f32;
  pad3 : f32;
  amp : vec4<f32>;
  freq : vec4<f32>;
  offset : vec4<f32>;
  phase : vec4<f32>;
};

@group(0) @binding(0) var inputTexture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : PaletteParams;

const TAU : f32 = 6.283185307179586;

fn as_u32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn clamp01(value : f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

fn srgb_to_lin(value : f32) -> f32 {
  if (value <= 0.04045) {
    return value / 12.92;
  }
  return pow((value + 0.055) / 1.055, 2.4);
}

fn oklab_l_component(rgb : vec3<f32>) -> f32 {
  let r_lin : f32 = srgb_to_lin(rgb.x);
  let g_lin : f32 = srgb_to_lin(rgb.y);
  let b_lin : f32 = srgb_to_lin(rgb.z);

  let l_val : f32 = 0.4121656120 * r_lin + 0.5362752080 * g_lin + 0.0514575653 * b_lin;
  let m_val : f32 = 0.2118591070 * r_lin + 0.6807189584 * g_lin + 0.1074065790 * b_lin;
  let s_val : f32 = 0.0883097947 * r_lin + 0.2818474174 * g_lin + 0.6302613616 * b_lin;

  let l_cbrt : f32 = pow(max(l_val, 0.0), 1.0 / 3.0);
  let m_cbrt : f32 = pow(max(m_val, 0.0), 1.0 / 3.0);
  let s_cbrt : f32 = pow(max(s_val, 0.0), 1.0 / 3.0);

  return 0.2104542553 * l_cbrt + 0.7936177850 * m_cbrt - 0.0040720468 * s_cbrt;
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
  let texel : vec4<f32> = textureLoad(inputTexture, coords, 0);
  let pixel_index : u32 = gid.y * width + gid.x;
  let base_index : u32 = pixel_index * channel_count;

  if (channel_count < 3u) {
    outputBuffer[base_index] = texel.x;
    if (channel_count > 1u) {
      outputBuffer[base_index + 1u] = texel.y;
    }
    return;
  }

  let base_rgb : vec3<f32> = clamp(texel.xyz, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));
  let lightness : f32 = clamp01(oklab_l_component(base_rgb));

  let freq_vec : vec3<f32> = params.freq.xyz;
  let amp_vec : vec3<f32> = params.amp.xyz;
  let offset_vec : vec3<f32> = params.offset.xyz;
  let phase_vec : vec3<f32> = params.phase.xyz;

  let arg0 : f32 = freq_vec.x * lightness * 0.875 + 0.0625 + phase_vec.x;
  let arg1 : f32 = freq_vec.y * lightness * 0.875 + 0.0625 + phase_vec.y;
  let arg2 : f32 = freq_vec.z * lightness * 0.875 + 0.0625 + phase_vec.z;

  let cos0 : f32 = cos(TAU * arg0);
  let cos1 : f32 = cos(TAU * arg1);
  let cos2 : f32 = cos(TAU * arg2);

  let palette_rgb : vec3<f32> = vec3<f32>(
    offset_vec.x + amp_vec.x * cos0,
    offset_vec.y + amp_vec.y * cos1,
    offset_vec.z + amp_vec.z * cos2,
  );

  let blend_amount : f32 = clamp(params.blend, 0.0, 1.0);
  let inv_blend : f32 = 1.0 - blend_amount;

  outputBuffer[base_index] = base_rgb.x * inv_blend + palette_rgb.x * blend_amount;
  outputBuffer[base_index + 1u] = base_rgb.y * inv_blend + palette_rgb.y * blend_amount;
  outputBuffer[base_index + 2u] = base_rgb.z * inv_blend + palette_rgb.z * blend_amount;

  if (channel_count > 3u) {
    let alpha : f32 = texel.w;
    outputBuffer[base_index + 3u] = alpha;
    if (channel_count > 4u) {
      var ch : u32 = 4u;
      loop {
        if (ch >= channel_count) {
          break;
        }
        outputBuffer[base_index + ch] = alpha;
        ch = ch + 1u;
      }
    }
  }
}
