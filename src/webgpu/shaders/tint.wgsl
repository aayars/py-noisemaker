struct TintParams {
  width : f32;
  height : f32;
  channels : f32;
  alpha : f32;
  rand1 : f32;
  rand2 : f32;
  padding0 : f32;
  padding1 : f32;
};

@group(0) @binding(0) var inputTexture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : TintParams;

fn as_u32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn clamp01(value : f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

fn hsv_to_rgb(hsv : vec3<f32>) -> vec3<f32> {
  let h : f32 = hsv.x;
  let s : f32 = hsv.y;
  let v : f32 = hsv.z;

  let dh : f32 = h * 6.0;
  let dr : f32 = clamp01(abs(dh - 3.0) - 1.0);
  let dg : f32 = clamp01(-abs(dh - 2.0) + 2.0);
  let db : f32 = clamp01(-abs(dh - 4.0) + 2.0);

  let one_minus_s : f32 = 1.0 - s;
  let sr : f32 = s * dr;
  let sg : f32 = s * dg;
  let sb : f32 = s * db;

  let r : f32 = (one_minus_s + sr) * v;
  let g : f32 = (one_minus_s + sg) * v;
  let b : f32 = (one_minus_s + sb) * v;

  return vec3<f32>(r, g, b);
}

fn positive_fract(value : f32) -> f32 {
  return value - floor(value);
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

  let base_r : f32 = texel.x;
  let base_g : f32 = texel.y;
  let base_b : f32 = texel.z;
  let base_a : f32 = texel.w;

  let hue : f32 = positive_fract(base_r * 0.333 + params.rand1 + params.rand2);
  let saturation : f32 = base_g;
  let value : f32 = max(max(base_r, base_g), base_b);

  let tinted_rgb : vec3<f32> = hsv_to_rgb(vec3<f32>(hue, saturation, value));
  let alpha : f32 = clamp(params.alpha, 0.0, 1.0);
  let inv_alpha : f32 = 1.0 - alpha;

  outputBuffer[base_index] = base_r * inv_alpha + tinted_rgb.x * alpha;
  outputBuffer[base_index + 1u] = base_g * inv_alpha + tinted_rgb.y * alpha;
  outputBuffer[base_index + 2u] = base_b * inv_alpha + tinted_rgb.z * alpha;

  if (channel_count > 3u) {
    outputBuffer[base_index + 3u] = base_a;
  }

  if (channel_count > 4u) {
    var ch : u32 = 4u;
    loop {
      if (ch >= channel_count) {
        break;
      }
      outputBuffer[base_index + ch] = base_a;
      ch = ch + 1u;
    }
  }
}
