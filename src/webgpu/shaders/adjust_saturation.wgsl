struct AdjustSaturationParams {
  width : f32,
  height : f32,
  channels : f32,
  amount : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : AdjustSaturationParams;

fn clamp01(value : f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

fn as_u32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn rgb_to_hsv(rgb : vec3<f32>) -> vec3<f32> {
  let r = rgb.x;
  let g = rgb.y;
  let b = rgb.z;
  let max_c = max(max(r, g), b);
  let min_c = min(min(r, g), b);
  let delta = max_c - min_c;

  var hue = 0.0;
  if (delta != 0.0) {
    if (max_c == r) {
      var raw = (g - b) / delta;
      raw = raw - floor(raw / 6.0) * 6.0;
      if (raw < 0.0) {
        raw = raw + 6.0;
      }
      hue = raw;
    } else if (max_c == g) {
      hue = (b - r) / delta + 2.0;
    } else {
      hue = (r - g) / delta + 4.0;
    }
  }

  hue = hue / 6.0;
  if (hue < 0.0) {
    hue = hue + 1.0;
  }

  var saturation = 0.0;
  if (max_c != 0.0) {
    saturation = delta / max_c;
  }

  return vec3<f32>(hue, saturation, max_c);
}

fn hsv_to_rgb(hsv : vec3<f32>) -> vec3<f32> {
  let h = hsv.x;
  let s = hsv.y;
  let v = hsv.z;
  let dh = h * 6.0;
  let dr = clamp01(abs(dh - 3.0) - 1.0);
  let dg = clamp01(-abs(dh - 2.0) + 2.0);
  let db = clamp01(-abs(dh - 4.0) + 2.0);
  let one_minus_s = 1.0 - s;
  let sr = s * dr;
  let sg = s * dg;
  let sb = s * db;
  let r = (one_minus_s + sr) * v;
  let g = (one_minus_s + sg) * v;
  let b = (one_minus_s + sb) * v;
  return vec3<f32>(r, g, b);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let width : u32 = as_u32(params.width);
  let height : u32 = as_u32(params.height);
  if (global_id.x >= width || global_id.y >= height) {
    return;
  }

  let channel_count : u32 = max(as_u32(params.channels + 0.5), 1u);
  if (channel_count < 3u) {
    return;
  }

  let coords : vec2<i32> = vec2<i32>(i32(global_id.x), i32(global_id.y));
  let texel : vec4<f32> = textureLoad(input_texture, coords, 0);
  var hsv = rgb_to_hsv(vec3<f32>(texel.x, texel.y, texel.z));
  hsv.y = hsv.y * params.amount;
  let rgb = hsv_to_rgb(hsv);

  let pixel_index : u32 = global_id.y * width + global_id.x;
  let base_index : u32 = pixel_index * channel_count;
  output_buffer[base_index] = rgb.x;
  output_buffer[base_index + 1u] = rgb.y;
  output_buffer[base_index + 2u] = rgb.z;

  if (channel_count > 3u) {
    output_buffer[base_index + 3u] = texel.w;
  }
  if (channel_count > 4u) {
    var ch : u32 = 4u;
    loop {
      if (ch >= channel_count) {
        break;
      }
      output_buffer[base_index + ch] = texel.w;
      ch = ch + 1u;
    }
  }
}
