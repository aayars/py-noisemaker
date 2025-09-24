struct ColorMapParams {
  width : f32;
  height : f32;
  channels : f32;
  displacement : f32;
  horizontal : f32;
  clutWidth : f32;
  clutHeight : f32;
  clutChannels : f32;
  stage : f32;
  _pad0 : f32;
  _pad1 : f32;
  _pad2 : f32;
};

@group(0) @binding(0) var inputTexture : texture_2d<f32>;
@group(0) @binding(1) var clutTexture : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(3) var<uniform> params : ColorMapParams;
@group(0) @binding(4) var<storage, read_write> statsBuffer : array<f32>;

fn write_stats(min_val : f32, max_val : f32) {
  if (arrayLength(&statsBuffer) < 2u) {
    return;
  }
  statsBuffer[0] = min_val;
  statsBuffer[1] = max_val;
}

fn as_u32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn as_i32(value : f32) -> i32 {
  return i32(max(value, 0.0));
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

fn cbrt(value : f32) -> f32 {
  if (value < 0.0) {
    return -pow(-value, 1.0 / 3.0);
  }
  return pow(value, 1.0 / 3.0);
}

fn oklab_l(rgb : vec3<f32>) -> f32 {
  let r : f32 = srgb_to_lin(clamp01(rgb.x));
  let g : f32 = srgb_to_lin(clamp01(rgb.y));
  let b : f32 = srgb_to_lin(clamp01(rgb.z));

  let l : f32 = 0.4121656120 * r + 0.5362752080 * g + 0.0514575653 * b;
  let m : f32 = 0.2118591070 * r + 0.6807189584 * g + 0.1074065790 * b;
  let s : f32 = 0.0883097947 * r + 0.2818474174 * g + 0.6302613616 * b;

  let l_ : f32 = cbrt(l);
  let m_ : f32 = cbrt(m);
  let s_ : f32 = cbrt(s);

  let L : f32 = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
  return clamp01(L);
}

fn compute_reference(texel : vec4<f32>, channel_count : u32) -> f32 {
  if (channel_count <= 1u) {
    return texel.x;
  }
  if (channel_count == 2u) {
    return texel.x;
  }
  let rgb : vec3<f32> = vec3<f32>(texel.x, texel.y, texel.z);
  return oklab_l(rgb);
}

fn positive_mod(value : i32, divisor : i32) -> i32 {
  var result : i32 = value % divisor;
  if (result < 0) {
    result = result + divisor;
  }
  return result;
}

fn write_channels(base_index : u32, channel_count : u32, color : vec4<f32>) {
  if (channel_count == 0u) {
    return;
  }
  outputBuffer[base_index] = color.x;
  if (channel_count == 1u) {
    return;
  }
  outputBuffer[base_index + 1u] = color.y;
  if (channel_count == 2u) {
    return;
  }
  outputBuffer[base_index + 2u] = color.z;
  if (channel_count == 3u) {
    return;
  }
  outputBuffer[base_index + 3u] = color.w;
  if (channel_count <= 4u) {
    return;
  }
  var extra : u32 = 4u;
  loop {
    if (extra >= channel_count) {
      break;
    }
    outputBuffer[base_index + extra] = color.w;
    extra = extra + 1u;
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let stage : u32 = as_u32(params.stage + 0.5);

  if (stage == 0u) {
    if (gid.x != 0u || gid.y != 0u || gid.z != 0u) {
      return;
    }

    let width0 : u32 = as_u32(params.width + 0.5);
    let height0 : u32 = as_u32(params.height + 0.5);
    let channel_count0 : u32 = max(as_u32(params.channels + 0.5), 1u);

    if (width0 == 0u || height0 == 0u) {
      write_stats(0.0, 0.0);
      return;
    }

    var min_val : f32 = 3.40282347e38;
    var max_val : f32 = -3.40282347e38;

    for (var y : u32 = 0u; y < height0; y = y + 1u) {
      for (var x : u32 = 0u; x < width0; x = x + 1u) {
        let coord0 : vec2<i32> = vec2<i32>(i32(x), i32(y));
        let texel0 : vec4<f32> = textureLoad(inputTexture, coord0, 0);
        let reference0 : f32 = compute_reference(texel0, channel_count0);
        min_val = min(min_val, reference0);
        max_val = max(max_val, reference0);
      }
    }

    write_stats(min_val, max_val);
    return;
  }

  if (stage != 1u) {
    return;
  }

  let width : u32 = as_u32(params.width + 0.5);
  let height : u32 = as_u32(params.height + 0.5);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channel_count : u32 = max(as_u32(params.channels + 0.5), 1u);
  let displacement : f32 = params.displacement;
  let horizontal : bool = params.horizontal > 0.5;
  let clut_width : i32 = max(as_i32(params.clutWidth + 0.5), 1);
  let clut_height : i32 = max(as_i32(params.clutHeight + 0.5), 1);
  let clut_channels : u32 = max(as_u32(params.clutChannels + 0.5), 1u);

  var min_val_stage1 : f32 = 0.0;
  var max_val_stage1 : f32 = 0.0;
  if (arrayLength(&statsBuffer) >= 2u) {
    min_val_stage1 = statsBuffer[0];
    max_val_stage1 = statsBuffer[1];
  }

  let coord : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let texel : vec4<f32> = textureLoad(inputTexture, coord, 0);
  let reference_raw : f32 = compute_reference(texel, channel_count);
  var normalized : f32 = reference_raw;
  let range : f32 = max_val_stage1 - min_val_stage1;
  if (range != 0.0) {
    normalized = clamp01((reference_raw - min_val_stage1) / range);
  } else {
    normalized = clamp01(normalized);
  }
  let reference : f32 = normalized * displacement;

  let width_i : i32 = max(i32(width), 1);
  let height_i : i32 = max(i32(height), 1);
  let max_x_offset : f32 = f32(max(width_i - 1, 0));
  let max_y_offset : f32 = f32(max(height_i - 1, 0));
  let offset_x : i32 = i32(floor(reference * max_x_offset + 1e-7));
  var offset_y : i32 = 0;
  if (!horizontal) {
    offset_y = i32(floor(reference * max_y_offset + 1e-7));
  }

  let xi : i32 = positive_mod(i32(gid.x) + offset_x, width_i);
  var yi : i32 = i32(gid.y);
  if (!horizontal) {
    yi = positive_mod(i32(gid.y) + offset_y, height_i);
  }

  let mapped_x : i32 = positive_mod((xi * clut_width) / width_i, clut_width);
  let mapped_y : i32 = positive_mod((positive_mod(yi, height_i) * clut_height) / height_i, clut_height);

  let clut_coord : vec2<i32> = vec2<i32>(mapped_x, mapped_y);
  let clut_texel : vec4<f32> = clamp(textureLoad(clutTexture, clut_coord, 0), vec4<f32>(0.0), vec4<f32>(1.0));

  let pixel_index : u32 = gid.y * width + gid.x;
  let base_index : u32 = pixel_index * clut_channels;
  write_channels(base_index, clut_channels, clut_texel);
}
