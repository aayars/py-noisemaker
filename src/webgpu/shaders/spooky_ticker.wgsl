struct FrameUniforms {
  resolution : vec2<f32>,
  time : f32,
  seed : u32,
  frame_index : u32,
  padding0 : u32,
  padding1 : vec2<f32>,
};

@group(0) @binding(1) var<uniform> frame_uniforms : FrameUniforms;
@group(0) @binding(2) var input_texture : texture_2d<f32>;
@group(0) @binding(3) var output_texture : texture_storage_2d<rgba32float, write>;

const INV_UINT_MAX : f32 = 1.0 / 4294967296.0;

fn wrap_coord(value : i32, size : i32) -> i32 {
  if (size <= 0) {
    return 0;
  }
  var wrapped = value % size;
  if (wrapped < 0) {
    wrapped = wrapped + size;
  }
  return wrapped;
}

fn clamp_i32(value : i32, lo : i32, hi : i32) -> i32 {
  var min_v = lo;
  var max_v = hi;
  if (min_v > max_v) {
    let tmp = min_v;
    min_v = max_v;
    max_v = tmp;
  }
  if (value < min_v) {
    return min_v;
  }
  if (value > max_v) {
    return max_v;
  }
  return value;
}

fn pcg3d(v_in : vec3<u32>) -> vec3<u32> {
  var v = v_in;
  v = v * 1664525u + 1013904223u;
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v = v ^ (v >> vec3<u32>(16u));
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  return v;
}

fn random_u32(base_seed : u32, salt_a : u32, salt_b : u32) -> u32 {
  let mix0 = base_seed ^ (salt_a * 0x9e3779b9u) ^ (salt_b * 0x7f4a7c15u);
  let mix1 = base_seed + salt_a * 0x6c8e9cf5u + salt_b * 0x85ebca6bu;
  let mix2 = (base_seed ^ (salt_a << 16)) + salt_b * 0x165667b1u + 0x9e3779b9u;
  let hashed = vec3<u32>(mix0, mix1, mix2);
  return pcg3d(hashed).x;
}

fn random_float(base_seed : u32, salt_a : u32, salt_b : u32) -> f32 {
  return f32(random_u32(base_seed, salt_a, salt_b)) * INV_UINT_MAX;
}

fn ticker_mask(
  coord_x : i32,
  coord_y : i32,
  width : i32,
  height : i32,
  base_seed : u32,
  time_value : f32,
) -> f32 {
  if (width <= 0 || height <= 0) {
    return 0.0;
  }

  let wrapped_x = wrap_coord(coord_x, width);
  let wrapped_y = wrap_coord(coord_y, height);

  let row_count_raw = random_u32(base_seed, 1u, 0u) % 3u + 1u;
  var bottom_padding : i32 = 2;
  var mask_value : f32 = 0.0;
  var row_index : i32 = 0;

  loop {
    if (row_index >= i32(row_count_raw)) {
      break;
    }

    let available = height - bottom_padding;
    if (available <= 0) {
      break;
    }

    let row_seed = random_u32(base_seed, 11u, u32(row_index));
    let min_height = max(1, height / (4 + row_index));
    let max_height = max(min_height, available);
    let range = max_height - min_height;
    var row_height = min_height;
    if (range > 0) {
      let delta = i32(random_u32(row_seed, 3u, 7u) % u32(range + 1));
      row_height = row_height + delta;
    }
    if (row_height > available) {
      row_height = available;
    }
    if (row_height <= 0) {
      break;
    }

    var start_y = height - bottom_padding - row_height;
    if (start_y < 0) {
      row_height = row_height + start_y;
      start_y = 0;
    }
    if (row_height <= 0) {
      break;
    }

    if (wrapped_y >= start_y && wrapped_y < start_y + row_height) {
      let shift = i32(floor(time_value * f32(width)));
      let shifted_x = wrap_coord(wrapped_x + shift, width);

      let glyph_count_raw = random_u32(row_seed, 5u, 13u) % 10u + 4u;
      var glyph_count = max(i32(glyph_count_raw), 1);
      if (glyph_count > width) {
        glyph_count = width;
      }
      var glyph_width = max(width / glyph_count, 1);
      if (glyph_width < 2) {
        glyph_width = 2;
      }

      var char_index = 0;
      if (glyph_width > 0) {
        char_index = shifted_x / glyph_width;
      }
      let char_seed = random_u32(row_seed, 97u, u32(char_index));
      let pattern_seed = random_u32(base_seed, 211u + u32(row_index), u32(char_index)) ^ char_seed;

      let local_x = glyph_width > 0 ? shifted_x % glyph_width : 0;
      let local_y = wrapped_y - start_y;

      let cell_x = clamp_i32((local_x * 4) / max(glyph_width, 1), 0, 3);
      let cell_y = clamp_i32((local_y * 4) / max(row_height, 1), 0, 3);
      let bit_index = u32(cell_y * 4 + cell_x) & 31u;
      let bit = f32((pattern_seed >> bit_index) & 1u);

      let border = if (local_y == 0 || local_y == row_height - 1) { 1.0 } else { 0.0 };
      let spark = f32((pattern_seed >> 30) & 1u);
      let brightness = f32((char_seed >> 8) & 255u) / 255.0;
      let density = f32(char_seed & 255u) / 255.0;

      let base_value = bit * (0.6 + 0.3 * density);
      let highlight = max(border * 0.75, spark * 0.2);
      let combined = clamp(base_value + highlight + brightness * 0.25, 0.0, 1.2);

      let depth_fade = max(0.0, 1.0 - f32(row_index) * 0.2);
      mask_value = max(mask_value, combined * depth_fade);
    }

    bottom_padding = bottom_padding + row_height + 2;
    if (bottom_padding >= height + row_height) {
      break;
    }

    row_index = row_index + 1;
  }

  return clamp(mask_value, 0.0, 1.0);
}

fn apply_blend(src : f32, diff : f32, mask_val : f32, alpha : f32) -> f32 {
  let shadow_alpha = alpha * (1.0 / 3.0);
  let first = mix(src, diff, shadow_alpha);
  let highlight = max(mask_val, first);
  let final_val = mix(first, highlight, alpha);
  return clamp(final_val, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width_u = u32(max(frame_uniforms.resolution.x, 0.0));
  let height_u = u32(max(frame_uniforms.resolution.y, 0.0));
  if (gid.x >= width_u || gid.y >= height_u) {
    return;
  }

  let width_i = i32(width_u);
  let height_i = i32(height_u);

  if (width_i <= 0 || height_i <= 0) {
    return;
  }

  let coords = vec2<i32>(i32(gid.x), i32(gid.y));
  let src = textureLoad(input_texture, coords, 0);

  let stage_seed = frame_uniforms.seed ^ (frame_uniforms.frame_index * 0x9e3779b9u);
  let alpha = clamp(0.5 + 0.25 * random_float(stage_seed, 401u, 7u), 0.0, 1.0);

  let mask_val = ticker_mask(coords.x, coords.y, width_i, height_i, stage_seed, frame_uniforms.time);
  let offset_mask = ticker_mask(coords.x - 1, coords.y - 1, width_i, height_i, stage_seed, frame_uniforms.time);

  var result = vec4<f32>(0.0);
  result.x = apply_blend(src.x, src.x - offset_mask, mask_val, alpha);
  result.y = apply_blend(src.y, src.y - offset_mask, mask_val, alpha);
  result.z = apply_blend(src.z, src.z - offset_mask, mask_val, alpha);
  result.w = apply_blend(src.w, src.w - offset_mask, mask_val, alpha);

  textureStore(output_texture, coords, result);
}
