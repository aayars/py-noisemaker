struct OnScreenDisplayParams {
  sizeAlpha : vec4<f32>;   // width, height, channels, alpha
  overlayInfo : vec4<f32>; // offsetX, offsetY, overlayWidth, overlayHeight
  glyphInfo : vec4<f32>;   // glyphWidth, glyphHeight, tileCount, atlasCount
  layoutInfo : vec4<f32>;  // glyphCount, reserved0, reserved1, reserved2
};

struct GlyphAtlasBuffer {
  values : array<f32>,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : OnScreenDisplayParams;
@group(0) @binding(3) var<storage, read> glyph_atlas : GlyphAtlasBuffer;
@group(0) @binding(4) var<storage, read> glyph_indices : array<u32>;

fn clamp01(value : f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

fn as_u32(value : f32) -> u32 {
  if (value <= 0.0) {
    return 0u;
  }
  return u32(floor(value + 0.5));
}

fn to_i32(value : f32) -> i32 {
  return i32(round(value));
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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width_u : u32 = max(as_u32(params.sizeAlpha.x), 1u);
  let height_u : u32 = max(as_u32(params.sizeAlpha.y), 1u);
  if (gid.x >= width_u || gid.y >= height_u) {
    return;
  }

  let channels_u : u32 = max(as_u32(params.sizeAlpha.z), 1u);
  let alpha : f32 = clamp01(params.sizeAlpha.w);

  let offset_x_i : i32 = to_i32(params.overlayInfo.x);
  let offset_y_i : i32 = to_i32(params.overlayInfo.y);
  let overlay_width_i : i32 = i32(as_u32(params.overlayInfo.z));
  let overlay_height_i : i32 = i32(as_u32(params.overlayInfo.w));

  let glyph_width_u : u32 = max(as_u32(params.glyphInfo.x), 1u);
  let glyph_height_u : u32 = max(as_u32(params.glyphInfo.y), 1u);
  let tile_count_u : u32 = max(as_u32(params.glyphInfo.z), 1u);
  let atlas_count_u : u32 = as_u32(params.glyphInfo.w);

  let glyph_count_u : u32 = as_u32(params.layoutInfo.x);

  let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let sample : vec4<f32> = textureLoad(input_texture, coords, 0);
  var overlay_value : f32 = 0.0;

  if (
    overlay_width_i > 0 &&
    overlay_height_i > 0 &&
    glyph_count_u > 0u &&
    atlas_count_u > 0u
  ) {
    if (
      coords.x >= offset_x_i &&
      coords.x < offset_x_i + overlay_width_i &&
      coords.y >= offset_y_i &&
      coords.y < offset_y_i + overlay_height_i
    ) {
      let local_x : i32 = coords.x - offset_x_i;
      let local_y : i32 = coords.y - offset_y_i;
      let tile_count_i : i32 = i32(tile_count_u);
      let glyph_width_i : i32 = i32(glyph_width_u);
      let glyph_height_i : i32 = i32(glyph_height_u);

      let scaled_x : i32 = local_x / max(tile_count_i, 1);
      let scaled_y : i32 = local_y / max(tile_count_i, 1);
      let glyph_local_x : i32 = glyph_width_i > 0 ? scaled_x % glyph_width_i : 0;
      let glyph_local_y : i32 = glyph_height_i > 0 ? scaled_y % glyph_height_i : 0;
      let glyph_column : i32 = if (glyph_width_i > 0) {
        scaled_x / glyph_width_i
      } else {
        0
      };

      let glyph_indices_len : u32 = arrayLength(&glyph_indices);
      if (glyph_indices_len > 0u) {
        let clamped_col : i32 = clamp_i32(glyph_column, 0, i32(glyph_count_u) - 1);
        let col_u : u32 = u32(clamped_col);
        if (col_u < glyph_indices_len) {
          let raw_index : u32 = glyph_indices[col_u];
          let max_atlas_index : u32 = if (atlas_count_u == 0u) { 0u } else { atlas_count_u - 1u };
          let glyph_index_u : u32 = min(raw_index, max_atlas_index);
          let glyph_stride_u : u32 = glyph_width_u * glyph_height_u;
          let atlas_index : u32 =
            glyph_index_u * glyph_stride_u +
            u32(clamp_i32(glyph_local_y, 0, glyph_height_i - 1)) * glyph_width_u +
            u32(clamp_i32(glyph_local_x, 0, glyph_width_i - 1));
          let atlas_len : u32 = arrayLength(&glyph_atlas.values);
          if (atlas_len > 0u && atlas_index < atlas_len) {
            overlay_value = clamp01(glyph_atlas.values[atlas_index]);
          }
        }
      }
    }
  }

  let pixel_index : u32 = gid.y * width_u + gid.x;
  let base_index : u32 = pixel_index * channels_u;

  let overlay_clamped : f32 = clamp01(overlay_value);

  let base_r : f32 = clamp01(sample.x);
  let base_g : f32 = clamp01(sample.y);
  let base_b : f32 = clamp01(sample.z);
  let base_a : f32 = clamp01(sample.w);

  let highlight_r : f32 = max(base_r, overlay_clamped);
  let highlight_g : f32 = max(base_g, overlay_clamped);
  let highlight_b : f32 = max(base_b, overlay_clamped);
  let highlight_a : f32 = max(base_a, overlay_clamped);

  let final_r : f32 = clamp01(mix(base_r, highlight_r, alpha));
  let final_g : f32 = clamp01(mix(base_g, highlight_g, alpha));
  let final_b : f32 = clamp01(mix(base_b, highlight_b, alpha));
  let final_a : f32 = clamp01(mix(base_a, highlight_a, alpha));

  if (channels_u == 1u) {
    output_buffer[base_index] = final_r;
    return;
  }

  if (channels_u == 2u) {
    output_buffer[base_index] = final_r;
    output_buffer[base_index + 1u] = final_g;
    return;
  }

  output_buffer[base_index] = final_r;
  if (channels_u > 1u) {
    output_buffer[base_index + 1u] = final_g;
  }
  if (channels_u > 2u) {
    output_buffer[base_index + 2u] = final_b;
  }
  if (channels_u > 3u) {
    output_buffer[base_index + 3u] = final_a;
  }
  if (channels_u > 4u) {
    var extra : u32 = 4u;
    loop {
      if (extra >= channels_u) {
        break;
      }
      output_buffer[base_index + extra] = final_a;
      extra = extra + 1u;
    }
  }
}
