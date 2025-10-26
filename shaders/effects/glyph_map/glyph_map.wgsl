// Glyph map effect. Recreates the Python glyph_map effect by mapping
// an input value map to glyph indices and optionally colorizing with the
// source image.

const PI : f32 = 3.141592653589793;

struct GlyphMapParams {
    size : vec4<f32>,      // width, height, channels, glyph width
    grid_layout : vec4<f32>,    // glyph height, glyph count, mask, colorize flag
    curve : vec4<f32>,     // zoom, alpha, spline order, time
    tempo : vec4<f32>,     // speed, _pad0, _pad1, _pad2
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var glyph_texture : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(3) var<uniform> params : GlyphMapParams;

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn round_to_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn sanitized_channel_count(raw : f32) -> u32 {
    let requested : u32 = round_to_u32(raw);
    if (requested < 1u) {
        return 1u;
    }
    if (requested > 4u) {
        return 4u;
    }
    return requested;
}

fn read_channel(texel : vec4<f32>, index : u32) -> f32 {
    switch index {
        case 0u: { return texel.x; }
        case 1u: { return texel.y; }
        case 2u: { return texel.z; }
        default: { return texel.w; }
    }
}

fn srgb_to_linear(value : f32) -> f32 {
    if (value <= 0.04045) {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn oklab_l_component(rgb : vec3<f32>) -> f32 {
    let r_lin : f32 = srgb_to_linear(rgb.x);
    let g_lin : f32 = srgb_to_linear(rgb.y);
    let b_lin : f32 = srgb_to_linear(rgb.z);

    let l_val : f32 = 0.4121656120 * r_lin + 0.5362752080 * g_lin + 0.0514575653 * b_lin;
    let m_val : f32 = 0.2118591070 * r_lin + 0.6807189584 * g_lin + 0.1074065790 * b_lin;
    let s_val : f32 = 0.0883097947 * r_lin + 0.2818474174 * g_lin + 0.6302613616 * b_lin;

    let l_cbrt : f32 = pow(max(l_val, 0.0), 1.0 / 3.0);
    let m_cbrt : f32 = pow(max(m_val, 0.0), 1.0 / 3.0);
    let s_cbrt : f32 = pow(max(s_val, 0.0), 1.0 / 3.0);

    return 0.2104542553 * l_cbrt + 0.7936177850 * m_cbrt - 0.0040720468 * s_cbrt;
}

fn value_map(texel : vec4<f32>, channel_count : u32) -> f32 {
    if (channel_count == 1u) {
        return clamp01(texel.x);
    }
    if (channel_count == 2u) {
        let lum : f32 = clamp01(texel.x);
        let alpha : f32 = clamp01(texel.y);
        return clamp01(lum * alpha);
    }
    let rgb : vec3<f32> = clamp(texel.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    return clamp01(oklab_l_component(rgb));
}

fn lerp_f32(a : f32, b : f32, t : f32) -> f32 {
    return a + (b - a) * t;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = max(round_to_u32(params.size.x), 1u);
    let height : u32 = max(round_to_u32(params.size.y), 1u);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let channel_count : u32 = sanitized_channel_count(params.size.z);
    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * channel_count;
    let src_coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let src_texel : vec4<f32> = textureLoad(input_texture, src_coords, 0);

    let atlas_dims : vec2<u32> = textureDimensions(glyph_texture);
    if (atlas_dims.x == 0u || atlas_dims.y == 0u) {
        for (var channel : u32 = 0u; channel < channel_count; channel = channel + 1u) {
            output_buffer[base_index + channel] = read_channel(src_texel, channel);
        }
        return;
    }

    let glyph_width_raw : u32 = max(round_to_u32(params.size.w), 1u);
    let glyph_height_raw : u32 = max(round_to_u32(params.grid_layout.x), 1u);
    let glyph_width : u32 = min(glyph_width_raw, max(atlas_dims.x, 1u));
    let glyph_height : u32 = min(glyph_height_raw, max(atlas_dims.y, 1u));

    var glyph_count : u32 = round_to_u32(params.grid_layout.y);
    let atlas_capacity : u32 = max(atlas_dims.y / max(glyph_height, 1u), 1u);
    if (glyph_count == 0u) {
        glyph_count = atlas_capacity;
    }
    glyph_count = clamp(glyph_count, 1u, atlas_capacity);

    let mask_value : f32 = params.grid_layout.z;
    let colorize : bool = params.grid_layout.w > 0.5;
    let zoom : f32 = max(params.curve.x, 1.0e-5);
    let alpha_factor : f32 = clamp01(params.curve.y);
    var spline_order : f32 = params.curve.z;
    if (abs(mask_value - 1020.0) < 0.5) {
        spline_order = 2.0;
    }

    let inv_zoom : f32 = 1.0 / zoom;
    let input_width : u32 = max(u32(floor(f32(width) * inv_zoom)), 1u);
    let input_height : u32 = max(u32(floor(f32(height) * inv_zoom)), 1u);

    var grid_width : u32 = input_width / glyph_width;
    if (grid_width == 0u) {
        grid_width = 1u;
    }
    var grid_height : u32 = input_height / glyph_height;
    if (grid_height == 0u) {
        grid_height = 1u;
    }

    let approx_width : u32 = max(glyph_width * grid_width, 1u);
    let approx_height : u32 = max(glyph_height * grid_height, 1u);

    let width_f : f32 = f32(width);
    let height_f : f32 = f32(height);
    let approx_width_f : f32 = f32(approx_width);
    let approx_height_f : f32 = f32(approx_height);

    let approx_x_f : f32 = (f32(gid.x) + 0.5) / max(width_f, 1.0) * approx_width_f;
    let approx_y_f : f32 = (f32(gid.y) + 0.5) / max(height_f, 1.0) * approx_height_f;
    let approx_x : u32 = min(u32(floor(approx_x_f)), approx_width - 1u);
    let approx_y : u32 = min(u32(floor(approx_y_f)), approx_height - 1u);

    let glyph_local_x : u32 = approx_x % glyph_width;
    let glyph_local_y : u32 = approx_y % glyph_height;
    let cell_x : u32 = min(approx_x / glyph_width, max(grid_width, 1u) - 1u);
    let cell_y : u32 = min(approx_y / glyph_height, max(grid_height, 1u) - 1u);

    let grid_width_f : f32 = f32(max(grid_width, 1u));
    let grid_height_f : f32 = f32(max(grid_height, 1u));
    let cell_center_x : f32 = (f32(cell_x) + 0.5) / grid_width_f;
    let cell_center_y : f32 = (f32(cell_y) + 0.5) / grid_height_f;

    let input_width_f : f32 = f32(input_width);
    let input_height_f : f32 = f32(input_height);
    let sample_input_x : f32 =
        clamp(cell_center_x * input_width_f, 0.0, max(input_width_f - 1.0, 0.0));
    let sample_input_y : f32 =
        clamp(cell_center_y * input_height_f, 0.0, max(input_height_f - 1.0, 0.0));

    let sample_source_x : f32 = clamp((sample_input_x + 0.5) * zoom, 0.0, max(width_f - 1.0, 0.0));
    let sample_source_y : f32 = clamp((sample_input_y + 0.5) * zoom, 0.0, max(height_f - 1.0, 0.0));

    let sample_coords : vec2<i32> = vec2<i32>(i32(sample_source_x), i32(sample_source_y));
    let sample_texel : vec4<f32> = textureLoad(input_texture, sample_coords, 0);

    let value_component : f32 = value_map(sample_texel, channel_count);
    var glyph_selector : f32 = clamp01(value_component);
    if (spline_order >= 1.5) {
        glyph_selector = 0.5 - 0.5 * cos(glyph_selector * PI);
    }

    var glyph_index : u32 = u32(floor(glyph_selector * f32(glyph_count)));
    if (glyph_index >= glyph_count) {
        glyph_index = glyph_count - 1u;
    }

    let glyph_sample_x : i32 = i32(min(glyph_local_x, glyph_width - 1u));
    let glyph_sample_y : i32 = i32(
        glyph_index * glyph_height + min(glyph_local_y, glyph_height - 1u)
    );
    let glyph_coords : vec2<i32> = vec2<i32>(glyph_sample_x, glyph_sample_y);
    let glyph_texel : vec4<f32> = textureLoad(glyph_texture, glyph_coords, 0);
    let glyph_value : f32 = clamp01(glyph_texel.x);

    for (var channel : u32 = 0u; channel < channel_count; channel = channel + 1u) {
        if (!colorize) {
            output_buffer[base_index + channel] = glyph_value;
            continue;
        }

        let src_value : f32 = read_channel(src_texel, channel);
        let overlay_value : f32 = glyph_value * read_channel(sample_texel, channel);
        if (alpha_factor >= 0.9995) {
            output_buffer[base_index + channel] = overlay_value;
        } else {
            let blended : f32 = lerp_f32(src_value, overlay_value, alpha_factor);
            output_buffer[base_index + channel] = blended;
        }
    }
}
