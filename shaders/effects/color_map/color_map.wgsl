// Applies a lookup-table (CLUT) color map based on the source image luminance.
// Mirrors `effects.color_map` in the Python reference implementation.

struct ColorMapParams {
    size : vec4<f32>,     // (width, height, channels, unused)
    options : vec4<f32>,  // (displacement, horizontal, time, speed)
};

struct StatsBuffer {
    min_value : atomic<u32>,
    max_value : atomic<u32>,
};

const CHANNEL_COUNT : u32 = 4u;
const F32_MAX : f32 = 0x1.fffffep+127;
const F32_MIN : f32 = -0x1.fffffep+127;
const EPSILON : f32 = 1e-6;

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ColorMapParams;
@group(0) @binding(3) var clut_texture : texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> stats_buffer : StatsBuffer;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn sanitized_channel_count(channel_value : f32) -> u32 {
    let rounded : i32 = i32(round(channel_value));
    if (rounded <= 1) {
        return 1u;
    }
    if (rounded >= i32(CHANNEL_COUNT)) {
        return CHANNEL_COUNT;
    }
    return u32(rounded);
}

fn wrap_coord(value : i32, extent : i32) -> i32 {
    if (extent <= 0) {
        return 0;
    }
    var wrapped : i32 = value % extent;
    if (wrapped < 0) {
        wrapped = wrapped + extent;
    }
    return wrapped;
}

fn float_to_ordered(value : f32) -> u32 {
    let bits : u32 = bitcast<u32>(value);
    if ((bits & 0x80000000u) != 0u) {
        return ~bits;
    }
    return bits | 0x80000000u;
}

fn ordered_to_float(value : u32) -> f32 {
    if ((value & 0x80000000u) != 0u) {
        return bitcast<f32>(value & 0x7fffffffu);
    }
    return bitcast<f32>(~value);
}

fn srgb_to_linear(value : f32) -> f32 {
    if (value <= 0.04045) {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn cube_root(value : f32) -> f32 {
    if (value == 0.0) {
        return 0.0;
    }
    let sign_value : f32 = select(-1.0, 1.0, value >= 0.0);
    return sign_value * pow(abs(value), 1.0 / 3.0);
}

fn oklab_l_component(rgb : vec3<f32>) -> f32 {
    let r : f32 = srgb_to_linear(clamp01(rgb.x));
    let g : f32 = srgb_to_linear(clamp01(rgb.y));
    let b : f32 = srgb_to_linear(clamp01(rgb.z));

    let l : f32 = 0.4121656120 * r + 0.5362752080 * g + 0.0514575653 * b;
    let m : f32 = 0.2118591070 * r + 0.6807189584 * g + 0.1074065790 * b;
    let s : f32 = 0.0883097947 * r + 0.2818474174 * g + 0.6302613616 * b;

    let l_c : f32 = cube_root(l);
    let m_c : f32 = cube_root(m);
    let s_c : f32 = cube_root(s);

    return 0.2104542553 * l_c + 0.7936177850 * m_c - 0.0040720468 * s_c;
}

fn value_map_component(texel : vec4<f32>, channel_count : u32) -> f32 {
    if (channel_count <= 2u) {
        return texel.x;
    }
    return oklab_l_component(vec3<f32>(texel.x, texel.y, texel.z));
}

fn write_pixel(base_index : u32, color : vec4<f32>) {
    output_buffer[base_index + 0u] = color.x;
    output_buffer[base_index + 1u] = color.y;
    output_buffer[base_index + 2u] = color.z;
    output_buffer[base_index + 3u] = color.w;
}

@compute @workgroup_size(1, 1, 1)
fn reset_stats_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (gid.x != 0u || gid.y != 0u || gid.z != 0u) {
        return;
    }

    atomicStore(&stats_buffer.min_value, float_to_ordered(F32_MAX));
    atomicStore(&stats_buffer.max_value, float_to_ordered(F32_MIN));
}

@compute @workgroup_size(8, 8, 1)
fn minmax_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.size.x);
    let height : u32 = as_u32(params.size.y);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let channel_count : u32 = sanitized_channel_count(params.size.z);
    let coord : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let texel : vec4<f32> = textureLoad(input_texture, coord, 0);
    let reference_value : f32 = value_map_component(texel, channel_count);
    let encoded : u32 = float_to_ordered(reference_value);

    atomicMin(&stats_buffer.min_value, encoded);
    atomicMax(&stats_buffer.max_value, encoded);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.size.x);
    let height : u32 = as_u32(params.size.y);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;

    let channel_count : u32 = sanitized_channel_count(params.size.z);
    let displacement : f32 = params.options.x;
    let horizontal : bool = params.options.y >= 0.5;

    let min_bits : u32 = atomicLoad(&stats_buffer.min_value);
    let max_bits : u32 = atomicLoad(&stats_buffer.max_value);
    var min_value : f32 = ordered_to_float(min_bits);
    var max_value : f32 = ordered_to_float(max_bits);

    if (min_value > max_value) {
        min_value = 0.0;
        max_value = 0.0;
    }

    let range : f32 = max_value - min_value;
    let has_range : bool = abs(range) > EPSILON;

    let coord : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let texel : vec4<f32> = textureLoad(input_texture, coord, 0);
    let reference_raw : f32 = value_map_component(texel, channel_count);

    var normalized : f32 = reference_raw;
    if (has_range) {
        normalized = (reference_raw - min_value) / range;
    }

    let reference : f32 = normalized * displacement;

    let width_i : i32 = i32(width);
    let height_i : i32 = i32(height);
    let max_x_offset : f32 = f32(max(width_i - 1, 0));
    let max_y_offset : f32 = f32(max(height_i - 1, 0));

    let offset_x : i32 = i32(reference * max_x_offset);
    var offset_y : i32 = 0;
    if (!horizontal) {
        offset_y = i32(reference * max_y_offset);
    }

    let sample_x : i32 = wrap_coord(coord.x + offset_x, width_i);
    var sample_y : i32 = coord.y;
    if (!horizontal) {
        sample_y = wrap_coord(coord.y + offset_y, height_i);
    }

    let clut_sample : vec4<f32> = textureLoad(clut_texture, vec2<i32>(sample_x, sample_y), 0);
    write_pixel(base_index, clut_sample);
}
