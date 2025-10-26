// GPU recreation of Noisemaker's jpeg_decimate effect. Approximates repeated JPEG
// compression in a single compute dispatch by performing a fixed series of randomized
// quantization and block-sampling steps that mutate the working color.

const CHANNEL_COUNT : u32 = 4u;
const INV_U32_MAX : f32 = 1.0 / 4294967296.0;
const QUALITY_MIN : u32 = 5u;
const QUALITY_MAX : u32 = 50u;
const QUALITY_SPAN : f32 = f32(QUALITY_MAX - QUALITY_MIN);
const DENSITY_MIN : u32 = 50u;
const DENSITY_MAX : u32 = 500u;
const DENSITY_SCALE_MIN : f32 = 0.5;
const DENSITY_SCALE_MAX : f32 = 4.0;
const BLOCK_SIZE_MIN : f32 = 1.0;
const BLOCK_SIZE_MAX : f32 = 12.0;
const BLOCK_BLEND_MIN : f32 = 0.25;
const BLOCK_BLEND_MAX : f32 = 0.95;
const NOISE_BASE : f32 = 0.02;
const TIME_SCALE : f32 = 1024.0;
const SPEED_SCALE : f32 = 2048.0;
const KEY_MAX_VALUE : f32 = 16777215.0;
const INTERNAL_ITERATIONS : u32 = 12u;

struct JpegDecimateParams {
    // Width, height, channel count, unused.
    size : vec4<f32>,
    // time, speed, unused, unused.
    time_speed : vec4<f32>,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : JpegDecimateParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn sanitized_time(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0e6);
}

fn sanitized_speed(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0e4);
}

fn hash_u32(value : u32) -> u32 {
    var hashed : u32 = value;
    hashed = hashed ^ 0x9e3779b9u;
    hashed = hashed * 0x85ebca6bu;
    hashed = hashed ^ (hashed >> 13u);
    hashed = hashed * 0xc2b2ae35u;
    hashed = hashed ^ (hashed >> 16u);
    return hashed;
}

fn build_time_key(time_value : f32, speed_value : f32) -> u32 {
    let time_component : u32 = u32(round(clamp(time_value * TIME_SCALE, 0.0, KEY_MAX_VALUE)));
    let speed_component : u32 = u32(round(clamp(speed_value * SPEED_SCALE, 0.0, KEY_MAX_VALUE)));
    let mixed : u32 = (time_component) ^ (speed_component * 0x27d4eb2du);
    return hash_u32(mixed);
}

fn sequence_random(iteration : u32, salt : u32, time_key : u32) -> f32 {
    let combined : u32 = (iteration * 0xcb1ab31fu) ^ (salt * 0x165667b1u) ^ time_key;
    return f32(hash_u32(combined)) * INV_U32_MAX;
}

fn random_inclusive(iteration : u32, salt : u32, time_key : u32, min_value : u32, max_value : u32) -> u32 {
    if (max_value <= min_value) {
        return min_value;
    }
    let range : u32 = max_value - min_value + 1u;
    let rand : f32 = sequence_random(iteration, salt, time_key);
    let scaled : f32 = rand * f32(range);
    let index : u32 = min(u32(scaled), range - 1u);
    return min_value + index;
}

fn jitter_random(coords : vec2<u32>, iteration : u32, salt : u32, time_key : u32) -> f32 {
    let mix0 : u32 = coords.x * 0x8da6b343u;
    let mix1 : u32 = coords.y * 0xd8163841u;
    let mix2 : u32 = iteration * 0xcb1ab31fu;
    let mix3 : u32 = salt * 0x165667b1u;
    return f32(hash_u32(mix0 ^ mix1 ^ mix2 ^ mix3 ^ time_key)) * INV_U32_MAX;
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn rgb_to_ycbcr(rgb : vec3<f32>) -> vec3<f32> {
    let y : f32 = dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
    let cb : f32 = (rgb.z - y) * 0.564 + 0.5;
    let cr : f32 = (rgb.x - y) * 0.713 + 0.5;
    return vec3<f32>(clamp01(y), clamp01(cb), clamp01(cr));
}

fn ycbcr_to_rgb(ycbcr : vec3<f32>) -> vec3<f32> {
    let y : f32 = ycbcr.x;
    let cb : f32 = ycbcr.y - 0.5;
    let cr : f32 = ycbcr.z - 0.5;
    let r : f32 = y + 1.403 * cr;
    let g : f32 = y - 0.344 * cb - 0.714 * cr;
    let b : f32 = y + 1.773 * cb;
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn quantize_channel(value : f32, level_count : f32) -> f32 {
    let safe_levels : f32 = max(level_count, 1.0);
    if (safe_levels <= 1.0) {
        return clamp01(value);
    }
    let steps : f32 = safe_levels - 1.0;
    let scaled : f32 = clamp01(value) * steps;
    return floor(scaled + 0.5) / steps;
}

fn quantize_ycbcr(color : vec3<f32>, quality_norm : f32) -> vec3<f32> {
    let luma_levels : f32 = mix(24.0, 256.0, quality_norm * quality_norm);
    let chroma_levels : f32 = mix(16.0, 196.0, quality_norm);
    var result : vec3<f32> = color;
    result.x = quantize_channel(result.x, luma_levels);
    let chroma_mix : f32 = mix(0.65, 1.0, quality_norm);
    result.y = mix(0.5, quantize_channel(result.y, chroma_levels), chroma_mix);
    result.z = mix(0.5, quantize_channel(result.z, chroma_levels), chroma_mix);
    return result;
}

fn compute_block_size(quality_norm : f32, density : f32, dimension : i32) -> i32 {
    if (dimension <= 1) {
        return 1;
    }
    let safe_density : f32 = max(density, 1.0);
    let density_scale : f32 = clamp(
        f32(DENSITY_MAX) / safe_density,
        DENSITY_SCALE_MIN,
        DENSITY_SCALE_MAX,
    );
    let base_block : f32 = mix(BLOCK_SIZE_MIN, BLOCK_SIZE_MAX, 1.0 - quality_norm);
    let sized : f32 = clamp(base_block * density_scale, BLOCK_SIZE_MIN, f32(dimension));
    let rounded : i32 = i32(round(sized));
    if (rounded < 1) {
        return 1;
    }
    if (rounded > dimension) {
        return dimension;
    }
    return rounded;
}

fn clamp_coord(value : i32, limit : i32) -> i32 {
    if (limit <= 1) {
        return 0;
    }
    var clamped : i32 = value;
    if (clamped < 0) {
        clamped = 0;
    }
    if (clamped >= limit) {
        clamped = limit - 1;
    }
    return clamped;
}

fn write_pixel(base_index : u32, rgba : vec4<f32>) {
    output_buffer[base_index + 0u] = clamp01(rgba.x);
    output_buffer[base_index + 1u] = clamp01(rgba.y);
    output_buffer[base_index + 2u] = clamp01(rgba.z);
    output_buffer[base_index + 3u] = clamp01(rgba.w);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = max(as_u32(params.size.x), 1u);
    let height : u32 = max(as_u32(params.size.y), 1u);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let width_i : i32 = i32(width);
    let height_i : i32 = i32(height);
    let coords_i : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let coords_u : vec2<u32> = gid.xy;

    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;

    let texel : vec4<f32> = textureLoad(input_texture, coords_i, 0);
    var state_rgb : vec3<f32> = texel.xyz;
    let alpha : f32 = texel.w;

    let time_value : f32 = sanitized_time(params.time_speed.x);
    let speed_value : f32 = sanitized_speed(params.time_speed.y);
    let time_key : u32 = build_time_key(time_value, speed_value);

    for (var iteration : u32 = 0u; iteration < INTERNAL_ITERATIONS; iteration = iteration + 1u) {
        let quality_value : u32 = random_inclusive(iteration, 0u, time_key, QUALITY_MIN, QUALITY_MAX);
        let quality : f32 = f32(quality_value);
        let quality_norm : f32 = clamp(
            (quality - f32(QUALITY_MIN)) / max(QUALITY_SPAN, 1.0),
            0.0,
            1.0,
        );

        let density_x_value : u32 = random_inclusive(iteration, 1u, time_key, DENSITY_MIN, DENSITY_MAX);
        let density_y_value : u32 = random_inclusive(iteration, 2u, time_key, DENSITY_MIN, DENSITY_MAX);
        let density_x : f32 = f32(density_x_value);
        let density_y : f32 = f32(density_y_value);

        let block_width : i32 = compute_block_size(quality_norm, density_x, width_i);
        let block_height : i32 = compute_block_size(quality_norm, density_y, height_i);

        let block_coord : vec2<i32> = vec2<i32>(
            coords_i.x / block_width,
            coords_i.y / block_height
        );
        let block_origin : vec2<i32> = vec2<i32>(
            block_coord.x * block_width,
            block_coord.y * block_height
        );
        let block_center : vec2<i32> = vec2<i32>(
            block_origin.x + block_width / 2,
            block_origin.y + block_height / 2
        );

        let jitter_x : i32 = i32(round(
            (jitter_random(coords_u, iteration, 3u, time_key) - 0.5) * f32(block_width)
        ));
        let jitter_y : i32 = i32(round(
            (jitter_random(coords_u, iteration, 4u, time_key) - 0.5) * f32(block_height)
        ));

        let sample_coords : vec2<i32> = vec2<i32>(
            clamp_coord(block_center.x + jitter_x, width_i),
            clamp_coord(block_center.y + jitter_y, height_i)
        );
        let sample_rgb : vec3<f32> = textureLoad(input_texture, sample_coords, 0).xyz;

        var state_ycbcr : vec3<f32> = rgb_to_ycbcr(state_rgb);
        state_ycbcr = quantize_ycbcr(state_ycbcr, quality_norm);

        var sample_ycbcr : vec3<f32> = rgb_to_ycbcr(sample_rgb);
        sample_ycbcr = quantize_ycbcr(sample_ycbcr, quality_norm);

        let current_rgb : vec3<f32> = ycbcr_to_rgb(state_ycbcr);
        let block_rgb : vec3<f32> = ycbcr_to_rgb(sample_ycbcr);

        let block_blend : f32 = mix(BLOCK_BLEND_MIN, BLOCK_BLEND_MAX, 1.0 - quality_norm);
        var mixed_rgb : vec3<f32> = mix(current_rgb, block_rgb, vec3<f32>(block_blend));

        let noise_strength : f32 = (1.0 - quality_norm) * NOISE_BASE;
        let noise_vec : vec3<f32> = vec3<f32>(
            (jitter_random(coords_u, iteration, 5u, time_key) - 0.5) * 2.0 * noise_strength,
            (jitter_random(coords_u, iteration, 6u, time_key) - 0.5) * 2.0 * noise_strength,
            (jitter_random(coords_u, iteration, 7u, time_key) - 0.5) * 2.0 * noise_strength
        );

        state_rgb = clamp(mixed_rgb + noise_vec, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    let final_color : vec4<f32> = vec4<f32>(state_rgb, alpha);
    write_pixel(base_index, final_color);
}
