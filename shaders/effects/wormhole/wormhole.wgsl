// Wormhole: per-pixel field flow driven by luminance, ported from the
// Noisemaker Python reference implementation. The shader scatters weighted
// samples according to a sinusoidal offset and normalizes the accumulated
// result before blending with the source image.

const TAU : f32 = 6.28318530717958647692;
const STRIDE_SCALE : f32 = 1024.0;
const MAX_FLOAT : f32 = 3.402823466e38;
const CHANNEL_COUNT : u32 = 4u;

struct WormholeParams {
    // flow = (kink, input_stride, alpha, time)
    flow : vec4<f32>,
    // motion = (speed, _pad0, _pad1, _pad2)
    motion : vec4<f32>,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : WormholeParams;

fn luminance(color : vec4<f32>) -> f32 {
    return dot(color.xyz, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn wrap_index(value : f32, limit : u32) -> u32 {
    if (limit == 0u) {
        return 0u;
    }

    let limit_i : i32 = i32(limit);
    var wrapped : i32 = i32(floor(value)) % limit_i;
    if (wrapped < 0) {
        wrapped = wrapped + limit_i;
    }

    return u32(wrapped);
}

fn apply_normalization(value : f32, min_value : f32, inv_range : f32, enabled : bool) -> f32 {
    if (!enabled) {
        return value;
    }

    return clamp01((value - min_value) * inv_range);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (gid.x != 0u || gid.y != 0u || gid.z != 0u) {
        return;
    }

    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    let width : u32 = dims.x;
    let height : u32 = dims.y;
    if (width == 0u || height == 0u) {
        return;
    }

    let buffer_length : u32 = arrayLength(&output_buffer);
    if (buffer_length == 0u) {
        return;
    }

    var clear_index : u32 = 0u;
    loop {
        if (clear_index >= buffer_length) {
            break;
        }

        output_buffer[clear_index] = 0.0;
        clear_index = clear_index + 1u;
    }

    let kink : f32 = params.flow.x;
    let stride_pixels : f32 = params.flow.y * STRIDE_SCALE;

    var y : u32 = 0u;
    loop {
        if (y >= height) {
            break;
        }

        var x : u32 = 0u;
        loop {
            if (x >= width) {
                break;
            }

            let src_texel : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0);
            let lum : f32 = luminance(src_texel);
            let angle : f32 = lum * TAU * kink;
            let offset_x : f32 = (cos(angle) + 1.0) * stride_pixels;
            let offset_y : f32 = (sin(angle) + 1.0) * stride_pixels;

            let dest_x : u32 = wrap_index(f32(x) + offset_x, width);
            let dest_y : u32 = wrap_index(f32(y) + offset_y, height);
            let dest_pixel : u32 = dest_y * width + dest_x;
            let base_index : u32 = dest_pixel * CHANNEL_COUNT;
            if (base_index + 3u >= buffer_length) {
                x = x + 1u;
                continue;
            }

            let weight : f32 = lum * lum;
            let scaled : vec4<f32> = src_texel * vec4<f32>(weight);

            output_buffer[base_index + 0u] = output_buffer[base_index + 0u] + scaled.x;
            output_buffer[base_index + 1u] = output_buffer[base_index + 1u] + scaled.y;
            output_buffer[base_index + 2u] = output_buffer[base_index + 2u] + scaled.z;
            output_buffer[base_index + 3u] = output_buffer[base_index + 3u] + scaled.w;

            x = x + 1u;
        }

        y = y + 1u;
    }

    var min_value : f32 = MAX_FLOAT;
    var max_value : f32 = -MAX_FLOAT;

    var scan_index : u32 = 0u;
    loop {
        if (scan_index >= buffer_length) {
            break;
        }

        let value : f32 = output_buffer[scan_index];
        if (value < min_value) {
            min_value = value;
        }
        if (value > max_value) {
            max_value = value;
        }

        scan_index = scan_index + 1u;
    }

    let can_normalize : bool = max_value > min_value;
    var inv_range : f32 = 0.0;
    if (can_normalize) {
        inv_range = 1.0 / (max_value - min_value);
    }

    let alpha : f32 = clamp01(params.flow.z);
    let inv_alpha : f32 = 1.0 - alpha;
    let alpha_vec3 : vec3<f32> = vec3<f32>(alpha, alpha, alpha);
    let inv_alpha_vec3 : vec3<f32> = vec3<f32>(inv_alpha, inv_alpha, inv_alpha);

    var out_y : u32 = 0u;
    loop {
        if (out_y >= height) {
            break;
        }

        var out_x : u32 = 0u;
        loop {
            if (out_x >= width) {
                break;
            }

            let pixel_index : u32 = out_y * width + out_x;
            let base_index : u32 = pixel_index * CHANNEL_COUNT;
            if (base_index + 3u >= buffer_length) {
                out_x = out_x + 1u;
                continue;
            }

            let original : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(out_x), i32(out_y)), 0);

            let raw_r : f32 = output_buffer[base_index + 0u];
            let raw_g : f32 = output_buffer[base_index + 1u];
            let raw_b : f32 = output_buffer[base_index + 2u];
            let norm_r : f32 = apply_normalization(raw_r, min_value, inv_range, can_normalize);
            let norm_g : f32 = apply_normalization(raw_g, min_value, inv_range, can_normalize);
            let norm_b : f32 = apply_normalization(raw_b, min_value, inv_range, can_normalize);
            let worm_rgb : vec3<f32> = vec3<f32>(
                sqrt(max(norm_r, 0.0)),
                sqrt(max(norm_g, 0.0)),
                sqrt(max(norm_b, 0.0)),
            );

            let blended_rgb : vec3<f32> = (original.xyz * inv_alpha_vec3) + (worm_rgb * alpha_vec3);
            let clamped_rgb : vec3<f32> = vec3<f32>(
                clamp01(blended_rgb.x),
                clamp01(blended_rgb.y),
                clamp01(blended_rgb.z),
            );

            output_buffer[base_index + 0u] = clamped_rgb.x;
            output_buffer[base_index + 1u] = clamped_rgb.y;
            output_buffer[base_index + 2u] = clamped_rgb.z;
            output_buffer[base_index + 3u] = original.w;

            out_x = out_x + 1u;
        }

        out_y = out_y + 1u;
    }
}
