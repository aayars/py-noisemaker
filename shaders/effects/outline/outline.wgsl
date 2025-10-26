// Outline effect: extract Sobel edges from a luminance map and multiply them with the source image.
// Mirrors noisemaker.effects.outline.

const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;
const CHANNEL_CAP : u32 = 4u;
const EPSILON : f32 = 1e-6;
const F32_MAX : f32 = 0x1.fffffep+127;
const F32_MIN : f32 = -0x1.fffffep+127;

struct OutlineParams {
    size : vec4<f32>,    // (width, height, channels, _pad0)
    options : vec4<f32>, // (sobel_metric, invert, time, speed)
};

@group(0) @binding(0) var inputTexture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : OutlineParams;
@group(0) @binding(3) var<storage, read_write> tempBuffer : array<f32>;

const BLUR_KERNEL : array<f32, 25> = array<f32, 25>(
    1.0 / 36.0,  4.0 / 36.0,  6.0 / 36.0,  4.0 / 36.0, 1.0 / 36.0,
    4.0 / 36.0, 16.0 / 36.0, 24.0 / 36.0, 16.0 / 36.0, 4.0 / 36.0,
    6.0 / 36.0, 24.0 / 36.0, 36.0 / 36.0, 24.0 / 36.0, 6.0 / 36.0,
    4.0 / 36.0, 16.0 / 36.0, 24.0 / 36.0, 16.0 / 36.0, 4.0 / 36.0,
    1.0 / 36.0,  4.0 / 36.0,  6.0 / 36.0,  4.0 / 36.0, 1.0 / 36.0
);

const BLUR_OFFSETS : array<vec2<i32>, 25> = array<vec2<i32>, 25>(
    vec2<i32>(-2, -2), vec2<i32>(-1, -2), vec2<i32>(0, -2), vec2<i32>(1, -2), vec2<i32>(2, -2),
    vec2<i32>(-2, -1), vec2<i32>(-1, -1), vec2<i32>(0, -1), vec2<i32>(1, -1), vec2<i32>(2, -1),
    vec2<i32>(-2,  0), vec2<i32>(-1,  0), vec2<i32>(0,  0), vec2<i32>(1,  0), vec2<i32>(2,  0),
    vec2<i32>(-2,  1), vec2<i32>(-1,  1), vec2<i32>(0,  1), vec2<i32>(1,  1), vec2<i32>(2,  1),
    vec2<i32>(-2,  2), vec2<i32>(-1,  2), vec2<i32>(0,  2), vec2<i32>(1,  2), vec2<i32>(2,  2)
);

const SOBEL_X : array<f32, 9> = array<f32, 9>(
    0.5,  0.0, -0.5,
    1.0,  0.0, -1.0,
    0.5,  0.0, -0.5
);

const SOBEL_Y : array<f32, 9> = array<f32, 9>(
    0.5,  1.0,  0.5,
    0.0,  0.0,  0.0,
   -0.5, -1.0, -0.5
);

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp_channel_count(channel_value : f32) -> u32 {
    let rounded : i32 = i32(round(channel_value));
    if (rounded <= 1) {
        return 1u;
    }
    if (rounded >= i32(CHANNEL_CAP)) {
        return CHANNEL_CAP;
    }
    return u32(rounded);
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn clamp_vec01(value : vec4<f32>) -> vec4<f32> {
    return clamp(value, vec4<f32>(0.0), vec4<f32>(1.0));
}

fn wrap_coord(value : i32, limit : i32) -> i32 {
    if (limit <= 0) {
        return 0;
    }
    var wrapped : i32 = value % limit;
    if (wrapped < 0) {
        wrapped = wrapped + limit;
    }
    return wrapped;
}

fn pixel_base_index(x : u32, y : u32, width : u32) -> u32 {
    return (y * width + x) * CHANNEL_CAP;
}

fn srgb_to_linear(value : f32) -> f32 {
    if (value <= 0.04045) {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn cbrt(value : f32) -> f32 {
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

    let l_c : f32 = cbrt(l);
    let m_c : f32 = cbrt(m);
    let s_c : f32 = cbrt(s);

    return clamp01(0.2104542553 * l_c + 0.7936177850 * m_c - 0.0040720468 * s_c);
}

fn compute_value_map(coords : vec2<i32>, width : i32, height : i32, channel_count : u32) -> f32 {
    let sx : i32 = wrap_coord(coords.x, width);
    let sy : i32 = wrap_coord(coords.y, height);
    let texel : vec4<f32> = textureLoad(inputTexture, vec2<i32>(sx, sy), 0);
    if (channel_count <= 1u) {
        return clamp01(texel.x);
    }
    if (channel_count == 2u) {
        return clamp01(texel.x);
    }
    if (channel_count == 3u) {
        return oklab_l_component(texel.xyz);
    }
    let clamped_rgb : vec3<f32> = clamp(texel.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    return oklab_l_component(clamped_rgb);
}

fn read_temp_value(x : u32, y : u32, width : u32) -> f32 {
    let base_index : u32 = pixel_base_index(x, y, width);
    return tempBuffer[base_index + 0u];
}

fn write_temp_value(x : u32, y : u32, width : u32, value : f32) {
    let base_index : u32 = pixel_base_index(x, y, width);
    tempBuffer[base_index + 0u] = value;
    tempBuffer[base_index + 1u] = 0.0;
    tempBuffer[base_index + 2u] = 0.0;
    tempBuffer[base_index + 3u] = 0.0;
}

fn read_output_value(x : u32, y : u32, width : u32) -> f32 {
    let base_index : u32 = pixel_base_index(x, y, width);
    return outputBuffer[base_index + 0u];
}

fn write_output_value(x : u32, y : u32, width : u32, value : vec4<f32>) {
    let base_index : u32 = pixel_base_index(x, y, width);
    outputBuffer[base_index + 0u] = value.x;
    outputBuffer[base_index + 1u] = value.y;
    outputBuffer[base_index + 2u] = value.z;
    outputBuffer[base_index + 3u] = value.w;
}

fn write_output_scalar(x : u32, y : u32, width : u32, value : f32) {
    let base_index : u32 = pixel_base_index(x, y, width);
    outputBuffer[base_index + 0u] = value;
    outputBuffer[base_index + 1u] = 0.0;
    outputBuffer[base_index + 2u] = 0.0;
    outputBuffer[base_index + 3u] = 0.0;
}

fn distance_metric(dx : f32, dy : f32, metric : i32) -> f32 {
    let ax : f32 = abs(dx);
    let ay : f32 = abs(dy);
    switch metric {
        case 2: {
            return ax + ay;
        }
        case 3: {
            return max(ax, ay);
        }
        case 4: {
            return max((ax + ay) / sqrt(2.0), max(ax, ay));
        }
        case 101: {
            return max(ax - dy * 0.5, dy);
        }
        case 102: {
            let term_a : f32 = max(ax - dy * 0.5, dy);
            let term_b : f32 = max(ax - dy * -0.5, dy * -1.0);
            return max(term_a, term_b);
        }
        case 201: {
            let angle : f32 = atan2(dx, -dy) + PI;
            let r : f32 = TAU / 5.0;
            return cos(floor(0.5 + angle / r) * r - angle) * sqrt(dx * dx + dy * dy);
        }
        default: {
            return sqrt(dx * dx + dy * dy);
        }
    }
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (gid.x != 0u || gid.y != 0u || gid.z != 0u) {
        return;
    }

    let width : u32 = as_u32(params.size.x);
    let height : u32 = as_u32(params.size.y);
    if (width == 0u || height == 0u) {
        return;
    }

    let channel_count : u32 = clamp_channel_count(params.size.z);
    let sobel_metric : i32 = i32(round(params.options.x));
    let invert_flag : bool = params.options.y >= 0.5;
    // Time and speed uniforms are retained to mirror the Python signature.

    let width_i : i32 = i32(width);
    let height_i : i32 = i32(height);

    var value_min : f32 = F32_MAX;
    var value_max : f32 = F32_MIN;

    for (var y : u32 = 0u; y < height; y = y + 1u) {
        for (var x : u32 = 0u; x < width; x = x + 1u) {
            let coords : vec2<i32> = vec2<i32>(i32(x), i32(y));
            let luminance : f32 = compute_value_map(coords, width_i, height_i, channel_count);
            value_min = min(value_min, luminance);
            value_max = max(value_max, luminance);
            write_temp_value(x, y, width, luminance);
        }
    }

    let value_delta : f32 = value_max - value_min;
    let has_value_range : bool = value_delta > EPSILON;
    let inv_value_delta : f32 = select(0.0, 1.0 / value_delta, has_value_range);

    for (var y : u32 = 0u; y < height; y = y + 1u) {
        for (var x : u32 = 0u; x < width; x = x + 1u) {
            let base_index : u32 = pixel_base_index(x, y, width);
            let raw_value : f32 = tempBuffer[base_index + 0u];
            let normalized : f32 = select(raw_value, (raw_value - value_min) * inv_value_delta, has_value_range);
            tempBuffer[base_index + 0u] = normalized;
        }
    }

    var blur_min : f32 = F32_MAX;
    var blur_max : f32 = F32_MIN;

    for (var y : u32 = 0u; y < height; y = y + 1u) {
        for (var x : u32 = 0u; x < width; x = x + 1u) {
            var accum : f32 = 0.0;
            for (var i : u32 = 0u; i < 25u; i = i + 1u) {
                let offset : vec2<i32> = BLUR_OFFSETS[i];
                let sample_x : u32 = u32(wrap_coord(i32(x) + offset.x, width_i));
                let sample_y : u32 = u32(wrap_coord(i32(y) + offset.y, height_i));
                let sample_value : f32 = read_temp_value(sample_x, sample_y, width);
                accum = accum + sample_value * BLUR_KERNEL[i];
            }
            blur_min = min(blur_min, accum);
            blur_max = max(blur_max, accum);
            write_output_scalar(x, y, width, accum);
        }
    }

    let blur_delta : f32 = blur_max - blur_min;
    let has_blur_range : bool = blur_delta > EPSILON;
    let inv_blur_delta : f32 = select(0.0, 1.0 / blur_delta, has_blur_range);

    for (var y : u32 = 0u; y < height; y = y + 1u) {
        for (var x : u32 = 0u; x < width; x = x + 1u) {
            let base_index : u32 = pixel_base_index(x, y, width);
            let raw_blur : f32 = outputBuffer[base_index + 0u];
            let normalized_blur : f32 = select(raw_blur, (raw_blur - blur_min) * inv_blur_delta, has_blur_range);
            outputBuffer[base_index + 0u] = normalized_blur;
        }
    }

    var dist_min : f32 = F32_MAX;
    var dist_max : f32 = F32_MIN;

    for (var y : u32 = 0u; y < height; y = y + 1u) {
        for (var x : u32 = 0u; x < width; x = x + 1u) {
            var grad_x : f32 = 0.0;
            var grad_y : f32 = 0.0;
            var kernel_index : u32 = 0u;
            for (var ky : i32 = -1; ky <= 1; ky = ky + 1) {
                for (var kx : i32 = -1; kx <= 1; kx = kx + 1) {
                    let sample_x : u32 = u32(wrap_coord(i32(x) + kx, width_i));
                    let sample_y : u32 = u32(wrap_coord(i32(y) + ky, height_i));
                    let sample_value : f32 = read_output_value(sample_x, sample_y, width);
                    grad_x = grad_x + sample_value * SOBEL_X[kernel_index];
                    grad_y = grad_y + sample_value * SOBEL_Y[kernel_index];
                    kernel_index = kernel_index + 1u;
                }
            }
            let dist_value : f32 = distance_metric(grad_x, grad_y, sobel_metric);
            dist_min = min(dist_min, dist_value);
            dist_max = max(dist_max, dist_value);
            write_temp_value(x, y, width, dist_value);
        }
    }

    let dist_delta : f32 = dist_max - dist_min;
    let has_dist_range : bool = dist_delta > EPSILON;
    let inv_dist_delta : f32 = select(0.0, 1.0 / dist_delta, has_dist_range);

    for (var y : u32 = 0u; y < height; y = y + 1u) {
        for (var x : u32 = 0u; x < width; x = x + 1u) {
            let base_index : u32 = pixel_base_index(x, y, width);
            let raw_dist : f32 = tempBuffer[base_index + 0u];
            let normalized_dist : f32 = select(raw_dist, (raw_dist - dist_min) * inv_dist_delta, has_dist_range);
            let edge_value : f32 = abs(normalized_dist * 2.0 - 1.0);
            tempBuffer[base_index + 0u] = edge_value;
        }
    }

    for (var y : u32 = 0u; y < height; y = y + 1u) {
        for (var x : u32 = 0u; x < width; x = x + 1u) {
            let offset_x : u32 = u32(wrap_coord(i32(x) - 1, width_i));
            let offset_y : u32 = u32(wrap_coord(i32(y) - 1, height_i));
            let mask_value : f32 = read_temp_value(offset_x, offset_y, width);
            let mask_final : f32 = select(mask_value, 1.0 - mask_value, invert_flag);
            let base_color : vec4<f32> = textureLoad(inputTexture, vec2<i32>(i32(x), i32(y)), 0);
            let scaled_color : vec4<f32> = clamp_vec01(base_color * mask_final);
            write_output_value(x, y, width, scaled_color);
        }
    }
}
