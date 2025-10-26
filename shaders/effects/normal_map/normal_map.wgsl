// Normal map generation. Mirrors noisemaker.effects.normal_map by computing a
// grayscale reference map, Sobel derivatives, and a stylized Z component.

const CHANNEL_COUNT : u32 = 4u;
const CHANNEL_CAP : u32 = 4u;
const F32_MAX : f32 = 0x1.fffffep+127;
const F32_MIN : f32 = -0x1.fffffep+127;

struct NormalMapParams {
    size : vec4<f32>,    // (width, height, channels, unused)
    motion : vec4<f32>,  // (time, speed, unused, unused)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : NormalMapParams;

const SOBEL_OFFSETS : array<vec2<i32>, 9> = array<vec2<i32>, 9>(
    vec2<i32>(-1, -1), vec2<i32>(0, -1), vec2<i32>(1, -1),
    vec2<i32>(-1,  0), vec2<i32>(0,  0), vec2<i32>(1,  0),
    vec2<i32>(-1,  1), vec2<i32>(0,  1), vec2<i32>(1,  1)
);

const SOBEL_X_KERNEL : array<f32, 9> = array<f32, 9>(
    0.5, 0.0, -0.5,
    1.0, 0.0, -1.0,
    0.5, 0.0, -0.5
);

const SOBEL_Y_KERNEL : array<f32, 9> = array<f32, 9>(
    0.5, 1.0, 0.5,
    0.0, 0.0, 0.0,
   -0.5, -1.0, -0.5
);

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn sanitize_channel_count(raw_value : f32) -> u32 {
    let count : u32 = as_u32(raw_value);
    if (count <= 1u) {
        return 1u;
    }
    if (count >= CHANNEL_CAP) {
        return CHANNEL_CAP;
    }
    return count;
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

fn srgb_to_linear(value : f32) -> f32 {
    if (value <= 0.04045) {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn cbrt_safe(value : f32) -> f32 {
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

    let l_c : f32 = cbrt_safe(l);
    let m_c : f32 = cbrt_safe(m);
    let s_c : f32 = cbrt_safe(s);

    return clamp01(0.2104542553 * l_c + 0.7936177850 * m_c - 0.0040720468 * s_c);
}

fn value_map_component(texel : vec4<f32>, channel_count : u32) -> f32 {
    if (channel_count <= 1u) {
        return texel.x;
    }
    if (channel_count == 2u) {
        return texel.x;
    }
    if (channel_count == 3u) {
        return oklab_l_component(texel.xyz);
    }
    let clamped_rgb : vec3<f32> = clamp(texel.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    return oklab_l_component(clamped_rgb);
}

fn normalize_value(value : f32, min_value : f32, delta : f32) -> f32 {
    if (delta == 0.0) {
        return value;
    }
    return (value - min_value) / delta;
}

fn compute_reference_value(coords : vec2<i32>, channel_count : u32) -> f32 {
    let texel : vec4<f32> = textureLoad(input_texture, coords, 0);
    return value_map_component(texel, channel_count);
}

fn compute_normalized_reference(coords : vec2<i32>, channel_count : u32, ref_min : f32, ref_delta : f32) -> f32 {
    let reference_value : f32 = compute_reference_value(coords, channel_count);
    return normalize_value(reference_value, ref_min, ref_delta);
}

fn sobel_response_x(x : u32, y : u32, width_i : i32, height_i : i32, channel_count : u32, ref_min : f32, ref_delta : f32) -> f32 {
    var accum : f32 = 0.0;
    for (var i : u32 = 0u; i < 9u; i = i + 1u) {
        let offset : vec2<i32> = SOBEL_OFFSETS[i];
        let sample_x : i32 = wrap_coord(i32(x) + offset.x, width_i);
        let sample_y : i32 = wrap_coord(i32(y) + offset.y, height_i);
        let coords : vec2<i32> = vec2<i32>(sample_x, sample_y);
        let sample_value : f32 = compute_normalized_reference(coords, channel_count, ref_min, ref_delta);
        accum = accum + sample_value * SOBEL_X_KERNEL[i];
    }
    return accum;
}

fn sobel_response_y(x : u32, y : u32, width_i : i32, height_i : i32, channel_count : u32, ref_min : f32, ref_delta : f32) -> f32 {
    var accum : f32 = 0.0;
    for (var i : u32 = 0u; i < 9u; i = i + 1u) {
        let offset : vec2<i32> = SOBEL_OFFSETS[i];
        let sample_x : i32 = wrap_coord(i32(x) + offset.x, width_i);
        let sample_y : i32 = wrap_coord(i32(y) + offset.y, height_i);
        let coords : vec2<i32> = vec2<i32>(sample_x, sample_y);
        let sample_value : f32 = compute_normalized_reference(coords, channel_count, ref_min, ref_delta);
        accum = accum + sample_value * SOBEL_Y_KERNEL[i];
    }
    return accum;
}

fn final_x_value(
    x : u32,
    y : u32,
    width_i : i32,
    height_i : i32,
    channel_count : u32,
    ref_min : f32,
    ref_delta : f32,
    inverted_min : f32,
    inverted_delta : f32
) -> f32 {
    let sobel_raw : f32 = sobel_response_x(x, y, width_i, height_i, channel_count, ref_min, ref_delta);
    let inverted_raw : f32 = 1.0 - sobel_raw;
    return normalize_value(inverted_raw, inverted_min, inverted_delta);
}

fn final_y_value(
    x : u32,
    y : u32,
    width_i : i32,
    height_i : i32,
    channel_count : u32,
    ref_min : f32,
    ref_delta : f32,
    sobel_min : f32,
    sobel_delta : f32
) -> f32 {
    let sobel_raw : f32 = sobel_response_y(x, y, width_i, height_i, channel_count, ref_min, ref_delta);
    return normalize_value(sobel_raw, sobel_min, sobel_delta);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (gid.x != 0u || gid.y != 0u) {
        return;
    }

    let width : u32 = as_u32(params.size.x);
    let height : u32 = as_u32(params.size.y);
    if (width == 0u || height == 0u) {
        return;
    }

    let channel_count : u32 = sanitize_channel_count(params.size.z);
    let width_i : i32 = i32(width);
    let height_i : i32 = i32(height);

    var reference_min : f32 = F32_MAX;
    var reference_max : f32 = F32_MIN;

    for (var y : u32 = 0u; y < height; y = y + 1u) {
        for (var x : u32 = 0u; x < width; x = x + 1u) {
            let coords : vec2<i32> = vec2<i32>(i32(x), i32(y));
            let value_component : f32 = compute_reference_value(coords, channel_count);
            reference_min = min(reference_min, value_component);
            reference_max = max(reference_max, value_component);
        }
    }

    let reference_delta : f32 = reference_max - reference_min;

    var inverted_min : f32 = F32_MAX;
    var inverted_max : f32 = F32_MIN;

    var sobel_y_min : f32 = F32_MAX;
    var sobel_y_max : f32 = F32_MIN;
    for (var y1 : u32 = 0u; y1 < height; y1 = y1 + 1u) {
        for (var x1 : u32 = 0u; x1 < width; x1 = x1 + 1u) {
            let sobel_x_raw : f32 = sobel_response_x(
                x1,
                y1,
                width_i,
                height_i,
                channel_count,
                reference_min,
                reference_delta
            );
            let inverted_raw : f32 = 1.0 - sobel_x_raw;
            inverted_min = min(inverted_min, inverted_raw);
            inverted_max = max(inverted_max, inverted_raw);

            let sobel_y_raw : f32 = sobel_response_y(
                x1,
                y1,
                width_i,
                height_i,
                channel_count,
                reference_min,
                reference_delta
            );
            sobel_y_min = min(sobel_y_min, sobel_y_raw);
            sobel_y_max = max(sobel_y_max, sobel_y_raw);
        }
    }

    let inverted_delta : f32 = inverted_max - inverted_min;
    let sobel_y_delta : f32 = sobel_y_max - sobel_y_min;

    var magnitude_min : f32 = F32_MAX;
    var magnitude_max : f32 = F32_MIN;

    for (var y5 : u32 = 0u; y5 < height; y5 = y5 + 1u) {
        for (var x5 : u32 = 0u; x5 < width; x5 = x5 + 1u) {
            let x_value : f32 = final_x_value(
                x5,
                y5,
                width_i,
                height_i,
                channel_count,
                reference_min,
                reference_delta,
                inverted_min,
                inverted_delta
            );
            let y_value : f32 = final_y_value(
                x5,
                y5,
                width_i,
                height_i,
                channel_count,
                reference_min,
                reference_delta,
                sobel_y_min,
                sobel_y_delta
            );
            let magnitude : f32 = sqrt(x_value * x_value + y_value * y_value);
            magnitude_min = min(magnitude_min, magnitude);
            magnitude_max = max(magnitude_max, magnitude);
        }
    }

    let magnitude_delta : f32 = magnitude_max - magnitude_min;

    for (var y6 : u32 = 0u; y6 < height; y6 = y6 + 1u) {
        for (var x6 : u32 = 0u; x6 < width; x6 = x6 + 1u) {
            let x_value : f32 = final_x_value(
                x6,
                y6,
                width_i,
                height_i,
                channel_count,
                reference_min,
                reference_delta,
                inverted_min,
                inverted_delta
            );
            let y_value : f32 = final_y_value(
                x6,
                y6,
                width_i,
                height_i,
                channel_count,
                reference_min,
                reference_delta,
                sobel_y_min,
                sobel_y_delta
            );
            let magnitude : f32 = sqrt(x_value * x_value + y_value * y_value);
            let normalized_magnitude : f32 = normalize_value(magnitude, magnitude_min, magnitude_delta);
            let two_z : f32 = normalized_magnitude * 2.0 - 1.0;
            let z_value : f32 = 1.0 - abs(two_z) * 0.5 + 0.5;

            let pixel : u32 = y6 * width + x6;
            let base_index : u32 = pixel * CHANNEL_COUNT;
            let texel : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(x6), i32(y6)), 0);

            output_buffer[base_index + 0u] = x_value;
            output_buffer[base_index + 1u] = y_value;
            output_buffer[base_index + 2u] = z_value;
            output_buffer[base_index + 3u] = texel.w;
        }
    }
}
