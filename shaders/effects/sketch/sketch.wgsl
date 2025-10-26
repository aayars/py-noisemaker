// Sketch effect compute shader.
//
// Mirrors the TensorFlow implementation in noisemaker/effects.py::sketch.
// Builds a grayscale value map, enhances contrast, extracts outlines with
// derivative kernels, applies a center-weighted vignette, generates a
// crosshatch shading pass inspired by worms(... behavior=2), blends the
// passes, and finishes with a subtle animated warp.

const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;
const SQRT_TWO : f32 = 1.4142135623730951;
const CHANNEL_COUNT : u32 = 4u;

struct SketchParams {
    size : vec4<f32>,      // width, height, channels, unused
    controls : vec4<f32>,  // time, speed, unused, unused
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : SketchParams;

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn sanitized_channel_count(raw_channels : f32) -> u32 {
    let rounded : i32 = i32(round(raw_channels));
    if (rounded <= 1) {
        return 1u;
    }
    if (rounded >= 4) {
        return 4u;
    }
    return u32(rounded);
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

fn cube_root(value : f32) -> f32 {
    if (value == 0.0) {
        return 0.0;
    }
    let sign_value : f32 = select(-1.0, 1.0, value >= 0.0);
    return sign_value * pow(abs(value), 1.0 / 3.0);
}

fn oklab_luminance(rgb : vec3<f32>) -> f32 {
    let r_lin : f32 = srgb_to_linear(clamp01(rgb.x));
    let g_lin : f32 = srgb_to_linear(clamp01(rgb.y));
    let b_lin : f32 = srgb_to_linear(clamp01(rgb.z));

    let l : f32 = 0.4121656120 * r_lin + 0.5362752080 * g_lin + 0.0514575653 * b_lin;
    let m : f32 = 0.2118591070 * r_lin + 0.6807189584 * g_lin + 0.1074065790 * b_lin;
    let s : f32 = 0.0883097947 * r_lin + 0.2818474174 * g_lin + 0.6302613616 * b_lin;

    let l_c : f32 = cube_root(l);
    let m_c : f32 = cube_root(m);
    let s_c : f32 = cube_root(s);

    return clamp01(0.2104542553 * l_c + 0.7936177850 * m_c - 0.0040720468 * s_c);
}

fn value_luminance(coord : vec2<i32>, channel_count : u32) -> f32 {
    let texel : vec4<f32> = textureLoad(input_texture, coord, 0);
    if (channel_count <= 2u) {
        return clamp01(texel.x);
    }
    return oklab_luminance(texel.xyz);
}

fn normalize_value(value : f32, min_value : f32, max_value : f32) -> f32 {
    let range : f32 = max(max_value - min_value, 1e-6);
    return clamp01((value - min_value) / range);
}

fn adjust_contrast(value : f32, mean_value : f32, amount : f32) -> f32 {
    return (value - mean_value) * amount + mean_value;
}

fn lerp(a : f32, b : f32, t : f32) -> f32 {
    return a + (b - a) * t;
}

const DERIVATIVE_OFFSETS : array<vec2<i32>, 9> = array<vec2<i32>, 9>(
    vec2<i32>(-1, -1), vec2<i32>(0, -1), vec2<i32>(1, -1),
    vec2<i32>(-1,  0), vec2<i32>(0,  0), vec2<i32>(1,  0),
    vec2<i32>(-1,  1), vec2<i32>(0,  1), vec2<i32>(1,  1)
);

const DERIVATIVE_KERNEL_X : array<f32, 9> = array<f32, 9>(
    0.0, 0.0, 0.0,
    0.0, 1.0, -1.0,
    0.0, 0.0, 0.0
);

const DERIVATIVE_KERNEL_Y : array<f32, 9> = array<f32, 9>(
    0.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, -1.0, 0.0
);

fn contrasted_value(
    coord : vec2<i32>,
    channel_count : u32,
    min_value : f32,
    max_value : f32,
    normalized_mean : f32
) -> f32 {
    let luminance : f32 = value_luminance(coord, channel_count);
    let normalized : f32 = normalize_value(luminance, min_value, max_value);
    let contrasted : f32 = adjust_contrast(normalized, normalized_mean, 2.0);
    return clamp01(contrasted);
}

fn derivative_response(
    coord : vec2<i32>,
    width : i32,
    height : i32,
    channel_count : u32,
    min_value : f32,
    max_value : f32,
    normalized_mean : f32,
    invert_source : bool
) -> f32 {
    var grad_x : f32 = 0.0;
    var grad_y : f32 = 0.0;
    for (var i : u32 = 0u; i < 9u; i = i + 1u) {
        let offset : vec2<i32> = coord + DERIVATIVE_OFFSETS[i];
        let wrapped : vec2<i32> = vec2<i32>(
            wrap_coord(offset.x, width),
            wrap_coord(offset.y, height)
        );
        var value : f32 = contrasted_value(wrapped, channel_count, min_value, max_value, normalized_mean);
        if (invert_source) {
            value = 1.0 - value;
        }
        grad_x = grad_x + value * DERIVATIVE_KERNEL_X[i];
        grad_y = grad_y + value * DERIVATIVE_KERNEL_Y[i];
    }
    return sqrt(max(grad_x * grad_x + grad_y * grad_y, 0.0));
}

fn vignette_weight(coord : vec2<i32>, width : f32, height : f32) -> f32 {
    let uv : vec2<f32> = (vec2<f32>(f32(coord.x), f32(coord.y)) + vec2<f32>(0.5, 0.5))
        / vec2<f32>(width, height);
    let center : vec2<f32> = vec2<f32>(0.5, 0.5);
    let dist : f32 = distance(uv, center);
    let max_dist : f32 = 0.5 * SQRT_TWO;
    let normalized : f32 = clamp(dist / max_dist, 0.0, 1.0);
    return pow(normalized, 2.0);
}

fn hash21(p : vec2<f32>) -> f32 {
    let h : f32 = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn triangle_wave(value : f32) -> f32 {
    let fractional : f32 = fract(value);
    return 1.0 - abs(fractional * 2.0 - 1.0);
}

fn rotate_2d(point : vec2<f32>, angle : f32) -> vec2<f32> {
    let c : f32 = cos(angle);
    let s : f32 = sin(angle);
    return vec2<f32>(
        point.x * c - point.y * s,
        point.x * s + point.y * c
    );
}

fn hatch_pattern(uv : vec2<f32>, angle : f32, density : f32, phase : f32) -> f32 {
    let rotated : vec2<f32> = rotate_2d(uv - vec2<f32>(0.5, 0.5), angle) + vec2<f32>(0.5, 0.5);
    let stripe : f32 = triangle_wave(rotated.x * density + phase);
    return clamp01(stripe);
}

fn crosshatch_value(
    coord : vec2<i32>,
    vignette_value : f32,
    time_value : f32,
    speed_value : f32,
    width : f32,
    height : f32
) -> f32 {
    let uv : vec2<f32> = (vec2<f32>(f32(coord.x), f32(coord.y)) + vec2<f32>(0.5, 0.5))
        / vec2<f32>(width, height);
    let darkness : f32 = clamp01(1.0 - vignette_value);
    let density_base : f32 = lerp(32.0, 220.0, pow(darkness, 0.85));

    let animation : f32 = time_value * (0.5 + speed_value * 0.25);
    let noise_seed : vec2<f32> = uv * vec2<f32>(width * 0.5, height * 0.5);
    let jitter : f32 = hash21(noise_seed + vec2<f32>(animation, animation * 1.37));

    let pattern0 : f32 = hatch_pattern(uv, 0.0, density_base, animation * 0.5 + jitter * 2.0);
    let pattern1 : f32 = hatch_pattern(uv, PI * 0.25, density_base * 0.85, animation * 0.75 + jitter * 1.3);
    let pattern2 : f32 = hatch_pattern(uv, -PI * 0.25, density_base * 0.9, animation * 0.95 + jitter * 3.7);

    let combined : f32 = min(pattern0, min(pattern1, pattern2));
    let texture_noise : f32 = hash21(noise_seed * 1.75 + vec2<f32>(animation * 0.25, animation * 0.62));

    let modulated : f32 = lerp(combined, texture_noise, 0.25);
    let attenuated : f32 = lerp(1.0, modulated, clamp01(pow(darkness, 1.4)));
    return clamp01(1.0 - attenuated);
}

fn write_pixel(base_index : u32, value : vec4<f32>) {
    output_buffer[base_index + 0u] = value.x;
    output_buffer[base_index + 1u] = value.y;
    output_buffer[base_index + 2u] = value.z;
    output_buffer[base_index + 3u] = value.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dimensions : vec2<u32> = textureDimensions(input_texture, 0);
    let width : u32 = dimensions.x;
    let height : u32 = dimensions.y;
    
    // Guard: exit if this thread is outside image bounds
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let channel_count : u32 = sanitized_channel_count(params.size.z);
    let width_i : i32 = i32(width);
    let height_i : i32 = i32(height);
    let width_f : f32 = f32(width);
    let height_f : f32 = f32(height);
    
    let time_value : f32 = params.controls.x;
    let speed_value : f32 = params.controls.y;

    // Use fixed statistical assumptions instead of computing global min/max/mean
    // This trades perfect accuracy for massive parallelization speedup
    let luminance_min : f32 = 0.0;
    let luminance_max : f32 = 1.0;
    let normalized_mean : f32 = 0.5;
    let outline_mean : f32 = 0.5;
    
    // Use fixed normalization ranges (reasonable defaults for sketch effect)
    let outline_range : f32 = 1.0;
    let vignette_range : f32 = 1.0;
    let cross_range : f32 = 1.0;
    let contrasted_outline_min : f32 = 0.0;
    let vignette_min : f32 = 0.0;
    let cross_min : f32 = 0.0;

    // Each thread processes its own pixel
    let coord : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;
    let source_color : vec4<f32> = textureLoad(input_texture, coord, 0);

    let grad_value : f32 = derivative_response(
        coord,
        width_i,
        height_i,
        channel_count,
        luminance_min,
        luminance_max,
        normalized_mean,
        false
    );
    let grad_inverted : f32 = derivative_response(
        coord,
        width_i,
        height_i,
        channel_count,
        luminance_min,
        luminance_max,
        normalized_mean,
        true
    );
    let outline_primary : f32 = 1.0 - grad_value;
    let outline_secondary : f32 = 1.0 - grad_inverted;
    let combined_outline : f32 = min(outline_primary, outline_secondary);
    let contrasted_outline : f32 = adjust_contrast(combined_outline, outline_mean, 0.25);
    let normalized_outline : f32 = clamp01((contrasted_outline - contrasted_outline_min) / outline_range);

    let contrasted : f32 = contrasted_value(
        coord,
        channel_count,
        luminance_min,
        luminance_max,
        normalized_mean
    );
    let vignette_weight_value : f32 = vignette_weight(coord, width_f, height_f);
    let edges : f32 = lerp(contrasted, 1.0, vignette_weight_value);
    let vignette_value : f32 = lerp(contrasted, edges, 0.875);
    let normalized_vignette : f32 = clamp01((vignette_value - vignette_min) / vignette_range);

    let cross_value : f32 = crosshatch_value(
        coord,
        normalized_vignette,
        time_value,
        speed_value,
        width_f,
        height_f
    );
    let normalized_cross : f32 = clamp01((cross_value - cross_min) / cross_range);

    let blended : f32 = lerp(normalized_cross, normalized_outline, 0.75);

    let uv : vec2<f32> = (vec2<f32>(f32(coord.x), f32(coord.y)) + vec2<f32>(0.5, 0.5))
        / vec2<f32>(width_f, height_f);
    let displacement_seed : vec2<f32> = uv * vec2<f32>(width_f, height_f) * 0.125;
    let warp_noise_a : f32 = hash21(displacement_seed + vec2<f32>(time_value * speed_value, 0.37));
    let warp_noise_b : f32 = hash21(displacement_seed * 1.37 + vec2<f32>(0.19, time_value * 0.5));
    let warp_offset : f32 = (warp_noise_a - warp_noise_b) * 0.0025;
    let warped : f32 = clamp01(blended + warp_offset);

    let final_value : f32 = clamp01(warped * warped);
    let output_color : vec4<f32> = vec4<f32>(final_value, final_value, final_value, source_color.w);
    write_pixel(base_index, output_color);
}
