// Glowing edges effect. Mirrors noisemaker.effects.glowing_edges with WGSL conventions.

const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;
const CHANNEL_CAP : u32 = 4u;
const F32_MAX : f32 = 0x1.fffffep+127;
const F32_MIN : f32 = -0x1.fffffep+127;
const EPSILON : f32 = 1e-6;

struct GlowingEdgesParams {
    // size packs (width, height, channels, unused).
    size : vec4<f32>,
    // options packs (sobel_metric, alpha, time, speed).
    options : vec4<f32>,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : GlowingEdgesParams;
@group(0) @binding(3) var<storage, read_write> temp_buffer : array<f32>;

const BLUR_KERNEL : array<f32, 25> = array<f32, 25>(
    1.0 / 36.0, 4.0 / 36.0, 6.0 / 36.0, 4.0 / 36.0, 1.0 / 36.0,
    4.0 / 36.0, 16.0 / 36.0, 24.0 / 36.0, 16.0 / 36.0, 4.0 / 36.0,
    6.0 / 36.0, 24.0 / 36.0, 36.0 / 36.0, 24.0 / 36.0, 6.0 / 36.0,
    4.0 / 36.0, 16.0 / 36.0, 24.0 / 36.0, 16.0 / 36.0, 4.0 / 36.0,
    1.0 / 36.0, 4.0 / 36.0, 6.0 / 36.0, 4.0 / 36.0, 1.0 / 36.0
);

const SOBEL_X : array<f32, 9> = array<f32, 9>(
    0.5, 0.0, -0.5,
    1.0, 0.0, -1.0,
    0.5, 0.0, -0.5
);

const SOBEL_Y : array<f32, 9> = array<f32, 9>(
    0.5, 1.0, 0.5,
    0.0, 0.0, 0.0,
   -0.5, -1.0, -0.5
);

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn sanitize_metric_id(value : f32) -> i32 {
    let rounded : i32 = i32(round(value));
    if (rounded < 0) {
        return 0;
    }
    return rounded;
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn clamp_vec01(value : vec4<f32>) -> vec4<f32> {
    return clamp(value, vec4<f32>(0.0), vec4<f32>(1.0));
}

fn clamp_vec_symmetric(value : vec4<f32>) -> vec4<f32> {
    return clamp(value, vec4<f32>(-1.0), vec4<f32>(1.0));
}

fn sanitized_channel_count(channel_value : f32) -> u32 {
    let rounded : i32 = i32(round(channel_value));
    if (rounded <= 1) {
        return 1u;
    }
    if (rounded >= i32(CHANNEL_CAP)) {
        return CHANNEL_CAP;
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

fn cubic_root(value : f32) -> f32 {
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

    let l_c : f32 = cubic_root(l);
    let m_c : f32 = cubic_root(m);
    let s_c : f32 = cubic_root(s);

    return clamp01(0.2104542553 * l_c + 0.7936177850 * m_c - 0.0040720468 * s_c);
}

fn compute_value_map(color : vec4<f32>, channel_count : u32) -> f32 {
    if (channel_count <= 1u) {
        return clamp01(color.x);
    }
    if (channel_count == 2u) {
        return clamp01(color.x);
    }
    if (channel_count == 3u) {
        return oklab_l_component(color.xyz);
    }
    let clamped_rgb : vec3<f32> = clamp(color.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    return oklab_l_component(clamped_rgb);
}

fn pixel_base_index(x : u32, y : u32, width : u32) -> u32 {
    return (y * width + x) * CHANNEL_CAP;
}

fn read_temp_channel(base_index : u32, channel : u32) -> f32 {
    return temp_buffer[base_index + channel];
}

fn write_temp_channel(base_index : u32, channel : u32, value : f32) {
    temp_buffer[base_index + channel] = value;
}

fn read_temp_pixel(base_index : u32) -> vec4<f32> {
    return vec4<f32>(
        temp_buffer[base_index + 0u],
        temp_buffer[base_index + 1u],
        temp_buffer[base_index + 2u],
        temp_buffer[base_index + 3u],
    );
}

fn write_temp_pixel(base_index : u32, value : vec4<f32>) {
    temp_buffer[base_index + 0u] = value.x;
    temp_buffer[base_index + 1u] = value.y;
    temp_buffer[base_index + 2u] = value.z;
    temp_buffer[base_index + 3u] = value.w;
}

fn read_output_pixel(base_index : u32) -> vec4<f32> {
    return vec4<f32>(
        output_buffer[base_index + 0u],
        output_buffer[base_index + 1u],
        output_buffer[base_index + 2u],
        output_buffer[base_index + 3u],
    );
}

fn write_output_pixel(base_index : u32, value : vec4<f32>) {
    output_buffer[base_index + 0u] = value.x;
    output_buffer[base_index + 1u] = value.y;
    output_buffer[base_index + 2u] = value.z;
    output_buffer[base_index + 3u] = value.w;
}

fn posterize_value(value : f32, levels : f32) -> f32 {
    if (levels <= 0.0) {
        return clamp01(value);
    }
    let inv_levels : f32 = 1.0 / levels;
    let half_step : f32 = inv_levels * 0.5;
    let scaled : f32 = value * levels;
    let shifted : f32 = scaled + half_step;
    let quantized : f32 = floor(shifted);
    return clamp01(quantized * inv_levels);
}

fn distance_metric(dx : f32, dy : f32, metric_id : i32) -> f32 {
    let ax : f32 = abs(dx);
    let ay : f32 = abs(dy);
    switch metric_id {
        case 2: {
            return ax + ay;
        }
        case 3: {
            return max(ax, ay);
        }
        case 4: {
            return max((ax + ay) * 0.7071067811865476, max(ax, ay));
        }
        case 101: {
            return max(ax - dy * 0.5, dy);
        }
        case 102: {
            let term0 : f32 = max(ax - dy * 0.5, dy);
            let term1 : f32 = max(ax - dy * -0.5, dy * -1.0);
            return max(term0, term1);
        }
        case 201: {
            let angle : f32 = atan2(dx, -dy) + PI;
            let sides : f32 = 5.0;
            let r : f32 = TAU / sides;
            let rotation : f32 = floor(0.5 + angle / r) * r - angle;
            return cos(rotation) * sqrt(dx * dx + dy * dy);
        }
        default: {
            return sqrt(dx * dx + dy * dy);
        }
    }
}

fn pseudo_random(seed : f32) -> f32 {
    return fract(sin(seed) * 43758.5453);
}

fn posterize_levels(time : f32, speed : f32) -> f32 {
    let noise : f32 = pseudo_random(time * 12.9898 + speed * 78.233);
    if (noise < 1.0 / 3.0) {
        return 3.0;
    }
    if (noise < 2.0 / 3.0) {
        return 4.0;
    }
    return 5.0;
}

fn min_component(value : vec4<f32>) -> f32 {
    return min(min(value.x, value.y), min(value.z, value.w));
}

fn max_component(value : vec4<f32>) -> f32 {
    return max(max(value.x, value.y), max(value.z, value.w));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.size.x);
    let height : u32 = as_u32(params.size.y);
    
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let x : u32 = gid.x;
    let y : u32 = gid.y;
    let width_i : i32 = i32(width);
    let height_i : i32 = i32(height);
    let channel_count : u32 = sanitized_channel_count(params.size.z);
    let sobel_metric : i32 = sanitize_metric_id(params.options.x);
    let alpha : f32 = clamp01(params.options.y);

    // Sobel edge detection
    var gx : f32 = 0.0;
    var gy : f32 = 0.0;
    var kernel_index : u32 = 0u;

    for (var ky : i32 = -1; ky <= 1; ky = ky + 1) {
        for (var kx : i32 = -1; kx <= 1; kx = kx + 1) {
            let sample_x : i32 = wrap_coord(i32(x) + kx, width_i);
            let sample_y : i32 = wrap_coord(i32(y) + ky, height_i);
            let coords : vec2<i32> = vec2<i32>(sample_x, sample_y);
            let sample_color : vec4<f32> = textureLoad(input_texture, coords, 0);
            let sample_value : f32 = compute_value_map(sample_color, channel_count);
            
            gx = gx + sample_value * SOBEL_X[kernel_index];
            gy = gy + sample_value * SOBEL_Y[kernel_index];
            kernel_index = kernel_index + 1u;
        }
    }

    let magnitude : f32 = sqrt(gx * gx + gy * gy);
    let edge_value : f32 = clamp01(magnitude);
    
    // Create glowing edge color (white edges)
    let edge_color : vec4<f32> = vec4<f32>(edge_value, edge_value, edge_value, 1.0);
    
    // Blend with original
    let coords : vec2<i32> = vec2<i32>(i32(x), i32(y));
    let original : vec4<f32> = textureLoad(input_texture, coords, 0);
    
    // Lighten blend mode
    let one : vec4<f32> = vec4<f32>(1.0);
    let lighten_color : vec4<f32> = one - (one - edge_color) * (one - original);
    let mixed_rgb : vec3<f32> = clamp(
        mix(original.xyz, lighten_color.xyz, vec3<f32>(alpha)),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    let final_alpha : f32 = clamp01(original.w);
    let final_color : vec4<f32> = vec4<f32>(mixed_rgb, final_alpha);
    
    // Write output
    let base_index : u32 = pixel_base_index(x, y, width);
    write_output_pixel(base_index, final_color);
}
