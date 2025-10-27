// Normal map generation. Mirrors noisemaker.effects.normal_map by computing a
// grayscale reference map, Sobel derivatives, and a stylized Z component.

const CHANNEL_COUNT : u32 = 4u;
const CHANNEL_CAP : u32 = 4u;

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

fn compute_reference_value(coords : vec2<i32>, channel_count : u32) -> f32 {
    let texel : vec4<f32> = textureLoad(input_texture, coords, 0);
    return value_map_component(texel, channel_count);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.size.x);
    let height : u32 = as_u32(params.size.y);
    
    // Parallel per-pixel computation: each thread handles one pixel
    let x : u32 = gid.x;
    let y : u32 = gid.y;
    
    if (x >= width || y >= height) {
        return;
    }

    let channel_count : u32 = sanitize_channel_count(params.size.z);
    let width_i : i32 = i32(width);
    let height_i : i32 = i32(height);
    
    // Compute Sobel X response (no normalization needed - matches Python reference)
    var sobel_x : f32 = 0.0;
    for (var i : u32 = 0u; i < 9u; i = i + 1u) {
        let offset : vec2<i32> = SOBEL_OFFSETS[i];
        let sample_x : i32 = wrap_coord(i32(x) + offset.x, width_i);
        let sample_y : i32 = wrap_coord(i32(y) + offset.y, height_i);
        let coords : vec2<i32> = vec2<i32>(sample_x, sample_y);
        let sample_value : f32 = compute_reference_value(coords, channel_count);
        sobel_x = sobel_x + sample_value * SOBEL_X_KERNEL[i];
    }
    
    // Compute Sobel Y response (no normalization needed - matches Python reference)
    var sobel_y : f32 = 0.0;
    for (var i : u32 = 0u; i < 9u; i = i + 1u) {
        let offset : vec2<i32> = SOBEL_OFFSETS[i];
        let sample_x : i32 = wrap_coord(i32(x) + offset.x, width_i);
        let sample_y : i32 = wrap_coord(i32(y) + offset.y, height_i);
        let coords : vec2<i32> = vec2<i32>(sample_x, sample_y);
        let sample_value : f32 = compute_reference_value(coords, channel_count);
        sobel_y = sobel_y + sample_value * SOBEL_Y_KERNEL[i];
    }
    
    // Normalize Sobel outputs to [0, 1] range
    // Sobel kernels can produce values roughly in [-4, 4] for typical gradients
    // We use a scaling factor to map this to a reasonable range
    let sobel_scale : f32 = 0.25;  // Approximates 1/4, mapping [-4,4] to [-1,1]
    
    // Python does: x = normalize(1 - sobel_x), y = normalize(sobel_y)
    // Map sobel responses to [0, 1] range and apply the inversion for x
    let sobel_x_scaled : f32 = sobel_x * sobel_scale + 0.5;  // Map to [0, 1]
    let sobel_y_scaled : f32 = sobel_y * sobel_scale + 0.5;  // Map to [0, 1]
    
    let x_value : f32 = clamp01(1.0 - sobel_x_scaled);
    let y_value : f32 = clamp01(sobel_y_scaled);
    
    // Compute Z component: z = 1 - abs(normalize(sqrt(x^2 + y^2)) * 2 - 1) * 0.5 + 0.5
    let magnitude : f32 = sqrt(x_value * x_value + y_value * y_value);
    let normalized_magnitude : f32 = clamp01(magnitude);
    let two_z : f32 = normalized_magnitude * 2.0 - 1.0;
    let z_value : f32 = 1.0 - abs(two_z) * 0.5 + 0.5;

    let pixel : u32 = y * width + x;
    let base_index : u32 = pixel * CHANNEL_COUNT;
    let texel : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0);

    output_buffer[base_index + 0u] = x_value;
    output_buffer[base_index + 1u] = y_value;
    output_buffer[base_index + 2u] = z_value;
    output_buffer[base_index + 3u] = texel.w;
}
