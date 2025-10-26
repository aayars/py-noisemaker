// Posterize effect: reduces tonal resolution while preserving flexible gamma
// control and optional temporal dithering to soften visible banding.

const CHANNEL_COUNT : u32 = 4u;
const MIN_GAMMA : f32 = 0.001;

struct PosterizeParams {
    size : vec4<f32>,    // (width, height, channels, levels)
    adjust : vec4<f32>,  // (gamma, time, reserved0, reserved1)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : PosterizeParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp_01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn srgb_to_linear_component(value : f32) -> f32 {
    if (value <= 0.04045) {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn linear_to_srgb_component(value : f32) -> f32 {
    if (value <= 0.0031308) {
        return value * 12.92;
    }
    return 1.055 * pow(value, 1.0 / 2.4) - 0.055;
}

fn srgb_to_linear_rgb(rgb : vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_to_linear_component(rgb.x),
        srgb_to_linear_component(rgb.y),
        srgb_to_linear_component(rgb.z)
    );
}

fn linear_to_srgb_rgb(rgb : vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        linear_to_srgb_component(rgb.x),
        linear_to_srgb_component(rgb.y),
        linear_to_srgb_component(rgb.z)
    );
}

fn pow_vec3(value : vec3<f32>, exponent : f32) -> vec3<f32> {
    return vec3<f32>(
        pow(value.x, exponent),
        pow(value.y, exponent),
        pow(value.z, exponent)
    );
}

fn round_vec3(value : vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        round(value.x),
        round(value.y),
        round(value.z)
    );
}

fn write_pixel(base_index : u32, color : vec4<f32>) {
    output_buffer[base_index + 0u] = color.x;
    output_buffer[base_index + 1u] = color.y;
    output_buffer[base_index + 2u] = color.z;
    output_buffer[base_index + 3u] = color.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.size.x);
    let height : u32 = as_u32(params.size.y);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let texel : vec4<f32> = textureLoad(input_texture, coords, 0);
    let base_index : u32 = (gid.y * width + gid.x) * CHANNEL_COUNT;

    let levels_raw : f32 = max(params.size.w, 0.0);
    let levels_quantized : f32 = max(round(levels_raw), 1.0);
    if (levels_quantized <= 1.0) {
        write_pixel(base_index, texel);
        return;
    }

    let steps : f32 = max(levels_quantized - 1.0, 1.0);
    let gamma_value : f32 = max(params.adjust.x, MIN_GAMMA);
    let inv_gamma : f32 = 1.0 / gamma_value;
    let time_value : f32 = params.adjust.y;

    let channel_count : i32 = i32(round(params.size.z));
    let convert_to_linear : bool = channel_count >= 3;

    var working_rgb : vec3<f32> = texel.xyz;
    if (convert_to_linear) {
        working_rgb = srgb_to_linear_rgb(working_rgb);
    }
    working_rgb = pow_vec3(clamp(working_rgb, vec3<f32>(0.0), vec3<f32>(1.0)), gamma_value);

    // Posterize: multiply by levels, add 0.5/levels offset, floor, divide by levels
    // This matches the Python reference implementation exactly
    working_rgb = working_rgb * steps;
    working_rgb = working_rgb + vec3<f32>((1.0 / steps) * 0.5);
    working_rgb = floor(working_rgb);
    var quantized_rgb : vec3<f32> = working_rgb / vec3<f32>(steps);
    quantized_rgb = pow_vec3(clamp(quantized_rgb, vec3<f32>(0.0), vec3<f32>(1.0)), inv_gamma);

    if (convert_to_linear) {
        quantized_rgb = linear_to_srgb_rgb(quantized_rgb);
    }

    let result_color : vec4<f32> = vec4<f32>(
        clamp_01(quantized_rgb.x),
        clamp_01(quantized_rgb.y),
        clamp_01(quantized_rgb.z),
        texel.w
    );

    write_pixel(base_index, result_color);
}
