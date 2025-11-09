// Sine: Apply a normalized sine curve to selected channels of the input texture.
// This mirrors noisemaker.effects.sine, with optional RGB mode for multi-channel data.

struct SineParams {
    width : f32,
    height : f32,
    channel_count : f32,
    amount : f32,
    time : f32,
    speed : f32,
    rgb : f32,
    _pad0 : f32,
};

const CHANNEL_COUNT : u32 = 4u;

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : SineParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn sanitized_channel_count(raw : f32) -> u32 {
    let count : u32 = as_u32(raw);
    if (count <= 1u) {
        return 1u;
    }
    if (count >= CHANNEL_COUNT) {
        return CHANNEL_COUNT;
    }
    return count;
}

fn normalized_sine(value : f32) -> f32 {
    return (sin(value) + 1.0) * 0.5;
}

fn normalized_sine_vec3(value : vec3<f32>) -> vec3<f32> {
    return (sin(value) + vec3<f32>(1.0)) * 0.5;
}

fn write_pixel(base_index : u32, color : vec4<f32>) {
    output_buffer[base_index + 0u] = color.x;
    output_buffer[base_index + 1u] = color.y;
    output_buffer[base_index + 2u] = color.z;
    output_buffer[base_index + 3u] = color.w;
}

fn apply_sine(texel : vec4<f32>, amount : f32, channel_count : u32, use_rgb : bool) -> vec4<f32> {
    var result : vec4<f32> = texel;

    if (channel_count <= 2u) {
        result.x = normalized_sine(texel.x * amount);
        return result;
    }

    if (channel_count == 3u) {
        if (use_rgb) {
            let rgb : vec3<f32> = normalized_sine_vec3(texel.xyz * amount);
            result = vec4<f32>(rgb, result.w);
        } else {
            result.z = normalized_sine(texel.z * amount);
        }
        return result;
    }

    if (use_rgb) {
        let rgb : vec3<f32> = normalized_sine_vec3(texel.xyz * amount);
        result = vec4<f32>(rgb, result.w);
    } else {
        result.z = normalized_sine(texel.z * amount);
    }
    return result;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.width);
    let height : u32 = as_u32(params.height);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let texel : vec4<f32> = textureLoad(input_texture, coords, 0);
    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;
    let amount : f32 = params.amount;
    let use_rgb : bool = params.rgb > 0.5;
    let channel_count : u32 = sanitized_channel_count(params.channel_count);

    let result : vec4<f32> = apply_sine(texel, amount, channel_count, use_rgb);
    write_pixel(base_index, result);
}
