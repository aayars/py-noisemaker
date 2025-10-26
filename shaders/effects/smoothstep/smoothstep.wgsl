// Smoothstep remaps values between thresholds `a` and `b` using a smooth Hermite curve.
// Mirrors noisemaker.value.smoothstep.

struct SmoothstepParams {
    width : f32,
    height : f32,
    channels : f32,
    a : f32,
    b : f32,
    time : f32,
    speed : f32,
    _pad0 : f32,
};

const CHANNEL_COUNT : u32 = 4u;
const MIN_RANGE_EPSILON : f32 = 1e-6;

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : SmoothstepParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn sanitized_channel_count(channel_value : f32) -> u32 {
    let count : u32 = as_u32(channel_value);
    return clamp(count, 1u, CHANNEL_COUNT);
}

fn smoothstep_value(value : f32, a : f32, b : f32) -> f32 {
    let range : f32 = b - a;
    if (abs(range) < MIN_RANGE_EPSILON) {
        return select(0.0, 1.0, value >= b);
    }

    let normalized : f32 = (value - a) / range;
    let t : f32 = clamp(normalized, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn write_texel(base_index : u32, value : vec4<f32>) {
    output_buffer[base_index + 0u] = value.x;
    output_buffer[base_index + 1u] = value.y;
    output_buffer[base_index + 2u] = value.z;
    output_buffer[base_index + 3u] = value.w;
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
    let channel_count : u32 = sanitized_channel_count(params.channels);

    let smoothed : vec4<f32> = vec4<f32>(
        smoothstep_value(texel.x, params.a, params.b),
        smoothstep_value(texel.y, params.a, params.b),
        smoothstep_value(texel.z, params.a, params.b),
        smoothstep_value(texel.w, params.a, params.b)
    );

    var result : vec4<f32> = texel;
    if (channel_count >= 1u) {
        result.x = smoothed.x;
    }
    if (channel_count >= 2u) {
        result.y = smoothed.y;
    }
    if (channel_count >= 3u) {
        result.z = smoothed.z;
    }
    if (channel_count >= 4u) {
        result.w = smoothed.w;
    }

    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;
    write_texel(base_index, result);
}
