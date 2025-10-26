// Snow effect: blends animated static noise into the source image.

const CHANNEL_COUNT : u32 = 4u;
const TAU : f32 = 6.283185307179586;
const TIME_SEED_OFFSETS : vec3<f32> = vec3<f32>(97.0, 57.0, 131.0);
const STATIC_SEED : vec3<f32> = vec3<f32>(37.0, 17.0, 53.0);
const LIMITER_SEED : vec3<f32> = vec3<f32>(113.0, 71.0, 193.0);

struct SnowParams {
    width : f32,
    height : f32,
    channels : f32,
    alpha : f32,
    time : f32,
    speed : f32,
    _pad0 : f32,
    _pad1 : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : SnowParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp_01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn write_pixel(base_index : u32, rgb : vec3<f32>, alpha : f32) {
    output_buffer[base_index + 0u] = clamp_01(rgb.x);
    output_buffer[base_index + 1u] = clamp_01(rgb.y);
    output_buffer[base_index + 2u] = clamp_01(rgb.z);
    output_buffer[base_index + 3u] = clamp_01(alpha);
}

fn normalized_sine(value : f32) -> f32 {
    return (sin(value) + 1.0) * 0.5;
}

fn periodic_value(time : f32, value : f32) -> f32 {
    return normalized_sine((time - value) * TAU);
}

fn snow_fract_vec3(value : vec3<f32>) -> vec3<f32> {
    return value - floor(value);
}

fn snow_hash(sample : vec3<f32>) -> f32 {
    let scaled : vec3<f32> = snow_fract_vec3(sample * 0.1031);
    let dot_val : f32 = dot(scaled, scaled.yzx + vec3<f32>(33.33));
    let shifted : vec3<f32> = scaled + dot_val;
    let combined : f32 = (shifted.x + shifted.y) * shifted.z;
    let fractional : f32 = combined - floor(combined);
    return clamp(fractional, 0.0, 1.0);
}

fn snow_noise(coord : vec2<f32>, time : f32, speed : f32, seed : vec3<f32>) -> f32 {
    let angle : f32 = time * TAU;
    let z_base : f32 = cos(angle) * speed;
    let base_sample : vec3<f32> = vec3<f32>(coord.x + seed.x, coord.y + seed.y, z_base + seed.z);
    let base_value : f32 = snow_hash(base_sample);

    if (speed == 0.0 || time == 0.0) {
        return base_value;
    }

    let time_seed : vec3<f32> = seed + TIME_SEED_OFFSETS;
    let time_sample : vec3<f32> = vec3<f32>(
        coord.x + time_seed.x,
        coord.y + time_seed.y,
        1.0 + time_seed.z
    );
    let time_value : f32 = snow_hash(time_sample);
    let scaled_time : f32 = periodic_value(time, time_value) * speed;
    let periodic : f32 = periodic_value(scaled_time, base_value);
    return clamp(periodic, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = max(as_u32(params.width), 1u);
    let height : u32 = max(as_u32(params.height), 1u);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let alpha : f32 = clamp(params.alpha, 0.0, 1.0);
    let base_index : u32 = (gid.y * width + gid.x) * CHANNEL_COUNT;
    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let texel : vec4<f32> = textureLoad(input_texture, coords, 0);

    if (alpha <= 0.0) {
        write_pixel(base_index, texel.xyz, texel.w);
        return;
    }

    let coord : vec2<f32> = vec2<f32>(f32(gid.x), f32(gid.y));
    let time : f32 = params.time;
    let speed : f32 = params.speed * 100.0;

    let static_value : f32 = snow_noise(coord, time, speed, STATIC_SEED);
    let limiter_value : f32 = snow_noise(coord, time, speed, LIMITER_SEED);
    let limiter_sq : f32 = limiter_value * limiter_value;
    let limiter_pow4 : f32 = limiter_sq * limiter_sq;
    let limiter_mask : f32 = clamp(limiter_pow4 * alpha, 0.0, 1.0);

    let static_color : vec3<f32> = vec3<f32>(static_value);
    let mixed_rgb : vec3<f32> = mix(texel.xyz, static_color, vec3<f32>(limiter_mask));

    write_pixel(base_index, mixed_rgb, texel.w);
}
