// Generates intermediate noise fields used by the Spatter effect.
// Variant selector (params.size_variant.w) controls which noise recipe to emit:
// 0 = smear base, 1 = primary spatter dots, 2 = secondary dots, 3 = ridge mask.

const CHANNEL_COUNT : u32 = 4u;

struct NoiseParams {
    size_variant : vec4<f32>,  // (width, height, channel_count, variant)
    timing_seed : vec4<f32>,   // (time, speed, base_seed, variant_seed)
};

@group(0) @binding(0) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(1) var<uniform> params : NoiseParams;

fn as_u32(value : f32) -> u32 {
    if (value <= 0.0) {
        return 0u;
    }
    return u32(round(value));
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn hash21(p : vec2<f32>) -> f32 {
    let h : f32 = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn hash31(p : vec3<f32>) -> f32 {
    let h : f32 = dot(p, vec3<f32>(127.1, 311.7, 74.7));
    return fract(sin(h) * 43758.5453123);
}

fn fade(value : f32) -> f32 {
    return value * value * (3.0 - 2.0 * value);
}

fn value_noise(p : vec2<f32>, seed : f32) -> f32 {
    let cell : vec2<f32> = floor(p);
    let frac_part : vec2<f32> = fract(p);
    let tl : f32 = hash31(vec3<f32>(cell, seed));
    let tr : f32 = hash31(vec3<f32>(cell + vec2<f32>(1.0, 0.0), seed));
    let bl : f32 = hash31(vec3<f32>(cell + vec2<f32>(0.0, 1.0), seed));
    let br : f32 = hash31(vec3<f32>(cell + vec2<f32>(1.0, 1.0), seed));
    let smooth_t : vec2<f32> = vec2<f32>(fade(frac_part.x), fade(frac_part.y));
    let top : f32 = mix(tl, tr, smooth_t.x);
    let bottom : f32 = mix(bl, br, smooth_t.x);
    return mix(top, bottom, smooth_t.y);
}

fn freq_for_shape(base_freq : f32, width : f32, height : f32) -> vec2<f32> {
    if (base_freq <= 0.0) {
        return vec2<f32>(1.0, 1.0);
    }
    if (abs(width - height) < 1e-5) {
        return vec2<f32>(base_freq, base_freq);
    }
    if (height < width && height > 0.0) {
        return vec2<f32>(base_freq, base_freq * width / height);
    }
    if (width > 0.0) {
        return vec2<f32>(base_freq * height / width, base_freq);
    }
    return vec2<f32>(base_freq, base_freq);
}

fn periodic_offset(time_value : f32, speed : f32, seed : f32) -> vec2<f32> {
    let angle : f32 = time_value * (0.35 + speed * 0.15) + seed * 1.97;
    let radius : f32 = 0.25 + 0.45 * hash21(vec2<f32>(seed, seed + 19.0));
    return vec2<f32>(cos(angle), sin(angle)) * radius;
}

fn simple_multires_exp(
    uv : vec2<f32>,
    base_freq : vec2<f32>,
    octaves : u32,
    time_value : f32,
    speed_value : f32,
    seed : f32,
) -> f32 {
    var freq : vec2<f32> = base_freq;
    var amplitude : f32 = 0.5;
    var accum : f32 = 0.0;
    var weight : f32 = 0.0;
    for (var octave : u32 = 0u; octave < octaves; octave = octave + 1u) {
        let octave_seed : f32 = seed + f32(octave) * 37.17;
        let offset : vec2<f32> = periodic_offset(time_value + f32(octave) * 0.31, speed_value, octave_seed);
        let sample : f32 = value_noise(uv * freq + offset, octave_seed);
        accum = accum + pow(sample, 4.0) * amplitude;
        weight = weight + amplitude;
        freq = freq * 2.0;
        amplitude = amplitude * 0.5;
    }
    if (weight > 0.0) {
        accum = accum / weight;
    }
    return clamp01(accum);
}

fn ridge(value : f32) -> f32 {
    return 1.0 - abs(value * 2.0 - 1.0);
}

fn random_range_u32(seed : vec2<f32>, min_value : u32, max_value : u32) -> u32 {
    if (max_value <= min_value) {
        return min_value;
    }
    let span : f32 = f32(max_value - min_value + 1u);
    let value : f32 = hash21(seed) * span;
    return min_value + u32(floor(value));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.size_variant.x);
    let height : u32 = as_u32(params.size_variant.y);
    if (width == 0u || height == 0u) {
        return;
    }
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let dims : vec2<f32> = vec2<f32>(max(params.size_variant.x, 1.0), max(params.size_variant.y, 1.0));
    let pixel_size : vec2<f32> = vec2<f32>(1.0 / dims.x, 1.0 / dims.y);
    let uv : vec2<f32> = (vec2<f32>(f32(coords.x), f32(coords.y)) + 0.5) * pixel_size;

    let time_value : f32 = params.timing_seed.x;
    let speed_value : f32 = params.timing_seed.y;
    let base_seed : f32 = params.timing_seed.z;
    let variant_seed : f32 = params.timing_seed.w;
    let variant : u32 = u32(max(round(params.size_variant.w), 0.0));

    var result : f32 = 0.0;
    if (variant == 0u) {
        let freq_choice : u32 = random_range_u32(vec2<f32>(time_value * 0.17 + base_seed + 3.0, base_seed + 29.0), 3u, 6u);
        let freq : vec2<f32> = freq_for_shape(f32(freq_choice), dims.x, dims.y);
        result = simple_multires_exp(uv, freq, 6u, time_value, speed_value, base_seed + 23.0 + variant_seed);
    } else if (variant == 1u) {
        let freq_choice : u32 = random_range_u32(vec2<f32>(time_value * 0.37 + base_seed + 5.0, base_seed + 59.0), 32u, 64u);
        let freq : vec2<f32> = freq_for_shape(f32(freq_choice), dims.x, dims.y);
        result = simple_multires_exp(uv, freq, 4u, time_value, speed_value, base_seed + 43.0 + variant_seed);
    } else if (variant == 2u) {
        let freq_choice : u32 = random_range_u32(vec2<f32>(time_value * 0.41 + base_seed + 13.0, base_seed + 97.0), 150u, 200u);
        let freq : vec2<f32> = freq_for_shape(f32(freq_choice), dims.x, dims.y);
        result = simple_multires_exp(uv, freq, 4u, time_value, speed_value, base_seed + 71.0 + variant_seed);
    } else {
        let freq_choice : u32 = random_range_u32(vec2<f32>(time_value * 0.23 + base_seed + 31.0, base_seed + 149.0), 2u, 3u);
        let freq : vec2<f32> = freq_for_shape(f32(freq_choice), dims.x, dims.y);
        let base_value : f32 = simple_multires_exp(uv, freq, 3u, time_value, speed_value, base_seed + 89.0 + variant_seed);
        result = ridge(base_value);
    }

    let clamped : f32 = clamp01(result);
    let base_index : u32 = (gid.y * width + gid.x) * CHANNEL_COUNT;
    output_buffer[base_index + 0u] = clamped;
    output_buffer[base_index + 1u] = clamped;
    output_buffer[base_index + 2u] = clamped;
    output_buffer[base_index + 3u] = 1.0;
}
