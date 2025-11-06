// Grime: dusty speckles and grime derived from Noisemaker's Python reference.
// Translated to WGSL with matching parameters (time, speed) and 4-channel output.

const CHANNEL_COUNT : u32 = 4u;

struct GrimeParams {
    size : vec4<f32>,        // width, height, channels, unused
    time_speed : vec4<f32>,  // time, speed, strength, debug_mode
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : GrimeParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn freq_for_shape(freq : f32, width : f32, height : f32) -> vec2<f32> {
    if (width <= 0.0 || height <= 0.0) {
        return vec2<f32>(freq, freq);
    }

    if (abs(width - height) < 0.5) {
        return vec2<f32>(freq, freq);
    }

    if (height < width) {
        let scaled : f32 = freq * width / height;
        return vec2<f32>(freq, scaled);
    }

    let scaled : f32 = freq * height / width;
    return vec2<f32>(scaled, freq);
}

fn hash21(p : vec2<f32>) -> f32 {
    let dot_value : f32 = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(dot_value) * 43758.5453123);
}

fn hash31(p : vec3<f32>) -> f32 {
    let dot_value : f32 = dot(p, vec3<f32>(127.1, 311.7, 74.7));
    return fract(sin(dot_value) * 43758.5453123);
}

fn fade(value : f32) -> f32 {
    return value * value * (3.0 - 2.0 * value);
}

fn value_noise(coord : vec2<f32>, seed : f32) -> f32 {
    let cell : vec2<f32> = floor(coord);
    let frac_part : vec2<f32> = fract(coord);

    let top_left : f32 = hash31(vec3<f32>(cell, seed));
    let top_right : f32 = hash31(vec3<f32>(cell + vec2<f32>(1.0, 0.0), seed));
    let bottom_left : f32 = hash31(vec3<f32>(cell + vec2<f32>(0.0, 1.0), seed));
    let bottom_right : f32 = hash31(vec3<f32>(cell + vec2<f32>(1.0, 1.0), seed));

    let smooth_t : vec2<f32> = vec2<f32>(fade(frac_part.x), fade(frac_part.y));
    let top : f32 = mix(top_left, top_right, smooth_t.x);
    let bottom : f32 = mix(bottom_left, bottom_right, smooth_t.x);
    return mix(top, bottom, smooth_t.y);
}

fn periodic_offset(time_value : f32, speed_value : f32, seed : f32) -> vec2<f32> {
    let angle : f32 = time_value * 0.5 + seed * 0.1375;
    let radius : f32 = (0.35 + speed_value * 0.15) * (0.25 + 0.75 * sin(seed * 1.37));
    return vec2<f32>(cos(angle), sin(angle)) * radius;
}

fn simple_multires(
    uv : vec2<f32>,
    base_freq : vec2<f32>,
    time_value : f32,
    speed_value : f32,
    octaves : u32,
    seed : f32,
) -> f32 {
    var freq : vec2<f32> = base_freq;
    var amplitude : f32 = 0.5;
    var total_weight : f32 = 0.0;
    var accum : f32 = 0.0;
    var octave : u32 = 0u;

    loop {
        if (octave >= octaves) {
            break;
        }

        let octave_seed : f32 = seed + f32(octave) * 37.11;
        let offset : vec2<f32> = periodic_offset(
            time_value + f32(octave) * 0.17,
            speed_value,
            octave_seed,
        );
        let sample_coord : vec2<f32> = uv * freq + offset;
        let sample_value : f32 = value_noise(sample_coord, octave_seed);

        accum = accum + sample_value * amplitude;
        total_weight = total_weight + amplitude;
        freq = freq * 2.0;
        amplitude = amplitude * 0.5;
        octave = octave + 1u;
    }

    if (total_weight > 0.0) {
        accum = accum / total_weight;
    }

    return clamp01(accum);
}

fn refracted_scalar_field(
    uv : vec2<f32>,
    base_freq : vec2<f32>,
    time_value : f32,
    speed_value : f32,
    pixel_size : vec2<f32>,
    displacement : f32,
    seed : f32,
) -> f32 {
    let base_mask : f32 = simple_multires(uv, base_freq, time_value, speed_value, 8u, seed);
    let offset_uv : vec2<f32> = fract(uv + vec2<f32>(0.5, 0.5));
    let offset_mask : f32 = simple_multires(
        offset_uv,
        base_freq,
        time_value,
        speed_value,
        8u,
        seed + 19.0,
    );

    let offset_vec : vec2<f32> = vec2<f32>(
        (base_mask * 2.0 - 1.0) * displacement * pixel_size.x,
        (offset_mask * 2.0 - 1.0) * displacement * pixel_size.y,
    );
    let warped_uv : vec2<f32> = fract(uv + offset_vec);
    return simple_multires(warped_uv, base_freq, time_value, speed_value, 8u, seed + 41.0);
}

fn chebyshev_gradient(
    uv : vec2<f32>,
    base_freq : vec2<f32>,
    time_value : f32,
    speed_value : f32,
    pixel_size : vec2<f32>,
    displacement : f32,
    seed : f32,
) -> f32 {
    let offset_x : vec2<f32> = vec2<f32>(pixel_size.x, 0.0);
    let offset_y : vec2<f32> = vec2<f32>(0.0, pixel_size.y);

    let right : f32 = refracted_scalar_field(
        fract(uv + offset_x),
        base_freq,
        time_value,
        speed_value,
        pixel_size,
        displacement,
        seed,
    );
    let left : f32 = refracted_scalar_field(
        fract(uv - offset_x),
        base_freq,
        time_value,
        speed_value,
        pixel_size,
        displacement,
        seed,
    );
    let up : f32 = refracted_scalar_field(
        fract(uv + offset_y),
        base_freq,
        time_value,
        speed_value,
        pixel_size,
        displacement,
        seed,
    );
    let down : f32 = refracted_scalar_field(
        fract(uv - offset_y),
        base_freq,
        time_value,
        speed_value,
        pixel_size,
        displacement,
        seed,
    );

    let dx : f32 = (right - left) * 0.5;
    let dy : f32 = (up - down) * 0.5;
    let gradient : f32 = max(abs(dx), abs(dy));
    return clamp01(gradient * 4.0);
}

fn dropout_mask(uv : vec2<f32>, dims : vec2<f32>, seed : f32) -> f32 {
    let rnd : f32 = hash21(uv * dims + vec2<f32>(seed, seed * 1.37));
    return select(0.0, 1.0, rnd < 0.25);
}

fn exponential_noise(
    uv : vec2<f32>,
    freq : vec2<f32>,
    time_value : f32,
    speed_value : f32,
    seed : f32,
) -> f32 {
    let offset : vec2<f32> = periodic_offset(time_value + seed * 0.07, speed_value, seed + 7.0);
    let noise_value : f32 = value_noise(uv * freq + offset, seed + 13.0);
    return pow(clamp01(noise_value), 4.0);
}

fn refracted_exponential(
    uv : vec2<f32>,
    freq : vec2<f32>,
    time_value : f32,
    speed_value : f32,
    pixel_size : vec2<f32>,
    displacement : f32,
    seed : f32,
) -> f32 {
    let base : f32 = exponential_noise(uv, freq, time_value, speed_value, seed);
    let offset_x : f32 = exponential_noise(uv, freq, time_value + 0.77, speed_value, seed + 23.0);
    let shifted_uv : vec2<f32> = fract(uv + vec2<f32>(0.5, 0.5));
    let offset_y : f32 = exponential_noise(shifted_uv, freq, time_value + 1.23, speed_value, seed + 47.0);

    let offset_vec : vec2<f32> = vec2<f32>(
        (offset_x * 2.0 - 1.0) * displacement * pixel_size.x,
        (offset_y * 2.0 - 1.0) * displacement * pixel_size.y,
    );
    let warped_uv : vec2<f32> = fract(uv + offset_vec);
    let warped : f32 = exponential_noise(warped_uv, freq, time_value, speed_value, seed + 59.0);
    return clamp01((base + warped) * 0.5);
}

fn store_color(index : u32, color : vec4<f32>) {
    output_buffer[index + 0u] = color.x;
    output_buffer[index + 1u] = color.y;
    output_buffer[index + 2u] = color.z;
    output_buffer[index + 3u] = color.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = max(as_u32(params.size.x), 1u);
    let height : u32 = max(as_u32(params.size.y), 1u);

    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let base_color : vec4<f32> = textureLoad(input_texture, coords, 0);

    let dims : vec2<f32> = vec2<f32>(
        max(params.size.x, 1.0),
        max(params.size.y, 1.0),
    );
    let pixel_size : vec2<f32> = vec2<f32>(1.0 / dims.x, 1.0 / dims.y);
    let uv : vec2<f32> = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) * pixel_size;

    let time_value : f32 = params.time_speed.x;
    let speed_value : f32 = params.time_speed.y;
    let strength : f32 = max(params.time_speed.z, 0.0);
    let debug_mode : f32 = params.time_speed.w;

    let freq_mask : vec2<f32> = freq_for_shape(5.0, dims.x, dims.y);
    let mask_refracted : f32 = refracted_scalar_field(
        uv,
        freq_mask,
        time_value,
        speed_value,
        pixel_size,
        1.0,
        11.0,
    );
    let mask_gradient : f32 = chebyshev_gradient(
        uv,
        freq_mask,
        time_value,
        speed_value,
        pixel_size,
        1.0,
        11.0,
    );
    let mask_value : f32 = clamp01(mix(mask_refracted, mask_gradient, 0.125));

    let mask_power : f32 = clamp01(mask_value * mask_value * 0.075);
    var dusty : vec3<f32> = mix(
        base_color.xyz,
        vec3<f32>(0.25, 0.25, 0.25),
        vec3<f32>(mask_power),
    );

    let freq_specks : vec2<f32> = dims * 0.25;
    let dropout : f32 = dropout_mask(uv, dims, 37.0);
    let specks_field : f32 = refracted_exponential(
        uv,
        freq_specks,
        time_value,
        speed_value,
        pixel_size,
        0.25,
        71.0,
    ) * dropout;
    let trimmed : f32 = clamp01((specks_field - 0.625) / 0.375);
    let specks : f32 = 1.0 - sqrt(trimmed);

    let freq_sparse : vec2<f32> = dims;
    let sparse_mask : f32 = select(0.0, 1.0, hash21(uv * dims + vec2<f32>(113.0, 171.0)) < 0.15);
    let sparse_noise : f32 = exponential_noise(
        uv,
        freq_sparse,
        time_value,
        speed_value,
        131.0,
    ) * sparse_mask;

    dusty = mix(dusty, vec3<f32>(sparse_noise), vec3<f32>(0.075));
    dusty = dusty * specks;

    let blend_mask : f32 = clamp01(mask_value * 0.75 * strength);
    
    // Debug visualization modes
    var final_rgb : vec3<f32>;
    if (debug_mode > 3.5) {
        // Mode 4: Show sparse noise
        final_rgb = vec3<f32>(sparse_noise);
    } else if (debug_mode > 2.5) {
        // Mode 3: Show specks
        final_rgb = vec3<f32>(specks);
    } else if (debug_mode > 1.5) {
        // Mode 2: Show dusty layer
        final_rgb = dusty;
    } else if (debug_mode > 0.5) {
        // Mode 1: Show mask
        final_rgb = vec3<f32>(mask_value);
    } else {
        // Mode 0: Normal blending - blend the dusty grime layer over the input
        // For now, just show the dusty layer directly to verify it's not identical to input
        final_rgb = clamp(dusty, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    
    let final_color : vec4<f32> = vec4<f32>(
        clamp01(final_rgb.x),
        clamp01(final_rgb.y),
        clamp01(final_rgb.z),
        base_color.w,
    );

    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;
    store_color(base_index, final_color);
}
