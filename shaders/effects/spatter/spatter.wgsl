// Spatter effect that recreates the Noisemaker Python implementation.
// Generates a smeared base mask, applies two rounds of spatter dots,
// erodes portions of the mask with a ridge noise removal pass, and
// optionally blends in a colored splash layer.

const CHANNEL_CAP : u32 = 4u;
const BLEND_FEATHER : f32 = 0.005;

fn pick_layer(index : u32, base_rgb : vec3<f32>, tinted_rgb : vec3<f32>) -> vec3<f32> {
    if (index == 0u) {
        return base_rgb;
    }
    return tinted_rgb;
}

fn blend_spatter_layers(
    control : f32,
    base_rgb : vec3<f32>,
    tinted_rgb : vec3<f32>,
) -> vec3<f32> {
    let normalized : f32 = clamp01(control);
    let layer_count : u32 = 2u;
    let extended_count : u32 = layer_count + 1u;
    let scaled : f32 = normalized * f32(extended_count);
    let floor_value : f32 = floor(scaled);
    let floor_index : u32 = min(u32(floor_value), extended_count - 1u);
    let next_index : u32 = (floor_index + 1u) % extended_count;
    let lower_layer : vec3<f32> = pick_layer(floor_index, base_rgb, tinted_rgb);
    let upper_layer : vec3<f32> = pick_layer(next_index, base_rgb, tinted_rgb);
    let fract_value : f32 = scaled - floor_value;
    let safe_feather : f32 = max(BLEND_FEATHER, 1e-6);
    let feather_mix : f32 = clamp(
        (fract_value - (1.0 - safe_feather)) / safe_feather,
        0.0,
        1.0,
    );
    return mix(lower_layer, upper_layer, feather_mix);
}

struct SpatterParams {
    size : vec4<f32>,   // (width, height, channels, unused)
    color : vec4<f32>,  // (mode flag, base_r, base_g, base_b)
    timing : vec4<f32>, // (time, speed, unused, unused)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : SpatterParams;

fn as_u32(value : f32) -> u32 {
    if (value <= 0.0) {
        return 0u;
    }
    return u32(value + 0.5);
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
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
    let safe_width : f32 = max(width, 1.0);
    let safe_height : f32 = max(height, 1.0);
    var fx : f32 = base_freq;
    var fy : f32 = base_freq;
    if (abs(safe_height - safe_width) >= 0.5) {
        if (safe_height < safe_width) {
            fx = base_freq * safe_width / safe_height;
        } else {
            fy = base_freq * safe_height / safe_width;
        }
    }
    return vec2<f32>(fx, fy);
}

fn periodic_offset(time : f32, speed : f32, seed : f32) -> vec2<f32> {
    let angle : f32 = time * (0.35 + speed * 0.15) + seed * 1.97;
    let radius : f32 = 0.25 + 0.45 * hash21(vec2<f32>(seed, seed + 19.0));
    return vec2<f32>(cos(angle), sin(angle)) * radius;
}

fn simple_multires_exp(
    uv : vec2<f32>,
    base_freq : vec2<f32>,
    octaves : u32,
    time : f32,
    speed : f32,
    seed : f32,
) -> f32 {
    var freq : vec2<f32> = base_freq;
    var amplitude : f32 = 0.5;
    var accum : f32 = 0.0;
    var weight : f32 = 0.0;
    for (var octave : u32 = 0u; octave < octaves; octave = octave + 1u) {
        let octave_seed : f32 = seed + f32(octave) * 37.17;
        let offset : vec2<f32> = periodic_offset(time + f32(octave) * 0.31, speed, octave_seed);
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

fn adjust_brightness_scalar(value : f32, amount : f32) -> f32 {
    let adjusted : f32 = value + amount;
    return clamp(adjusted, -1.0, 1.0);
}

fn adjust_contrast_scalar(value : f32, amount : f32) -> f32 {
    let normalized : f32 = (value + 1.0) * 0.5;
    let contrasted : f32 = (normalized - 0.5) * amount + 0.5;
    return clamp01(contrasted);
}

fn rgb_to_hsv(rgb : vec3<f32>) -> vec3<f32> {
    let c_max : f32 = max(max(rgb.x, rgb.y), rgb.z);
    let c_min : f32 = min(min(rgb.x, rgb.y), rgb.z);
    let delta : f32 = c_max - c_min;

    var hue : f32 = 0.0;
    if (delta > 0.0) {
        if (c_max == rgb.x) {
            hue = (rgb.y - rgb.z) / delta;
        } else if (c_max == rgb.y) {
            hue = (rgb.z - rgb.x) / delta + 2.0;
        } else {
            hue = (rgb.x - rgb.y) / delta + 4.0;
        }
        hue = fract(hue / 6.0);
    }

    let sat : f32 = select(0.0, delta / c_max, c_max > 0.0);
    return vec3<f32>(hue, sat, c_max);
}

fn hsv_to_rgb(hsv : vec3<f32>) -> vec3<f32> {
    let hue : f32 = fract(hsv.x) * 6.0;
    let sat : f32 = clamp01(hsv.y);
    let val : f32 = clamp01(hsv.z);
    let c : f32 = val * sat;
    let x : f32 = c * (1.0 - abs(fract(hue) * 2.0 - 1.0));
    let m : f32 = val - c;
    if (hue < 1.0) {
        return vec3<f32>(c + m, x + m, m);
    }
    if (hue < 2.0) {
        return vec3<f32>(x + m, c + m, m);
    }
    if (hue < 3.0) {
        return vec3<f32>(m, c + m, x + m);
    }
    if (hue < 4.0) {
        return vec3<f32>(m, x + m, c + m);
    }
    if (hue < 5.0) {
        return vec3<f32>(x + m, m, c + m);
    }
    return vec3<f32>(c + m, m, x + m);
}

fn random_range(seed : vec2<f32>, min_value : f32, max_value : f32) -> f32 {
    return mix(min_value, max_value, hash21(seed));
}

fn random_range_u32(seed : vec2<f32>, min_value : u32, max_value : u32) -> u32 {
    if (max_value <= min_value) {
        return min_value;
    }
    let span : f32 = f32(max_value - min_value + 1u);
    let value : f32 = hash21(seed) * span;
    return min_value + u32(floor(value));
}

fn apply_warp(
    uv : vec2<f32>,
    base_freq : vec2<f32>,
    octaves : u32,
    time : f32,
    speed : f32,
    pixel_size : vec2<f32>,
    displacement : f32,
    seed : f32,
) -> vec2<f32> {
    var offset : vec2<f32> = vec2<f32>(0.0, 0.0);
    var freq : vec2<f32> = base_freq;
    var amp : f32 = displacement;
    for (var octave : u32 = 0u; octave < octaves; octave = octave + 1u) {
        let octave_seed : f32 = seed + f32(octave) * 29.0;
        let noise_x : f32 = simple_multires_exp(
            uv,
            freq,
            3u,
            time + f32(octave) * 0.37,
            speed,
            octave_seed,
        );
        let noise_y : f32 = simple_multires_exp(
            uv,
            freq,
            3u,
            time + f32(octave) * 0.59,
            speed,
            octave_seed + 13.0,
        );
        let signed : vec2<f32> = vec2<f32>(noise_x * 2.0 - 1.0, noise_y * 2.0 - 1.0);
        offset = offset + signed * amp;
        freq = freq * 2.0;
        amp = amp * 0.5;
    }
    return fract(uv + offset * pixel_size);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.size.x);
    let height : u32 = as_u32(params.size.y);
    if (width == 0u || height == 0u) {
        return;
    }
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let base_color : vec4<f32> = textureLoad(input_texture, coords, 0);

    let dims : vec2<f32> = vec2<f32>(max(params.size.x, 1.0), max(params.size.y, 1.0));
    let pixel_size : vec2<f32> = vec2<f32>(1.0 / dims.x, 1.0 / dims.y);
    let uv : vec2<f32> = (vec2<f32>(f32(coords.x), f32(coords.y)) + 0.5) * pixel_size;

    let time : f32 = params.timing.x;
    let speed : f32 = params.timing.y;

    let smear_freq_choice : u32 = random_range_u32(vec2<f32>(time * 0.17 + 3.0, 29.0), 3u, 6u);
    let smear_freq : vec2<f32> = freq_for_shape(f32(smear_freq_choice), dims.x, dims.y);

    let warp_freq_x_choice : u32 = random_range_u32(vec2<f32>(time * 0.11 + 19.0, 37.0), 2u, 3u);
    let warp_freq_y_choice : u32 = random_range_u32(vec2<f32>(time * 0.13 + 41.0, 71.0), 1u, 3u);
    let warp_freq : vec2<f32> = vec2<f32>(f32(warp_freq_x_choice), f32(warp_freq_y_choice));
    let warp_octaves : u32 = random_range_u32(vec2<f32>(time * 0.07 + 11.0, 131.0), 1u, 2u);
    let warp_displacement : f32 = 1.0 + random_range(
        vec2<f32>(time * 0.19 + 53.0, 173.0),
        0.0,
        1.0,
    );

    let warped_uv : vec2<f32> = apply_warp(
        uv,
        warp_freq,
        warp_octaves,
        time,
        speed,
        pixel_size,
        warp_displacement,
        17.0,
    );
    var smear : f32 = simple_multires_exp(warped_uv, smear_freq, 6u, time, speed, 23.0);

    let spatter_freq1_choice : u32 = random_range_u32(vec2<f32>(time * 0.37 + 5.0, 59.0), 32u, 64u);
    let spatter_freq1 : vec2<f32> = freq_for_shape(f32(spatter_freq1_choice), dims.x, dims.y);
    let spatter_noise1 : f32 = simple_multires_exp(uv, spatter_freq1, 4u, time, speed, 43.0);
    let spatter_layer1 : f32 = adjust_contrast_scalar(
        adjust_brightness_scalar(spatter_noise1, -1.0),
        4.0,
    );
    smear = max(smear, spatter_layer1);

    let spatter_freq2_choice : u32 = random_range_u32(
        vec2<f32>(time * 0.41 + 13.0, 97.0),
        150u,
        200u,
    );
    let spatter_freq2 : vec2<f32> = freq_for_shape(f32(spatter_freq2_choice), dims.x, dims.y);
    let spatter_noise2 : f32 = simple_multires_exp(uv, spatter_freq2, 4u, time, speed, 71.0);
    let spatter_layer2 : f32 = adjust_contrast_scalar(
        adjust_brightness_scalar(spatter_noise2, -1.25),
        4.0,
    );
    smear = max(smear, spatter_layer2);

    let removal_freq_choice : u32 = random_range_u32(vec2<f32>(time * 0.23 + 31.0, 149.0), 2u, 3u);
    let removal_freq : vec2<f32> = freq_for_shape(f32(removal_freq_choice), dims.x, dims.y);
    let removal_noise : f32 = simple_multires_exp(uv, removal_freq, 3u, time, speed, 89.0);
    let removal_mask : f32 = ridge(removal_noise);
    smear = max(0.0, smear - removal_mask);

    let smear_mask : f32 = clamp01(smear);

    let channel_count : u32 = sanitized_channel_count(params.size.z);
    let color_toggle : f32 = params.color.x;
    let base_color_rgb : vec3<f32> = clamp(
        vec3<f32>(params.color.y, params.color.z, params.color.w),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );

    var splash_rgb : vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    if (color_toggle > 0.5 && channel_count >= 3u) {
        if (color_toggle > 1.5) {
            splash_rgb = base_color_rgb;
        } else {
            let base_hsv : vec3<f32> = rgb_to_hsv(base_color_rgb);
            let hue_jitter : f32 = hash21(vec2<f32>(floor(time * 60.0) + 211.0, 307.0)) - 0.5;
            let randomized_hsv : vec3<f32> = vec3<f32>(
                base_hsv.x + hue_jitter,
                base_hsv.y,
                base_hsv.z,
            );
            splash_rgb = hsv_to_rgb(randomized_hsv);
        }
    }

    let tinted : vec3<f32> = base_color.xyz * splash_rgb;
    let final_rgb : vec3<f32> = blend_spatter_layers(smear_mask, base_color.xyz, tinted);

    let base_index : u32 = (gid.y * width + gid.x) * CHANNEL_CAP;
    output_buffer[base_index + 0u] = clamp01(final_rgb.x);
    output_buffer[base_index + 1u] = clamp01(final_rgb.y);
    output_buffer[base_index + 2u] = clamp01(final_rgb.z);
    output_buffer[base_index + 3u] = base_color.w;
}
