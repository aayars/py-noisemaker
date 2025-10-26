// Frame effect: replicates the gritty film border treatment from the Python reference.
// Steps:
//   1. Downsample the input to mimic the softened half-resolution processing chain.
//   2. Build a soft chebyshev-based frame mask perturbed by multires value noise.
//   3. Blend in edge glow, chromatic aberration, grime, scratches, stray hairs, and grain.
//   4. Apply saturation reduction and a subtle random hue rotation.

const PI : f32 = 3.141592653589793;
const CHANNEL_COUNT : u32 = 4u;

struct FrameParams {
    size : vec4<f32>,      // (width, height, channels, unused)
    timeSpeed : vec4<f32>, // (time, speed, unused, unused)
};

@group(0) @binding(0) var inputTexture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : FrameParams;

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn as_u32(value : f32) -> u32 {
    if (value <= 0.0) {
        return 0u;
    }
    return u32(value + 0.5);
}

fn as_i32(value : f32) -> i32 {
    if (value <= 0.0) {
        return 0;
    }
    return i32(value + 0.5);
}

fn luminance(rgb : vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
}

fn hash21(p : vec2<f32>) -> f32 {
    let h : f32 = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

fn hash31(p : vec3<f32>) -> f32 {
    let h : f32 = dot(p, vec3<f32>(12.9898, 78.233, 37.719));
    return fract(sin(h) * 43758.5453);
}

fn fade(t : f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

fn value_noise(p : vec2<f32>, seed : f32) -> f32 {
    let cell : vec2<f32> = floor(p);
    let local : vec2<f32> = fract(p);

    let a : f32 = hash21(vec2<f32>(cell.x + seed, cell.y - seed));
    let b : f32 = hash21(vec2<f32>(cell.x + 1.0 - seed, cell.y + seed));
    let c : f32 = hash21(vec2<f32>(cell.x - seed, cell.y + 1.0 + seed));
    let d : f32 = hash21(vec2<f32>(cell.x + 1.0 + seed, cell.y + 1.0 - seed));

    let ux : f32 = fade(local.x);
    let uy : f32 = fade(local.y);
    let lerp_x0 : f32 = mix(a, b, ux);
    let lerp_x1 : f32 = mix(c, d, ux);
    return mix(lerp_x0, lerp_x1, uy);
}

fn simple_multires(
    uv : vec2<f32>,
    base_freq : vec2<f32>,
    time_value : f32,
    speed_value : f32,
    octaves : u32,
    seed : f32
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

        let octave_seed : f32 = seed + f32(octave) * 19.37;
        let offset : vec2<f32> = vec2<f32>(
            time_value * (0.05 + speed_value * 0.02) + octave_seed * 0.11,
            time_value * (0.09 + speed_value * 0.015) - octave_seed * 0.07
        );
        let sample : f32 = value_noise(uv * freq + offset, octave_seed);

        accum = accum + sample * amplitude;
        total_weight = total_weight + amplitude;
        freq = freq * 2.0;
        amplitude = amplitude * 0.5;
        octave = octave + 1u;
    }

    if (total_weight > 0.0) {
        return clamp01(accum / total_weight);
    }
    return 0.0;
}

fn safe_load(x : i32, y : i32, width : i32, height : i32) -> vec4<f32> {
    let sx : i32 = clamp(x, 0, max(width - 1, 0));
    let sy : i32 = clamp(y, 0, max(height - 1, 0));
    return textureLoad(inputTexture, vec2<i32>(sx, sy), 0);
}

fn downsample_color(x : i32, y : i32, width : i32, height : i32) -> vec3<f32> {
    // Average a 2x2 block at the current location to simulate downsampling
    let a : vec3<f32> = safe_load(x, y, width, height).xyz;
    let b : vec3<f32> = safe_load(x + 1, y, width, height).xyz;
    let c : vec3<f32> = safe_load(x, y + 1, width, height).xyz;
    let d : vec3<f32> = safe_load(x + 1, y + 1, width, height).xyz;
    return (a + b + c + d) * 0.25;
}

fn adjust_brightness(rgb : vec3<f32>, value : f32) -> vec3<f32> {
    return clamp(rgb + vec3<f32>(value), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn adjust_contrast(rgb : vec3<f32>, contrast : f32) -> vec3<f32> {
    let mid : vec3<f32> = vec3<f32>(0.5);
    return clamp((rgb - mid) * contrast + mid, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn adjust_saturation(rgb : vec3<f32>, factor : f32) -> vec3<f32> {
    let gray : f32 = luminance(rgb);
    let base : vec3<f32> = vec3<f32>(gray, gray, gray);
    return clamp(base + (rgb - base) * factor, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn rotate_hue(rgb : vec3<f32>, angle : f32) -> vec3<f32> {
    let cos_a : f32 = cos(angle);
    let sin_a : f32 = sin(angle);

    let to_yiq : mat3x3<f32> = mat3x3<f32>(
        vec3<f32>(0.299, 0.587, 0.114),
        vec3<f32>(0.596, -0.274, -0.322),
        vec3<f32>(0.211, -0.523, 0.312)
    );
    let from_yiq : mat3x3<f32> = mat3x3<f32>(
        vec3<f32>(1.0, 0.956, 0.621),
        vec3<f32>(1.0, -0.272, -0.647),
        vec3<f32>(1.0, -1.105, 1.702)
    );

    let yiq : vec3<f32> = to_yiq * rgb;
    let i : f32 = yiq.y * cos_a - yiq.z * sin_a;
    let q : f32 = yiq.y * sin_a + yiq.z * cos_a;
    return clamp(from_yiq * vec3<f32>(yiq.x, i, q), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn chebyshev_distance(uv : vec2<f32>, aspect : vec2<f32>) -> f32 {
    let centered : vec2<f32> = abs(uv - vec2<f32>(0.5, 0.5));
    return max(centered.x * aspect.x, centered.y * aspect.y);
}

fn frame_mask_value(
    uv : vec2<f32>,
    aspect : vec2<f32>,
    time_value : f32,
    speed_value : f32,
    seed_value : f32
) -> f32 {
    let dist : f32 = chebyshev_distance(uv, aspect);
    let noise : f32 = simple_multires(
        uv,
        vec2<f32>(64.0, 64.0),
        time_value + seed_value * 2.0,
        speed_value,
        8u,
        11.0
    );
    // Narrower frame with subtle anti-aliased edge
    let threshold : f32 = 0.38 + (noise - 0.5) * 0.025;
    let edge_width : f32 = 0.008; // Small AA band
    let mask : f32 = clamp((dist - threshold) / edge_width, 0.0, 1.0);
    return mask;
}

fn vignette_mask(uv : vec2<f32>, aspect : vec2<f32>) -> f32 {
    let dist : f32 = length((uv - vec2<f32>(0.5, 0.5)) * aspect);
    return clamp(dist, 0.0, 1.0);
}

fn light_leak_color(uv : vec2<f32>, time_value : f32, speed_value : f32) -> vec3<f32> {
    let motion : f32 = time_value * (0.1 + speed_value * 0.05);
    let corner : f32 = pow(clamp01((1.0 - uv.x) * uv.y), 2.2);
    let sweep : f32 = sin((uv.x + uv.y + motion) * PI) * 0.5 + 0.5;
    let warm : vec3<f32> = vec3<f32>(1.0, 0.75, 0.55);
    let cool : vec3<f32> = vec3<f32>(0.75, 0.6, 1.0);
    return mix(cool, warm, sweep) * (0.08 + corner * 0.25);
}

fn grime_overlay(uv : vec2<f32>, time_value : f32, speed_value : f32, seed_value : f32) -> f32 {
    let drift : f32 = time_value * (0.25 + speed_value * 0.1);
    let coarse : f32 = simple_multires(
        uv + vec2<f32>(drift, -drift * 0.5),
        vec2<f32>(8.0, 8.0),
        time_value,
        speed_value,
        8u,
        seed_value * 3.1
    );
    let streaks : f32 = simple_multires(
        vec2<f32>(uv.x * 2.0, uv.y * 6.0) + drift,
        vec2<f32>(12.0, 18.0),
        time_value,
        speed_value,
        8u,
        seed_value * 4.7
    );
    let speck : f32 = pow(
        clamp01(1.0 - abs(fract(uv.y * 180.0 + seed_value * 5.0) - 0.5) * 2.0),
        6.0
    );
    return clamp(coarse * 0.7 + streaks * 0.2 + speck * 0.3, 0.0, 1.0);
}

fn scratch_mask(uv : vec2<f32>, time_value : f32, speed_value : f32, seed_value : f32) -> f32 {
    // Temporarily disable scratches to test if they're causing the vertical lines
    return 0.0;
}

fn stray_hair_mask(uv : vec2<f32>, time_value : f32, speed_value : f32, seed_value : f32) -> f32 {
    // Use noise-based approach instead of vertical strands
    let base_mask : f32 = simple_multires(
        uv,
        vec2<f32>(4.0, 4.0),
        time_value,
        speed_value,
        8u,
        23.0
    );
    
    // Create sparse, irregular patterns instead of regular vertical strands
    let threshold : f32 = 0.975 + seed_value * 0.02;
    let hair_presence : f32 = select(0.0, 1.0, base_mask > threshold);
    
    return clamp(hair_presence * 0.35, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = max(as_u32(params.size.x), 1u);
    let height : u32 = max(as_u32(params.size.y), 1u);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let width_f : f32 = max(params.size.x, 1.0);
    let height_f : f32 = max(params.size.y, 1.0);
    let time_value : f32 = params.timeSpeed.x;
    let speed_value : f32 = params.timeSpeed.y;
    let seed_value : f32 = hash31(vec3<f32>(
        time_value * 0.123,
        speed_value * 1.37,
        0.417
    ));

    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;
    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let sample : vec4<f32> = textureLoad(inputTexture, coords, 0);

    let width_i : i32 = as_i32(params.size.x);
    let height_i : i32 = as_i32(params.size.y);

    var color : vec3<f32> = sample.xyz;
    color = adjust_brightness(color, 0.1);
    color = adjust_contrast(color, 0.75);

    let uv : vec2<f32> = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5, 0.5)) /
        vec2<f32>(width_f, height_f);
    let aspect : vec2<f32> = vec2<f32>(width_f / height_f, 1.0);

    let leak_color : vec3<f32> = clamp(
        color + light_leak_color(uv, time_value, speed_value),
        vec3<f32>(0.0),
        vec3<f32>(1.0)
    );
    color = mix(color, leak_color, 0.125);

    let vignette : f32 = pow(clamp01(vignette_mask(uv, aspect)), 1.25);
    color = mix(color, color * 0.75 + vec3<f32>(0.05) * 0.25, vignette * 0.75);

    let mask_value : f32 = frame_mask_value(uv, aspect, time_value, speed_value, seed_value);

    let base_noise : f32 = simple_multires(
        uv,
        vec2<f32>(64.0, 64.0),
        time_value,
        speed_value,
        8u,
        7.0
    );
    let delta_x : vec2<f32> = vec2<f32>(1.0 / width_f, 0.0);
    let delta_y : vec2<f32> = vec2<f32>(0.0, 1.0 / height_f);
    let grad_x : f32 = simple_multires(
        uv + delta_x,
        vec2<f32>(64.0, 64.0),
        time_value,
        speed_value,
        8u,
        9.0
    ) - base_noise;
    let grad_y : f32 = simple_multires(
        uv + delta_y,
        vec2<f32>(64.0, 64.0),
        time_value,
        speed_value,
        8u,
        13.0
    ) - base_noise;
    let gradient_mag : f32 = clamp01(length(vec2<f32>(grad_x, grad_y)) * 12.0);
    let edge_value : f32 = 0.9 + gradient_mag * 0.1;
    let edge_texture : vec3<f32> = vec3<f32>(edge_value);
    // Frame should fully replace image where mask > 0, not blend
    let blended_frame : vec3<f32> = clamp(
        select(color, edge_texture, mask_value > 0.5),
        vec3<f32>(0.0),
        vec3<f32>(1.0)
    );

    var chroma : vec3<f32> = blended_frame;
    // Only apply aberration where mask is low (center area, not frame)
    let aberration : f32 = 0.00666 * (1.0 - mask_value);
    let offset : i32 = i32(round(aberration * width_f));
    if (offset > 0) {
        let red_sample : vec3<f32> = safe_load(coords.x + offset, coords.y, width_i, height_i).xyz;
        let blue_sample : vec3<f32> = safe_load(coords.x - offset, coords.y, width_i, height_i).xyz;
        chroma.x = clamp01(mix(chroma.x, red_sample.x, 0.35));
        chroma.z = clamp01(mix(chroma.z, blue_sample.z, 0.35));
    }

    let grime : f32 = grime_overlay(uv, time_value, speed_value, seed_value);
    chroma = mix(chroma, chroma * 0.82 + vec3<f32>(0.24, 0.18, 0.12) * 0.35, grime * 0.5);

    let scratches : f32 = scratch_mask(uv, time_value, speed_value, seed_value);
    let scratch_lift : f32 = clamp01(scratches * 8.0);
    chroma = max(chroma, vec3<f32>(scratch_lift));

    let grain_strength : f32 = 0.35;
    let grain_seed : vec3<f32> = vec3<f32>(
        uv * vec2<f32>(width_f, height_f),
        time_value * speed_value + seed_value * 13.0
    );
    let grain_noise : f32 = (
        hash31(grain_seed + vec3<f32>(0.37, 0.11, 0.53)) - 0.5
    ) * grain_strength;
    chroma = clamp(
        chroma + grain_noise * (0.25 + mask_value * 0.3),
        vec3<f32>(0.0),
        vec3<f32>(1.0)
    );

    chroma = adjust_saturation(chroma, 0.5);
    let hue_shift : f32 = (hash31(vec3<f32>(seed_value, time_value, speed_value)) - 0.5) * 0.1;
    chroma = rotate_hue(chroma, hue_shift);

    let alpha : f32 = clamp01(sample.w * mix(1.0, 0.88, mask_value));

    outputBuffer[base_index + 0u] = chroma.x;
    outputBuffer[base_index + 1u] = chroma.y;
    outputBuffer[base_index + 2u] = chroma.z;
    outputBuffer[base_index + 3u] = alpha;
}
