// Nebula effect: attenuates the source image with ridged value-noise layers and
// adds a tinted emission pass derived from the positive components of the
// overlay noise. Matches the Python implementation in noisemaker.effects.nebula.

const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647693;
const FLOAT_SCALE : f32 = 1.0 / f32(0xffffffffu);
const EPSILON : f32 = 1e-4;

const PRIMARY_FREQ_SEED : u32 = 0x0011u;
const SECONDARY_FREQ_SEED : u32 = 0x0021u;
const ROTATION_SEED : u32 = 0x0031u;
const HUE_JITTER_SEED_A : u32 = 0x00a5u;
const HUE_JITTER_SEED_B : u32 = 0x00b7u;

struct NebulaParams {
    size : vec4<f32>,       // (width, height, channels, time)
    time_speed : vec4<f32>, // (speed, unused, unused, unused)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : NebulaParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn saturate(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn wrap_unit(value : f32) -> f32 {
    return value - floor(value);
}

fn wrap_coord_unit(coord : vec2<f32>) -> vec2<f32> {
    return vec2<f32>(wrap_unit(coord.x), wrap_unit(coord.y));
}

fn rotate_coord(coord : vec2<f32>, angle : f32) -> vec2<f32> {
    let center : vec2<f32> = vec2<f32>(0.5, 0.5);
    let offset : vec2<f32> = coord - center;
    let cos_a : f32 = cos(angle);
    let sin_a : f32 = sin(angle);
    let rotated : vec2<f32> = vec2<f32>(
        offset.x * cos_a - offset.y * sin_a,
        offset.x * sin_a + offset.y * cos_a
    );
    return rotated + center;
}

fn ridge_transform(value : f32) -> f32 {
    return 1.0 - abs(value * 2.0 - 1.0);
}

fn pcg3d(value : vec3<u32>) -> vec3<u32> {
    var v : vec3<u32> = value * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

fn seeded_float(seed : u32) -> f32 {
    let hashed : vec3<u32> = pcg3d(vec3<u32>(seed, seed ^ 0x9e3779b9u, seed + 0x7f4a7c15u));
    return f32(hashed.x) * FLOAT_SCALE;
}

fn seeded_int(seed : u32, min_value : i32, max_value : i32) -> i32 {
    let span : i32 = max_value - min_value + 1;
    if (span <= 1) {
        return min_value;
    }
    let choice : i32 = i32(floor(seeded_float(seed) * f32(span)));
    return min_value + clamp(choice, 0, span - 1);
}

fn random_from_cell(cell : vec3<i32>, seed : u32) -> f32 {
    let hashed : vec3<u32> = vec3<u32>(
        bitcast<u32>(cell.x) ^ seed,
        bitcast<u32>(cell.y) ^ (seed * 0x9e3779b9u + 0x7f4a7c15u),
        bitcast<u32>(cell.z) ^ (seed * 0x632be59bu + 0x5bf03635u)
    );
    let noise : vec3<u32> = pcg3d(hashed);
    return f32(noise.x) * FLOAT_SCALE;
}

fn fade(t : f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

fn sample_value_noise(point : vec2<f32>, seed : u32) -> f32 {
    let cell : vec2<i32> = vec2<i32>(floor(point));
    let frac : vec2<f32> = fract(point);
    let base_seed : u32 = seed ^ (seed >> 17);
    let z_layer : i32 = i32(base_seed & 1023u);
    let p00 : f32 = random_from_cell(vec3<i32>(cell.x, cell.y, z_layer), base_seed);
    let p10 : f32 = random_from_cell(vec3<i32>(cell.x + 1, cell.y, z_layer), base_seed);
    let p01 : f32 = random_from_cell(vec3<i32>(cell.x, cell.y + 1, z_layer), base_seed);
    let p11 : f32 = random_from_cell(vec3<i32>(cell.x + 1, cell.y + 1, z_layer), base_seed);
    let ux : f32 = fade(frac.x);
    let uy : f32 = fade(frac.y);
    let mix_x0 : f32 = mix(p00, p10, ux);
    let mix_x1 : f32 = mix(p01, p11, ux);
    return mix(mix_x0, mix_x1, uy);
}

fn octave_motion(octave : u32, time_value : f32, speed_value : f32, seed : u32) -> vec2<f32> {
    let hashed : vec3<u32> = pcg3d(vec3<u32>(seed + octave * 0x9e3779b9u, seed ^ 0x7f4a7c15u, octave + 1u));
    let jitter : vec2<f32> = (vec2<f32>(f32(hashed.x), f32(hashed.y)) * FLOAT_SCALE - vec2<f32>(0.5, 0.5)) * 0.75;
    let angle : f32 = f32(hashed.z) * FLOAT_SCALE * TAU;
    let motion_scale : f32 = time_value * speed_value * 0.35;
    let direction : vec2<f32> = vec2<f32>(cos(angle), sin(angle));
    return jitter + direction * motion_scale;
}

fn simple_multires(
    uv : vec2<f32>,
    dims : vec2<i32>,
    base_freq : vec2<i32>,
    octaves : u32,
    time_value : f32,
    speed_value : f32,
    seed : u32,
    use_ridge : bool,
    use_exp : bool
) -> f32 {
    var total : f32 = 0.0;
    var weight : f32 = 0.0;
    var current_seed : u32 = seed;

    for (var octave : u32 = 1u; octave <= octaves; octave = octave + 1u) {
        let multiplier : u32 = 1u << octave;
        let scaled_x : f32 = f32(base_freq.x) * 0.5 * f32(multiplier);
        let scaled_y : f32 = f32(base_freq.y) * 0.5 * f32(multiplier);
        let freq_x : i32 = max(i32(floor(scaled_x)), 1);
        let freq_y : i32 = max(i32(floor(scaled_y)), 1);

        if (freq_x > dims.x && freq_y > dims.y) {
            break;
        }

        let freq_vec : vec2<f32> = vec2<f32>(f32(freq_x), f32(freq_y));
        let motion : vec2<f32> = octave_motion(octave - 1u, time_value, speed_value, current_seed);
        let sample_pos : vec2<f32> = uv * freq_vec + motion;
        var noise_value : f32 = sample_value_noise(sample_pos, current_seed);

        if (use_ridge) {
            noise_value = ridge_transform(noise_value);
        }
        if (use_exp) {
            noise_value = pow(noise_value, 4.0);
        }

        let amplitude : f32 = 1.0 / f32(multiplier);
        total = total + noise_value * amplitude;
        weight = weight + amplitude;
        current_seed = current_seed ^ (0x9e3779b9u + octave * 0x7f4a7c15u);
    }

    if (weight <= EPSILON) {
        return 0.0;
    }

    return clamp(total / weight, 0.0, 1.0);
}

fn hsv_to_rgb(hsv : vec3<f32>) -> vec3<f32> {
    let h : f32 = hsv.x;
    let s : f32 = saturate(hsv.y);
    let v : f32 = saturate(hsv.z);
    let dh : f32 = h * 6.0;
    let dr : f32 = saturate(abs(dh - 3.0) - 1.0);
    let dg : f32 = saturate(-abs(dh - 2.0) + 2.0);
    let db : f32 = saturate(-abs(dh - 4.0) + 2.0);
    let one_minus_s : f32 = 1.0 - s;
    let sr : f32 = s * dr;
    let sg : f32 = s * dg;
    let sb : f32 = s * db;
    let r : f32 = (one_minus_s + sr) * v;
    let g : f32 = (one_minus_s + sg) * v;
    let b : f32 = (one_minus_s + sb) * v;
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn tint_overlay(overlay_value : f32, channel_count : u32) -> vec4<f32> {
    let clamped : f32 = clamp(overlay_value, 0.0, 1.0);

    if (channel_count < 3u) {
        return vec4<f32>(clamped, clamped, clamped, clamped);
    }

    let hue_bias_a : f32 = seeded_float(HUE_JITTER_SEED_A) * 0.33333334;
    let hue_bias_b : f32 = seeded_float(HUE_JITTER_SEED_B);
    let hue : f32 = fract(clamped * 0.33333334 + hue_bias_a + hue_bias_b);
    let saturation : f32 = clamped;
    let value : f32 = clamped;
    let rgb : vec3<f32> = hsv_to_rgb(vec3<f32>(hue, saturation, value));
    return vec4<f32>(rgb, clamped);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims_tex : vec2<u32> = textureDimensions(input_texture, 0);
    let width : u32 = select(as_u32(params.size.x), dims_tex.x, dims_tex.x > 0u);
    let height : u32 = select(as_u32(params.size.y), dims_tex.y, dims_tex.y > 0u);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let channel_count : u32 = max(as_u32(params.size.z), 1u);
    let dims_f : vec2<f32> = vec2<f32>(max(f32(width), 1.0), max(f32(height), 1.0));
    let dims_i : vec2<i32> = vec2<i32>(i32(width), i32(height));
    let pixel : vec2<f32> = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
    let uv : vec2<f32> = pixel / dims_f;

    let time_value : f32 = params.size.w;
    let speed_value : f32 = params.time_speed.x;

    let primary_freq_x : i32 = seeded_int(PRIMARY_FREQ_SEED, 3, 4);
    let secondary_freq_x : i32 = seeded_int(SECONDARY_FREQ_SEED, 2, 4);
    let rotation_degrees : f32 = f32(seeded_int(ROTATION_SEED, -15, 15));

    let rotation_angle : f32 = radians(rotation_degrees) + time_value * speed_value * 0.05;
    let rotated_uv : vec2<f32> = wrap_coord_unit(rotate_coord(uv, rotation_angle));

    let primary_noise : f32 = simple_multires(
        rotated_uv,
        dims_i,
        vec2<i32>(primary_freq_x, 1),
        6u,
        time_value,
        speed_value,
        0x6c8e9cf5u,
        true,
        true
    );
    let secondary_noise : f32 = simple_multires(
        rotated_uv,
        dims_i,
        vec2<i32>(secondary_freq_x, 1),
        4u,
        time_value,
        speed_value,
        0x9e3779b9u,
        true,
        false
    );

    let overlay : f32 = (primary_noise - secondary_noise) * 0.125;
    let overlay_positive : f32 = max(overlay, 0.0);

    let base_texel : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    let tinted : vec4<f32> = tint_overlay(overlay_positive, channel_count);

    let attenuation : f32 = 1.0 - overlay;
    let attenuated_rgb : vec3<f32> = base_texel.xyz * attenuation;
    var final_rgb : vec3<f32> = attenuated_rgb + tinted.xyz;
    var final_alpha : f32 = base_texel.w;

    if (channel_count >= 4u) {
        final_alpha = base_texel.w * attenuation + tinted.w;
    }

    if (channel_count >= 3u) {
        final_rgb = clamp(final_rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    } else {
        let gray : f32 = clamp(attenuated_rgb.x + overlay_positive, 0.0, 1.0);
        final_rgb = vec3<f32>(gray, gray, gray);
    }

    // Always write 4 channels with RGBA stride expected by the viewer
    let base_index : u32 = (gid.y * width + gid.x) * 4u;
    output_buffer[base_index + 0u] = final_rgb.x;
    output_buffer[base_index + 1u] = final_rgb.y;
    output_buffer[base_index + 2u] = final_rgb.z;
    output_buffer[base_index + 3u] = clamp(final_alpha, 0.0, 1.0);
}
