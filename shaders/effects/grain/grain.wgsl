// Grain: blend the source image with animated value noise.
// Mirrors noisemaker.effects.grain, which calls value.values()
// using simplex-based value noise with bicubic interpolation.

const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;
const UINT32_TO_FLOAT : f32 = 1.0 / 4294967296.0;
const CHANNEL_COUNT : u32 = 4u;
const INTERPOLATION_CONSTANT : u32 = 0u;
const INTERPOLATION_LINEAR : u32 = 1u;
const INTERPOLATION_COSINE : u32 = 2u;
const INTERPOLATION_BICUBIC : u32 = 3u;
const BASE_SEED : u32 = 0x1234u;

struct GrainParams {
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
@group(0) @binding(2) var<uniform> params : GrainParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn pcg3d(v_in : vec3<u32>) -> vec3<u32> {
    var v : vec3<u32> = v_in * 1664525u + 1013904223u;
    v.x = v.x + v.y * v.z;
    v.y = v.y + v.z * v.x;
    v.z = v.z + v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x = v.x + v.y * v.z;
    v.y = v.y + v.z * v.x;
    v.z = v.z + v.x * v.y;
    return v;
}

fn random_from_cell_3d(cell : vec3<i32>, seed : u32) -> f32 {
    let hashed : vec3<u32> = vec3<u32>(
        bitcast<u32>(cell.x) ^ seed,
        bitcast<u32>(cell.y) ^ (seed * 0x9e3779b9u + 0x7f4a7c15u),
        bitcast<u32>(cell.z) ^ (seed * 0x632be59bu + 0x5bf03635u),
    );
    let noise : vec3<u32> = pcg3d(hashed);
    return f32(noise.x) * UINT32_TO_FLOAT;
}

fn periodic_value(time_value : f32, sample : f32) -> f32 {
    return (sin((time_value - sample) * TAU) + 1.0) * 0.5;
}

fn interpolation_weight(value : f32, spline_order : u32) -> f32 {
    if (spline_order == INTERPOLATION_COSINE) {
        let clamped : f32 = clamp(value, 0.0, 1.0);
        let angle : f32 = clamped * PI;
        let cos_value : f32 = cos(angle);
        return (1.0 - cos_value) * 0.5;
    }
    return value;
}

fn blend_cubic(a : f32, b : f32, c : f32, d : f32, g : f32) -> f32 {
    let t : f32 = clamp(g, 0.0, 1.0);
    let t2 : f32 = t * t;
    let a0 : f32 = ((d - c) - a) + b;
    let a1 : f32 = (a - b) - a0;
    let a2 : f32 = c - a;
    let a3 : f32 = b;
    let term1 : f32 = (a0 * t) * t2;
    let term2 : f32 = a1 * t2;
    let term3 : f32 = (a2 * t) + a3;
    return (term1 + term2) + term3;
}

fn sample_bicubic_layer(
    cell : vec2<i32>,
    frac : vec2<f32>,
    z_cell : i32,
    base_seed : u32,
) -> f32 {
    let row0 : f32 = blend_cubic(
        random_from_cell_3d(vec3<i32>(cell.x - 1, cell.y - 1, z_cell), base_seed),
        random_from_cell_3d(vec3<i32>(cell.x + 0, cell.y - 1, z_cell), base_seed),
        random_from_cell_3d(vec3<i32>(cell.x + 1, cell.y - 1, z_cell), base_seed),
        random_from_cell_3d(vec3<i32>(cell.x + 2, cell.y - 1, z_cell), base_seed),
        frac.x,
    );
    let row1 : f32 = blend_cubic(
        random_from_cell_3d(vec3<i32>(cell.x - 1, cell.y + 0, z_cell), base_seed),
        random_from_cell_3d(vec3<i32>(cell.x + 0, cell.y + 0, z_cell), base_seed),
        random_from_cell_3d(vec3<i32>(cell.x + 1, cell.y + 0, z_cell), base_seed),
        random_from_cell_3d(vec3<i32>(cell.x + 2, cell.y + 0, z_cell), base_seed),
        frac.x,
    );
    let row2 : f32 = blend_cubic(
        random_from_cell_3d(vec3<i32>(cell.x - 1, cell.y + 1, z_cell), base_seed),
        random_from_cell_3d(vec3<i32>(cell.x + 0, cell.y + 1, z_cell), base_seed),
        random_from_cell_3d(vec3<i32>(cell.x + 1, cell.y + 1, z_cell), base_seed),
        random_from_cell_3d(vec3<i32>(cell.x + 2, cell.y + 1, z_cell), base_seed),
        frac.x,
    );
    let row3 : f32 = blend_cubic(
        random_from_cell_3d(vec3<i32>(cell.x - 1, cell.y + 2, z_cell), base_seed),
        random_from_cell_3d(vec3<i32>(cell.x + 0, cell.y + 2, z_cell), base_seed),
        random_from_cell_3d(vec3<i32>(cell.x + 1, cell.y + 2, z_cell), base_seed),
        random_from_cell_3d(vec3<i32>(cell.x + 2, cell.y + 2, z_cell), base_seed),
        frac.x,
    );
    return blend_cubic(row0, row1, row2, row3, frac.y);
}

fn sample_raw_value_noise(
    uv : vec2<f32>,
    freq : vec2<f32>,
    base_seed : u32,
    time_value : f32,
    speed_value : f32,
    spline_order : u32,
) -> f32 {
    let scaled_freq : vec2<f32> = max(freq, vec2<f32>(1.0, 1.0));
    let scaled_uv : vec2<f32> = uv * scaled_freq;
    let cell_f : vec2<f32> = floor(scaled_uv);
    let cell : vec2<i32> = vec2<i32>(i32(cell_f.x), i32(cell_f.y));
    let frac : vec2<f32> = fract(scaled_uv);
    let angle : f32 = time_value * TAU;
    let time_coord : f32 = cos(angle) * speed_value;
    let time_floor : f32 = floor(time_coord);
    let time_cell : i32 = i32(time_floor);
    let time_frac : f32 = fract(time_coord);

    if (spline_order == INTERPOLATION_CONSTANT) {
        return random_from_cell_3d(vec3<i32>(cell.x, cell.y, time_cell), base_seed);
    }

    if (spline_order == INTERPOLATION_LINEAR) {
        let tl : f32 = random_from_cell_3d(vec3<i32>(cell.x, cell.y, time_cell), base_seed);
        let tr : f32 = random_from_cell_3d(vec3<i32>(cell.x + 1, cell.y, time_cell), base_seed);
        let bl : f32 = random_from_cell_3d(vec3<i32>(cell.x, cell.y + 1, time_cell), base_seed);
        let br : f32 = random_from_cell_3d(vec3<i32>(cell.x + 1, cell.y + 1, time_cell), base_seed);
        let weight_x : f32 = interpolation_weight(frac.x, spline_order);
        let top : f32 = mix(tl, tr, weight_x);
        let bottom : f32 = mix(bl, br, weight_x);
        let weight_y : f32 = interpolation_weight(frac.y, spline_order);
        return mix(top, bottom, weight_y);
    }

    if (spline_order == INTERPOLATION_COSINE) {
        let weight_x : f32 = interpolation_weight(frac.x, spline_order);
        let weight_y : f32 = interpolation_weight(frac.y, spline_order);
        let tl : f32 = random_from_cell_3d(vec3<i32>(cell.x, cell.y, time_cell), base_seed);
        let tr : f32 = random_from_cell_3d(vec3<i32>(cell.x + 1, cell.y, time_cell), base_seed);
        let bl : f32 = random_from_cell_3d(vec3<i32>(cell.x, cell.y + 1, time_cell), base_seed);
        let br : f32 = random_from_cell_3d(vec3<i32>(cell.x + 1, cell.y + 1, time_cell), base_seed);
        let top : f32 = mix(tl, tr, weight_x);
        let bottom : f32 = mix(bl, br, weight_x);
        return mix(top, bottom, weight_y);
    }

    let slice0 : f32 = sample_bicubic_layer(cell, frac, time_cell - 1, base_seed);
    let slice1 : f32 = sample_bicubic_layer(cell, frac, time_cell + 0, base_seed);
    let slice2 : f32 = sample_bicubic_layer(cell, frac, time_cell + 1, base_seed);
    let slice3 : f32 = sample_bicubic_layer(cell, frac, time_cell + 2, base_seed);
    return blend_cubic(slice0, slice1, slice2, slice3, time_frac);
}

fn sample_value_noise(
    uv : vec2<f32>,
    freq : vec2<f32>,
    seed : u32,
    time_value : f32,
    speed_value : f32,
    spline_order : u32,
) -> f32 {
    let base_seed : u32 = seed;
    let base_value : f32 = sample_raw_value_noise(
        uv,
        freq,
        base_seed,
        time_value,
        speed_value,
        spline_order,
    );

    if (speed_value == 0.0 || time_value == 0.0) {
        return base_value;
    }

    let time_seed : u32 = base_seed + 0x9e3779b1u;
    let time_field : f32 = sample_raw_value_noise(
        uv,
        freq,
        time_seed,
        0.0,
        1.0,
        spline_order,
    );
    let scaled_time : f32 = periodic_value(time_value, time_field) * speed_value;
    return periodic_value(scaled_time, base_value);
}

fn sample_grain_noise(
    pixel_coords : vec2<u32>,
    dims : vec2<f32>,
    time_value : f32,
    speed_value : f32,
) -> f32 {
    let width : f32 = max(dims.x, 1.0);
    let height : f32 = max(dims.y, 1.0);
    let uv : vec2<f32> = vec2<f32>(f32(pixel_coords.x) / width, f32(pixel_coords.y) / height);
    let freq : vec2<f32> = vec2<f32>(width, height);
    return sample_value_noise(uv, freq, BASE_SEED, time_value, speed_value, INTERPOLATION_BICUBIC);
}

fn write_pixel(base_index : u32, rgba : vec4<f32>) {
    output_buffer[base_index + 0u] = rgba.x;
    output_buffer[base_index + 1u] = rgba.y;
    output_buffer[base_index + 2u] = rgba.z;
    output_buffer[base_index + 3u] = rgba.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = max(as_u32(params.width), 1u);
    let height : u32 = max(as_u32(params.height), 1u);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let texel : vec4<f32> = textureLoad(input_texture, coords, 0);
    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;

    let blend_alpha : f32 = clamp(params.alpha, 0.0, 1.0);
    if (blend_alpha <= 0.0) {
        write_pixel(base_index, texel);
        return;
    }

    let noise_value : f32 = sample_grain_noise(
        gid.xy,
        vec2<f32>(f32(width), f32(height)),
        params.time,
        params.speed * 100.0,
    );
    let noise_rgba : vec4<f32> = vec4<f32>(noise_value, noise_value, noise_value, noise_value);
    let blend_weight : vec4<f32> = vec4<f32>(blend_alpha, blend_alpha, blend_alpha, blend_alpha);
    let mixed : vec4<f32> = mix(texel, noise_rgba, blend_weight);
    write_pixel(base_index, vec4<f32>(
        clamp01(mixed.x),
        clamp01(mixed.y),
        clamp01(mixed.z),
        clamp01(mixed.w),
    ));
}
