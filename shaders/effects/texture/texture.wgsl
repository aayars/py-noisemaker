// Texture effect: applies animated ridged noise-based shading similar to
// Noisemaker's CPU implementation. The shader generates ridged value noise,
// derives a Sobel-based shadow map, sharpens it, and modulates the source
// texture brightness. Parameters mirror the Python reference: time and speed.
//
// Performance optimizations:
// - Uses 4 octaves instead of Python's 8 (acceptable quality/speed tradeoff)
// - 3x3 window instead of 5x5 (Sobel only needs 3x3)
// - Effect strength increased to 0.7-1.0 range for visibility (Python: 0.9-1.0)

const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647693;
const UINT32_TO_FLOAT : f32 = 1.0 / 4294967296.0;

const INTERPOLATION_CONSTANT : u32 = 0u;
const INTERPOLATION_LINEAR : u32 = 1u;
const INTERPOLATION_COSINE : u32 = 2u;
const INTERPOLATION_BICUBIC : u32 = 3u;

struct TextureParams {
    width : f32,           // @offset(0)
    height : f32,          // @offset(4)
    channel_count : f32,   // @offset(8)
    _pad0 : f32,           // @offset(12)
    time : f32,            // @offset(16)
    speed : f32,           // @offset(20)
    _pad1 : f32,           // @offset(24)
    _pad2 : f32,           // @offset(28)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : TextureParams;

const SOBEL_X : array<f32, 9> = array<f32, 9>(
    0.5, 0.0, -0.5,
    1.0, 0.0, -1.0,
    0.5, 0.0, -0.5
);

const SOBEL_Y : array<f32, 9> = array<f32, 9>(
    0.5, 1.0, 0.5,
    0.0, 0.0, 0.0,
    -0.5, -1.0, -0.5
);

const SHARPEN_KERNEL : array<f32, 9> = array<f32, 9>(
    0.0, -0.2, 0.0,
    -0.2, 1.0, -0.2,
    0.0, -0.2, 0.0
);

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn wrap_unit(value : f32) -> f32 {
    if (!(value == value)) {
        return 0.0;
    }
    // Proper modulo that handles negative values: ((value % 1.0) + 1.0) % 1.0
    let mod_val : f32 = value - floor(value);
    return mod_val;
}

fn wrap_coord(value : i32, size : i32) -> i32 {
    if (size <= 0) {
        return 0;
    }
    var wrapped : i32 = value % size;
    if (wrapped < 0) {
        wrapped = wrapped + size;
    }
    return wrapped;
}

fn freq_for_shape(base_freq : f32, dims : vec2<f32>) -> vec2<f32> {
    let width : f32 = max(dims.x, 1.0);
    let height : f32 = max(dims.y, 1.0);
    if (abs(height - width) < 0.5) {
        return vec2<f32>(base_freq, base_freq);
    }
    if (height < width) {
        return vec2<f32>(base_freq, base_freq * width / height);
    }
    return vec2<f32>(base_freq * height / width, base_freq);
}

fn pcg3d(v_in : vec3<u32>) -> vec3<u32> {
    var v : vec3<u32> = v_in * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
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

fn wrap_cell_2d(cell : vec2<i32>, freq : vec2<i32>) -> vec2<i32> {
    // Modulo wrapping for seamless tiling
    var wrapped : vec2<i32>;
    wrapped.x = cell.x % freq.x;
    if (wrapped.x < 0) { wrapped.x += freq.x; }
    wrapped.y = cell.y % freq.y;
    if (wrapped.y < 0) { wrapped.y += freq.y; }
    return wrapped;
}

fn random_from_cell_3d_wrapped(cell : vec3<i32>, freq : vec2<i32>, seed : u32) -> f32 {
    let wrapped_xy : vec2<i32> = wrap_cell_2d(vec2<i32>(cell.x, cell.y), freq);
    return random_from_cell_3d(vec3<i32>(wrapped_xy.x, wrapped_xy.y, cell.z), seed);
}

fn periodic_value(time_value : f32, sample : f32) -> f32 {
    let wrapped_delta : f32 = wrap_unit(time_value - sample);
    return (sin(wrapped_delta * TAU) + 1.0) * 0.5;
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
    freq : vec2<i32>,
    base_seed : u32,
) -> f32 {
    let row0 : f32 = blend_cubic(
        random_from_cell_3d_wrapped(vec3<i32>(cell.x - 1, cell.y - 1, z_cell), freq, base_seed),
        random_from_cell_3d_wrapped(vec3<i32>(cell.x + 0, cell.y - 1, z_cell), freq, base_seed),
        random_from_cell_3d_wrapped(vec3<i32>(cell.x + 1, cell.y - 1, z_cell), freq, base_seed),
        random_from_cell_3d_wrapped(vec3<i32>(cell.x + 2, cell.y - 1, z_cell), freq, base_seed),
        frac.x,
    );
    let row1 : f32 = blend_cubic(
        random_from_cell_3d_wrapped(vec3<i32>(cell.x - 1, cell.y + 0, z_cell), freq, base_seed),
        random_from_cell_3d_wrapped(vec3<i32>(cell.x + 0, cell.y + 0, z_cell), freq, base_seed),
        random_from_cell_3d_wrapped(vec3<i32>(cell.x + 1, cell.y + 0, z_cell), freq, base_seed),
        random_from_cell_3d_wrapped(vec3<i32>(cell.x + 2, cell.y + 0, z_cell), freq, base_seed),
        frac.x,
    );
    let row2 : f32 = blend_cubic(
        random_from_cell_3d_wrapped(vec3<i32>(cell.x - 1, cell.y + 1, z_cell), freq, base_seed),
        random_from_cell_3d_wrapped(vec3<i32>(cell.x + 0, cell.y + 1, z_cell), freq, base_seed),
        random_from_cell_3d_wrapped(vec3<i32>(cell.x + 1, cell.y + 1, z_cell), freq, base_seed),
        random_from_cell_3d_wrapped(vec3<i32>(cell.x + 2, cell.y + 1, z_cell), freq, base_seed),
        frac.x,
    );
    let row3 : f32 = blend_cubic(
        random_from_cell_3d_wrapped(vec3<i32>(cell.x - 1, cell.y + 2, z_cell), freq, base_seed),
        random_from_cell_3d_wrapped(vec3<i32>(cell.x + 0, cell.y + 2, z_cell), freq, base_seed),
        random_from_cell_3d_wrapped(vec3<i32>(cell.x + 1, cell.y + 2, z_cell), freq, base_seed),
        random_from_cell_3d_wrapped(vec3<i32>(cell.x + 2, cell.y + 2, z_cell), freq, base_seed),
        frac.x,
    );
    return blend_cubic(row0, row1, row2, row3, frac.y);
}

fn sample_raw_value_noise(
    uv : vec2<f32>,
    freq : vec2<f32>,
    base_seed : u32,
    time_value : f32,
    speed : f32,
    spline_order : u32,
) -> f32 {
    let scaled_freq : vec2<f32> = max(freq, vec2<f32>(1.0, 1.0));
    let scaled_uv : vec2<f32> = uv * scaled_freq;
    let cell_f : vec2<f32> = floor(scaled_uv);
    let freq_i : vec2<i32> = vec2<i32>(i32(floor(scaled_freq.x)), i32(floor(scaled_freq.y)));
    // Wrap cell coordinates modulo frequency for seamless tiling
    let cell : vec2<i32> = vec2<i32>(
        i32(cell_f.x) % freq_i.x,
        i32(cell_f.y) % freq_i.y
    );
    let frac : vec2<f32> = fract(scaled_uv);
    let normalized_time : f32 = wrap_unit(time_value);
    let angle : f32 = normalized_time * TAU;
    let time_coord : f32 = cos(angle) * speed;
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

    let slice0 : f32 = sample_bicubic_layer(cell, frac, time_cell - 1, freq_i, base_seed);
    let slice1 : f32 = sample_bicubic_layer(cell, frac, time_cell + 0, freq_i, base_seed);
    let slice2 : f32 = sample_bicubic_layer(cell, frac, time_cell + 1, freq_i, base_seed);
    let slice3 : f32 = sample_bicubic_layer(cell, frac, time_cell + 2, freq_i, base_seed);
    return blend_cubic(slice0, slice1, slice2, slice3, time_frac);
}

fn sample_value_noise(
    uv : vec2<f32>,
    freq : vec2<f32>,
    seed : u32,
    octave : u32,
    time_value : f32,
    speed : f32,
    spline_order : u32,
) -> f32 {
    let salt : u32 = (octave * 0x85ebca6bu);
    let base_seed : u32 = seed ^ salt;
    let normalized_time : f32 = wrap_unit(time_value);
    let base_value : f32 = sample_raw_value_noise(
        uv,
        freq,
        base_seed,
        normalized_time,
        speed,
        spline_order,
    );

    if (speed == 0.0 || normalized_time == 0.0) {
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
    let scaled_time : f32 = periodic_value(normalized_time, time_field) * speed;
    return periodic_value(scaled_time, base_value);
}

fn ridge_transform(value : f32) -> f32 {
    return 1.0 - abs(value * 2.0 - 1.0);
}

fn simple_multires(
    uv : vec2<f32>,
    base_freq : vec2<f32>,
    time_value : f32,
    speed_value : f32,
) -> f32 {
    var freq : vec2<f32> = base_freq;
    var amplitude : f32 = 0.5;
    var total : f32 = 0.0;
    var accum : f32 = 0.0;
    let seed : u32 = 0x1234u;
    // Reduced from 8 to 4 octaves for performance
    for (var octave : u32 = 0u; octave < 4u; octave = octave + 1u) {
        let sample : f32 = sample_value_noise(
            uv,
            freq,
            seed,
            octave,
            time_value,
            speed_value,
            INTERPOLATION_BICUBIC,
        );
        let ridged : f32 = ridge_transform(sample);
        accum = accum + ridged * amplitude;
        total = total + amplitude;
        freq = freq * 2.0;
        amplitude = amplitude * 0.5;
    }
    if (total > 0.0) {
        accum = accum / total;
    }
    return clamp01(accum);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.width);
    let height : u32 = as_u32(params.height);
    if (width == 0u || height == 0u) {
        return;
    }
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let base_color : vec4<f32> = textureLoad(input_texture, coords, 0);
    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * 4u;

    let dims : vec2<f32> = vec2<f32>(max(params.width, 1.0), max(params.height, 1.0));
    let width_i32 : i32 = i32(width);
    let height_i32 : i32 = i32(height);
    let time_value : f32 = params.time;
    let speed_value : f32 = params.speed;
    let base_freq : vec2<f32> = freq_for_shape(64.0, dims);

    // Reduced from 5x5 to 3x3 window for performance (Sobel only needs 3x3)
    var noise_window : array<f32, 9>;
    var write_index : i32 = 0;
    for (var offset_y : i32 = -1; offset_y <= 1; offset_y = offset_y + 1) {
        for (var offset_x : i32 = -1; offset_x <= 1; offset_x = offset_x + 1) {
            let sample_x : i32 = wrap_coord(coords.x + offset_x, width_i32);
            let sample_y : i32 = wrap_coord(coords.y + offset_y, height_i32);
            let uv : vec2<f32> = (vec2<f32>(f32(sample_x), f32(sample_y)) + 0.5) / dims;
            let noise_value : f32 = simple_multires(uv, base_freq, time_value, speed_value);
            noise_window[u32(write_index)] = noise_value;
            write_index = write_index + 1;
        }
    }

    // Compute Sobel gradient directly on the 3x3 window
    var gx : f32 = 0.0;
    var gy : f32 = 0.0;
    for (var i : u32 = 0u; i < 9u; i = i + 1u) {
        gx = gx + noise_window[i] * SOBEL_X[i];
        gy = gy + noise_window[i] * SOBEL_Y[i];
    }
    let gradient : f32 = sqrt(gx * gx + gy * gy);
    let shade : f32 = clamp01(gradient * 0.5);
    
    // Apply sharpening kernel
    var sharpened : f32 = 0.0;
    for (var i : u32 = 0u; i < 9u; i = i + 1u) {
        sharpened = sharpened + shade * SHARPEN_KERNEL[i];
    }
    let final_shade_value : f32 = mix(shade, clamp01(sharpened), 0.5);
    
    let highlight : f32 = final_shade_value * final_shade_value;
    let base_noise : f32 = noise_window[4u];  // Center pixel
    let composite : f32 = 1.0 - ((1.0 - base_noise) * (1.0 - highlight));
    let final_shade : f32 = composite * final_shade_value;
    // Increased from 0.9 + 0.1 to 0.7 + 0.3 for more visible effect
    let factor : f32 = 0.7 + final_shade * 0.3;
    let scaled_rgb : vec3<f32> = clamp(
        base_color.xyz * vec3<f32>(factor, factor, factor),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );

    output_buffer[base_index + 0u] = scaled_rgb.x;
    output_buffer[base_index + 1u] = scaled_rgb.y;
    output_buffer[base_index + 2u] = scaled_rgb.z;
    output_buffer[base_index + 3u] = base_color.w;
}
