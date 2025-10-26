// Simplified multi-resolution generator compute shader.
//
// This version keeps the original binding model but now feeds its parameters
// through the `MultiresParams` struct, which groups related Python arguments
// into `vec4<f32>` packs for WGSL alignment.  The implementation focuses on the
// core octave stacking behaviour; procedural masks, lattice drift, permutation
// table indirection, and other advanced features are intentionally left out so
// the shader is easier to reason about and cheaper to execute while we continue
// iterating on the GPU path.

struct FrameUniforms {
    resolution : vec2<f32>,
    time : f32,
    seed : f32,
    frame_index : f32,
    padding0 : f32,
    padding1 : vec2<f32>,
};

struct MultiresParams {
    freq_octaves_ridges : vec4<f32>,
    sin_spline_distrib_corners : vec4<f32>,
    mask_options : vec4<f32>,
    supersample_color : vec4<f32>,
    saturation_hue : vec4<f32>,
    brightness_settings : vec4<f32>,
    ai_flags_time : vec4<f32>,
    speed_padding : vec4<f32>,
};

struct NormalizationState {
    min_value : atomic<u32>,
    max_value : atomic<u32>,
    count : atomic<u32>,
    phase : atomic<u32>,
};

struct SinNormalizationState {
    min_value : atomic<u32>,
    max_value : atomic<u32>,
    count : atomic<u32>,
    phase : atomic<u32>,
};

// Dummy structures for bindings that are still part of the pipeline layout.
struct MaskData {
    values : array<f32>,
};

struct PermutationTableStorage {
    values : array<u32>,
};

const TAU : f32 = 6.283185307179586;
const PI : f32 = 3.141592653589793;

const OCTAVE_BLENDING_FALLOFF : u32 = 0u;
const OCTAVE_BLENDING_REDUCE_MAX : u32 = 10u;
const OCTAVE_BLENDING_ALPHA : u32 = 20u;

const COLOR_SPACE_GRAYSCALE : u32 = 1u;
const COLOR_SPACE_RGB : u32 = 11u;
const COLOR_SPACE_HSV : u32 = 21u;
const COLOR_SPACE_OKLAB : u32 = 31u;

const DISTRIB_NONE : u32 = 0u;
const DISTRIB_SIMPLEX : u32 = 1u;
const DISTRIB_EXP : u32 = 2u;
const DISTRIB_ONES : u32 = 5u;
const DISTRIB_MIDS : u32 = 6u;
const DISTRIB_ZEROS : u32 = 7u;
const DISTRIB_COLUMN_INDEX : u32 = 10u;
const DISTRIB_ROW_INDEX : u32 = 11u;
const DISTRIB_CENTER_CIRCLE : u32 = 20u;
const DISTRIB_CENTER_DIAMOND : u32 = 21u;
const DISTRIB_CENTER_TRIANGLE : u32 = 23u;
const DISTRIB_CENTER_SQUARE : u32 = 24u;
const DISTRIB_CENTER_PENTAGON : u32 = 25u;
const DISTRIB_CENTER_HEXAGON : u32 = 26u;
const DISTRIB_CENTER_HEPTAGON : u32 = 27u;
const DISTRIB_CENTER_OCTAGON : u32 = 28u;
const DISTRIB_CENTER_NONAGON : u32 = 29u;
const DISTRIB_CENTER_DECAGON : u32 = 30u;
const DISTRIB_CENTER_HENDECAGON : u32 = 31u;
const DISTRIB_CENTER_DODECAGON : u32 = 32u;

const INTERPOLATION_CONSTANT : u32 = 0u;
const INTERPOLATION_LINEAR : u32 = 1u;
const INTERPOLATION_COSINE : u32 = 2u;
const INTERPOLATION_BICUBIC : u32 = 3u;

const FLOAT_SIGN_BIT : u32 = 0x80000000u;
const F32_MAX : f32 = 0x1.fffffep+127;
const UINT32_TO_FLOAT : f32 = 1.0 / 4294967296.0;

fn wrap_unit(value : f32) -> f32 {
    if (!(value == value)) {
        return 0.0;
    }
    // Proper modulo that handles negative values: ((value % 1.0) + 1.0) % 1.0
    let wrapped = value - floor(value);
    return wrapped;
}

fn bool_from_u32(value : u32) -> bool {
    return value != 0u;
}

fn bool_from_f32(value : f32) -> bool {
    return abs(value) >= 0.5;
}

fn clamp_to_u32(value : f32) -> u32 {
    let finite : f32 = select(0.0, value, value == value);
    return u32(max(round(finite), 0.0));
}

fn consume_u32(_value : u32) {
}

fn saturate(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn float_is_valid(value : f32) -> bool {
    return value == value && abs(value) <= F32_MAX;
}

fn float_to_ordered_uint(value : f32) -> u32 {
    let bits : u32 = bitcast<u32>(value);
    if ((bits & FLOAT_SIGN_BIT) != 0u) {
        return ~bits;
    }
    return bits | FLOAT_SIGN_BIT;
}

fn ridge_transform(value : f32) -> f32 {
    return 1.0 - abs(value * 2.0 - 1.0);
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

fn random_from_cell(cell : vec2<i32>, seed : u32) -> f32 {
    let packed : vec3<u32> = vec3<u32>(
        bitcast<u32>(cell.x),
        bitcast<u32>(cell.y),
        seed,
    );
    let noise : vec3<u32> = pcg3d(packed);
    return f32(noise.x) * UINT32_TO_FLOAT;
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

    if (spline_order == INTERPOLATION_LINEAR || spline_order == INTERPOLATION_COSINE) {
        let weight_x : f32 = interpolation_weight(frac.x, spline_order);
        let weight_y : f32 = interpolation_weight(frac.y, spline_order);
        let weight_z : f32 = interpolation_weight(time_frac, spline_order);
        let v000 : f32 = random_from_cell_3d_wrapped(vec3<i32>(cell.x, cell.y, time_cell), freq_i, base_seed);
        let v100 : f32 = random_from_cell_3d_wrapped(vec3<i32>(cell.x + 1, cell.y, time_cell), freq_i, base_seed);
        let v010 : f32 = random_from_cell_3d_wrapped(vec3<i32>(cell.x, cell.y + 1, time_cell), freq_i, base_seed);
        let v110 : f32 = random_from_cell_3d_wrapped(vec3<i32>(cell.x + 1, cell.y + 1, time_cell), freq_i, base_seed);
        let v001 : f32 = random_from_cell_3d_wrapped(vec3<i32>(cell.x, cell.y, time_cell + 1), freq_i, base_seed);
        let v101 : f32 = random_from_cell_3d_wrapped(vec3<i32>(cell.x + 1, cell.y, time_cell + 1), freq_i, base_seed);
        let v011 : f32 = random_from_cell_3d_wrapped(vec3<i32>(cell.x, cell.y + 1, time_cell + 1), freq_i, base_seed);
        let v111 : f32 = random_from_cell_3d_wrapped(vec3<i32>(cell.x + 1, cell.y + 1, time_cell + 1), freq_i, base_seed);

        let x00 : f32 = mix(v000, v100, weight_x);
        let x10 : f32 = mix(v010, v110, weight_x);
        let x01 : f32 = mix(v001, v101, weight_x);
        let x11 : f32 = mix(v011, v111, weight_x);
        let y0 : f32 = mix(x00, x10, weight_y);
        let y1 : f32 = mix(x01, x11, weight_y);
        return mix(y0, y1, weight_z);
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
    channel : u32,
    octave : u32,
    time_value : f32,
    speed : f32,
    spline_order : u32,
) -> f32 {
    let salt : u32 = (channel * 0x9e3779b9u) ^ (octave * 0x85ebca6bu);
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

fn wrap_pixel_coordinate(value : f32, limit : f32) -> f32 {
    if (limit <= 0.0) {
        return 0.0;
    }
    let normalized : f32 = value / limit;
    let wrapped : f32 = normalized - floor(normalized);
    return wrapped * limit;
}

fn apply_corner_offset(
    pixel : vec2<f32>,
    resolution : vec2<f32>,
    freq : vec2<f32>,
    corners_enabled : bool,
) -> vec2<f32> {
    let safe_freq : vec2<f32> = max(freq, vec2<f32>(1.0, 1.0));
    let freq_y_int : u32 = max(u32(floor(safe_freq.y)), 1u);
    let freq_even : bool = (freq_y_int & 1u) == 0u;
    let should_shift : bool = corners_enabled != freq_even;
    if (!should_shift) {
        return pixel;
    }
    let safe_resolution : vec2<f32> = max(resolution, vec2<f32>(1.0, 1.0));
    // Python shifts the image by +offset, so we need to sample from -offset
    let offset_pixels : vec2<f32> = floor((safe_resolution / safe_freq) * 0.5);
    let shifted : vec2<f32> = pixel - offset_pixels;
    return vec2<f32>(
        wrap_pixel_coordinate(shifted.x, safe_resolution.x),
        wrap_pixel_coordinate(shifted.y, safe_resolution.y),
    );
}

fn wrap_unit_vec(value : vec2<f32>) -> vec2<f32> {
    return vec2<f32>(wrap_unit(value.x), wrap_unit(value.y));
}

fn random_scalar_from_seed(seed : u32, time_value : f32, speed : f32) -> f32 {
    let hashed : vec3<u32> = pcg3d(vec3<u32>(
        seed ^ 0x352ef5cdu,
        seed ^ bitcast<u32>(time_value) ^ 0x9e3779b9u,
        seed ^ bitcast<u32>(speed) ^ 0x7f4a7c15u,
    ));
    return f32(hashed.x) * UINT32_TO_FLOAT;
}

fn apply_lattice_drift(
    uv : vec2<f32>,
    base_uv : vec2<f32>,
    freq : vec2<f32>,
    seed : u32,
    octave_index : u32,
    time_value : f32,
    speed : f32,
    spline_order : u32,
    amplitude : f32,
) -> vec2<f32> {
    if (amplitude == 0.0) {
        return uv;
    }
    let offset_seed : u32 = seed ^ 0x41c64e6du;
    let offset_x : f32 = sample_value_noise(
        base_uv,
        freq,
        offset_seed,
        3u,
        octave_index,
        time_value,
        speed,
        spline_order,
    );
    let offset_y : f32 = sample_value_noise(
        base_uv,
        freq,
        offset_seed,
        4u,
        octave_index,
        time_value,
        speed,
        spline_order,
    );
    // Center offsets to [-1, 1] range
    let centered : vec2<f32> = vec2<f32>(offset_x, offset_y) * 2.0 - vec2<f32>(1.0, 1.0);
    return wrap_unit_vec(uv + centered * amplitude);
}

fn regular_polygon_weight(centered : vec2<f32>, sides : f32) -> f32 {
    if (sides <= 0.0) {
        return 0.0;
    }
    let angle : f32 = atan2(centered.y, centered.x);
    let radius : f32 = length(centered);
    if (radius == 0.0) {
        return 1.0;
    }
    let sector : f32 = TAU / sides;
    let snapped : f32 = floor(0.5 + angle / sector) * sector;
    let distance : f32 = cos(snapped - angle) * radius;
    return saturate(1.0 - distance * 2.0);
}

fn center_distribution(distrib : u32, uv : vec2<f32>, aspect_ratio : f32) -> f32 {
    let center : vec2<f32> = vec2<f32>(0.5 * aspect_ratio, 0.5);
    let scaled : vec2<f32> = vec2<f32>(uv.x * aspect_ratio, uv.y) - center;
    let circle : f32 = saturate(1.0 - length(scaled) * 2.0);

    if (distrib == DISTRIB_CENTER_CIRCLE) {
        return circle;
    }
    if (distrib == DISTRIB_CENTER_DIAMOND) {
        let diamond : f32 = saturate(1.0 - (abs(scaled.x) + abs(scaled.y)) * 1.5);
        return diamond;
    }
    if (distrib == DISTRIB_CENTER_SQUARE) {
        let square : f32 = saturate(1.0 - max(abs(scaled.x), abs(scaled.y)) * 2.0);
        return square;
    }

    var sides : u32 = 0u;
    if (distrib >= DISTRIB_CENTER_TRIANGLE && distrib <= DISTRIB_CENTER_DODECAGON) {
        sides = 3u + (distrib - DISTRIB_CENTER_TRIANGLE);
    }
    if (sides >= 3u) {
        return regular_polygon_weight(scaled, f32(sides));
    }
    return circle;
}

fn apply_distribution(
    base_value : f32,
    distrib : u32,
    uv : vec2<f32>,
    shifted_uv : vec2<f32>,
    aspect_ratio : f32,
) -> f32 {
    if (distrib == DISTRIB_NONE || distrib == 0u) {
        return base_value;
    }
    if (distrib >= DISTRIB_CENTER_CIRCLE) {
        return center_distribution(distrib, uv, aspect_ratio);
    }
    let effective_uv : vec2<f32> = shifted_uv;
    switch (distrib) {
        case DISTRIB_SIMPLEX: {
            return base_value;
        }
        case DISTRIB_EXP: {
            return pow(base_value, 3.0);
        }
        case DISTRIB_ONES: {
            return 1.0;
        }
        case DISTRIB_MIDS: {
            return 0.5;
        }
        case DISTRIB_ZEROS: {
            return 0.0;
        }
        case DISTRIB_COLUMN_INDEX: {
            return saturate(effective_uv.x);
        }
        case DISTRIB_ROW_INDEX: {
            return saturate(effective_uv.y);
        }
        default: {
            return base_value;
        }
    }
}

fn sample_distribution_value(
    uv : vec2<f32>,
    shifted_uv : vec2<f32>,
    freq : vec2<f32>,
    seed : u32,
    distrib : u32,
    octave : u32,
    time_value : f32,
    speed : f32,
    spline_order : u32,
    aspect_ratio : f32,
) -> f32 {
    let noise_value : f32 = sample_value_noise(
        shifted_uv,
        freq,
        seed,
        0u,
        octave,
        time_value,
        speed,
        spline_order,
    );
    return apply_distribution(noise_value, distrib, uv, shifted_uv, aspect_ratio);
}

fn linear_to_srgb_component(value : f32) -> f32 {
    if (value <= 0.0031308) {
        return value * 12.92;
    }
    return 1.055 * pow(max(value, 0.0), 1.0 / 2.4) - 0.055;
}

fn linear_to_srgb(linear : vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        linear_to_srgb_component(linear.x),
        linear_to_srgb_component(linear.y),
        linear_to_srgb_component(linear.z),
    );
}

fn rgb_to_hsv(rgb : vec3<f32>) -> vec3<f32> {
    let cmax : f32 = max(max(rgb.x, rgb.y), rgb.z);
    let cmin : f32 = min(min(rgb.x, rgb.y), rgb.z);
    let delta : f32 = cmax - cmin;

    var hue : f32 = 0.0;
    if (delta > 0.0) {
        if (cmax == rgb.x) {
            hue = (rgb.y - rgb.z) / delta;
            if (hue < 0.0) {
                hue = hue + 6.0;
            }
        } else if (cmax == rgb.y) {
            hue = ((rgb.z - rgb.x) / delta) + 2.0;
        } else {
            hue = ((rgb.x - rgb.y) / delta) + 4.0;
        }
        hue = hue / 6.0;
    }

    var saturation : f32 = 0.0;
    if (cmax > 0.0) {
        saturation = delta / cmax;
    }

    return vec3<f32>(hue, saturation, cmax);
}

fn hsv_to_rgb(hsv : vec3<f32>) -> vec3<f32> {
    // Ensure hue is wrapped to [0, 1] range
    let hue : f32 = hsv.x - floor(hsv.x);
    let saturation : f32 = hsv.y;
    let value : f32 = hsv.z;

    let dh : f32 = hue * 6.0;
    let dr : f32 = clamp(abs(dh - 3.0) - 1.0, 0.0, 1.0);
    let dg : f32 = clamp(-abs(dh - 2.0) + 2.0, 0.0, 1.0);
    let db : f32 = clamp(-abs(dh - 4.0) + 2.0, 0.0, 1.0);

    let one_minus_s : f32 = 1.0 - saturation;
    let sr : f32 = saturation * dr;
    let sg : f32 = saturation * dg;
    let sb : f32 = saturation * db;

    let r : f32 = (one_minus_s + sr) * value;
    let g : f32 = (one_minus_s + sg) * value;
    let b : f32 = (one_minus_s + sb) * value;

    return vec3<f32>(r, g, b);
}

const OKLAB_FWD_A : mat3x3<f32> = mat3x3<f32>(
    vec3<f32>(1.0, 1.0, 1.0),
    vec3<f32>(0.3963377774, -0.1055613458, -0.0894841775),
    vec3<f32>(0.2158037573, -0.0638541728, -1.2914855480),
);

const OKLAB_FWD_B : mat3x3<f32> = mat3x3<f32>(
    vec3<f32>(4.0767245293, -1.2681437731, -0.0041119885),
    vec3<f32>(-3.3072168827, 2.6093323231, -0.7034763098),
    vec3<f32>(0.2307590544, -0.3411344290, 1.7068625689),
);

fn oklab_to_srgb(lab : vec3<f32>) -> vec3<f32> {
    let lms : vec3<f32> = OKLAB_FWD_A * lab;
    let cubic : vec3<f32> = vec3<f32>(lms.x * lms.x * lms.x, lms.y * lms.y * lms.y, lms.z * lms.z * lms.z);
    return linear_to_srgb(OKLAB_FWD_B * cubic);
}

fn compute_octave_frequency(base_freq : vec2<f32>, octave_index : u32) -> vec2<f32> {
    let multiplier : f32 = pow(2.0, f32(octave_index));
    return base_freq * multiplier;
}

fn combine_alpha(base_color : vec4<f32>, layer : vec4<f32>) -> vec4<f32> {
    let alpha_vec : vec4<f32> = vec4<f32>(layer.w, layer.w, layer.w, layer.w);
    return base_color * (vec4<f32>(1.0, 1.0, 1.0, 1.0) - alpha_vec) + layer * alpha_vec;
}

fn default_mask_value() -> vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}

fn sample_mask_value(
    octave_index : u32,
    pixel_index : u32,
    width : u32,
    height : u32,
    value_count : u32,
) -> vec4<f32> {
    if (width == 0u || height == 0u) {
        return default_mask_value();
    }
    let pixel_count : u32 = width * height;
    if (pixel_count == 0u) {
        return default_mask_value();
    }
    let stride : u32 = 4u;
    let pixels_per_octave : u32 = pixel_count * stride;
    if (pixels_per_octave == 0u || value_count < stride) {
        return default_mask_value();
    }
    let available_octaves : u32 = value_count / pixels_per_octave;
    if (available_octaves == 0u || octave_index >= available_octaves) {
        return default_mask_value();
    }
    let base_index : u32 = ((octave_index * pixel_count) + pixel_index) * stride;
    if (base_index + 3u >= value_count) {
        return default_mask_value();
    }
    return vec4<f32>(
        mask_data.values[base_index + 0u],
        mask_data.values[base_index + 1u],
        mask_data.values[base_index + 2u],
        mask_data.values[base_index + 3u],
    );
}

fn update_normalization(sample : vec4<f32>, with_alpha : bool, state : ptr<storage, NormalizationState, read_write>) {
    var min_value : f32 = 0.0;
    var max_value : f32 = 0.0;
    var has_valid_sample : bool = false;

    let components : array<f32, 4> = array<f32, 4>(sample.x, sample.y, sample.z, sample.w);
    let limit : u32 = select(3u, 4u, with_alpha);

    for (var i : u32 = 0u; i < limit; i = i + 1u) {
        let value : f32 = components[i];
        if (!float_is_valid(value)) {
            continue;
        }
        if (!has_valid_sample) {
            min_value = value;
            max_value = value;
            has_valid_sample = true;
        } else {
            min_value = min(min_value, value);
            max_value = max(max_value, value);
        }
    }

    if (!has_valid_sample) {
        return;
    }

    atomicMin(&(*state).min_value, float_to_ordered_uint(min_value));
    atomicMax(&(*state).max_value, float_to_ordered_uint(max_value));
}

@group(0) @binding(0) var<uniform> params : MultiresParams;
@group(0) @binding(1) var<uniform> frame_uniforms : FrameUniforms;
@group(0) @binding(3) var output_texture : texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var<storage, read_write> normalization_state : NormalizationState;
@group(0) @binding(5) var<storage, read_write> sin_normalization_state : SinNormalizationState;
@group(0) @binding(6) var<storage, read> mask_data : MaskData;
@group(0) @binding(7) var<storage, read> permutation_table_storage : PermutationTableStorage;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let width : u32 = max(u32(round(frame_uniforms.resolution.x)), 1u);
    let height : u32 = max(u32(round(frame_uniforms.resolution.y)), 1u);
    if (global_id.x >= width || global_id.y >= height) {
        return;
    }

    // Touch optional bindings so the compiler keeps them alive when the stage
    // descriptor still provides the associated buffers.
    consume_u32(atomicLoad(&sin_normalization_state.phase));
    let mask_value_count : u32 = arrayLength(&mask_data.values);
    consume_u32(mask_value_count);
    consume_u32(arrayLength(&permutation_table_storage.values));

    let resolution_vec : vec2<f32> = vec2<f32>(frame_uniforms.resolution.x, frame_uniforms.resolution.y);
    let aspect_ratio : f32 = select(1.0, resolution_vec.x / max(resolution_vec.y, 1.0), resolution_vec.y != 0.0);
    let pixel_raw : vec2<f32> = vec2<f32>(f32(global_id.x) + 0.5, f32(global_id.y) + 0.5);

    let freq_octaves_ridges : vec4<f32> = params.freq_octaves_ridges;
    let base_freq : vec2<f32> = max(freq_octaves_ridges.xy, vec2<f32>(1.0, 1.0));
    let octaves : u32 = max(clamp_to_u32(freq_octaves_ridges.z), 1u);
    let ridges_enabled : bool = bool_from_f32(freq_octaves_ridges.w);

    let sin_spline_distrib_corners : vec4<f32> = params.sin_spline_distrib_corners;
    let sin_amount : f32 = sin_spline_distrib_corners.x;
    let spline_order : u32 = min(clamp_to_u32(sin_spline_distrib_corners.y), INTERPOLATION_BICUBIC);
    let distrib : u32 = clamp_to_u32(sin_spline_distrib_corners.z);
    let corners_enabled : bool = bool_from_f32(sin_spline_distrib_corners.w);

    // Apply corner offset if enabled
    let pixel : vec2<f32> = apply_corner_offset(pixel_raw, resolution_vec, base_freq, corners_enabled);
    let uv : vec2<f32> = pixel / resolution_vec;

    let mask_options : vec4<f32> = params.mask_options;
    let mask_enabled : bool = bool_from_f32(mask_options.x);
    let mask_inverse : bool = bool_from_f32(mask_options.y);
    let mask_static : bool = bool_from_f32(mask_options.z);
    let lattice_drift : f32 = mask_options.w;

    let supersample_color : vec4<f32> = params.supersample_color;
    let with_supersample : bool = bool_from_f32(supersample_color.x);
    let raw_color_space : u32 = clamp_to_u32(supersample_color.y);
    var color_space : u32 = raw_color_space;
    let color_space_valid : bool =
        raw_color_space == COLOR_SPACE_GRAYSCALE ||
        raw_color_space == COLOR_SPACE_RGB ||
        raw_color_space == COLOR_SPACE_HSV ||
        raw_color_space == COLOR_SPACE_OKLAB;
    if (!color_space_valid) {
        color_space = COLOR_SPACE_HSV;
    }
    var hue_range_value : f32 = supersample_color.z;
    var hue_rotation_param : f32 = supersample_color.w;

    let saturation_hue : vec4<f32> = params.saturation_hue;
    let saturation_scale : f32 = saturation_hue.x;
    let hue_distrib : u32 = clamp_to_u32(saturation_hue.y);
    let saturation_distrib : u32 = clamp_to_u32(saturation_hue.z);
    let brightness_distrib : u32 = clamp_to_u32(saturation_hue.w);

    let brightness_settings : vec4<f32> = params.brightness_settings;
    let brightness_freq_override : vec2<f32> = brightness_settings.xy;
    let octave_blending : u32 = clamp_to_u32(brightness_settings.z);
    let with_alpha_output : bool = bool_from_f32(brightness_settings.w);

    let ai_flags_time : vec4<f32> = params.ai_flags_time;
    let with_ai : bool = bool_from_f32(ai_flags_time.x);
    let with_upscale : bool = bool_from_f32(ai_flags_time.y);
    let with_fxaa : bool = bool_from_f32(ai_flags_time.z);
    let stage_time : f32 = ai_flags_time.w;
    let time_value : f32 = frame_uniforms.time + stage_time;

    let speed_padding : vec4<f32> = params.speed_padding;
    let raw_speed : f32 = speed_padding.x;
    var speed : f32 = clamp(raw_speed, 0.0, 2.0);
    if (!(speed == speed)) {
        speed = 0.1;
    }
    let base_seed : u32 = clamp_to_u32(frame_uniforms.seed);
    var hue_rotation_value : f32 = hue_rotation_param;
    if (color_space != COLOR_SPACE_HSV) {
        hue_range_value = 1.0;
        hue_rotation_value = 0.0;
    }
    let hue_rotation : f32 = hue_rotation_value;
    let hue_range : f32 = hue_range_value;
    if (mask_static || with_supersample || with_ai || with_upscale || with_fxaa) {
        // Parameters reserved for CPU parity; no GPU-side action yet.
    }
    let pixel_index : u32 = global_id.y * width + global_id.x;

    var accum : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    for (var octave_index : u32 = 0u; octave_index < octaves; octave_index = octave_index + 1u) {
        let octave_freq : vec2<f32> = compute_octave_frequency(base_freq, octave_index);
        let octave_seed : u32 = base_seed ^ (octave_index * 0x9e3779b9u + 0x7f4a7c15u);
        let safe_freq_min : f32 = max(min(octave_freq.x, octave_freq.y), 1.0);
        let drift_amplitude : f32 = select(0.0, lattice_drift / safe_freq_min, lattice_drift != 0.0);
        let warped_uv : vec2<f32> = apply_lattice_drift(uv, uv, octave_freq, octave_seed, octave_index, time_value, speed, spline_order, drift_amplitude);

        let freq_exceeds_width : bool = octave_freq.x > frame_uniforms.resolution.x;
        let freq_exceeds_height : bool = octave_freq.y > frame_uniforms.resolution.y;
        if (freq_exceeds_width && freq_exceeds_height) {
            break;
        }

        // Center distance distributions apply the same SDF value to all channels
        let is_center_distrib : bool = distrib >= DISTRIB_CENTER_CIRCLE;
        
        var c0 : f32;
        var c1 : f32;
        var c2 : f32;
        
        if (is_center_distrib) {
            // For center distance distributions, use the same SDF value for all RGB channels
            let sdf_value : f32 = center_distribution(distrib, uv, aspect_ratio);
            c0 = sdf_value;
            c1 = sdf_value;
            c2 = sdf_value;
        } else {
            // For noise distributions, sample per-channel noise
            c0 = sample_value_noise(
                    warped_uv,
                octave_freq,
                octave_seed,
                0u,
                octave_index,
                time_value,
                speed,
                spline_order,
            );
            c1 = sample_value_noise(
                    warped_uv,
                octave_freq,
                octave_seed,
                1u,
                octave_index,
                time_value,
                speed,
                spline_order,
            );
            c2 = sample_value_noise(
                    warped_uv,
                octave_freq,
                octave_seed,
                2u,
                octave_index,
                time_value,
                speed,
                spline_order,
            );
            // Apply non-noise distributions (exp, ones, mids, zeros, index) to noise values
            c0 = apply_distribution(c0, distrib, uv, warped_uv, aspect_ratio);
            c1 = apply_distribution(c1, distrib, uv, warped_uv, aspect_ratio);
            c2 = apply_distribution(c2, distrib, uv, warped_uv, aspect_ratio);
        }
        
        // Generate independent alpha channel noise when with_alpha is enabled
        let c3 : f32 = sample_value_noise(
                warped_uv,
            octave_freq,
            octave_seed,
            3u,
            octave_index,
            time_value,
            speed,
            spline_order,
        );
        var layer_color : vec3<f32> = vec3<f32>(c0, c1, c2);

        var mask_value : vec4<f32> = vec4<f32>(1.0, 1.0, 1.0, 1.0);
        if (mask_enabled) {
            mask_value = sample_mask_value(
                octave_index,
                pixel_index,
                width,
                height,
                mask_value_count,
            );
            if (mask_inverse) {
                mask_value = vec4<f32>(1.0, 1.0, 1.0, 1.0) - mask_value;
            }
            layer_color = layer_color * mask_value.xyz;
        }

        let override_seed : u32 = octave_seed ^ 0x94d049b4u;
        var hue_value : f32;
        var saturation_value : f32;
        if (hue_distrib != 0u) {
            hue_value = sample_distribution_value(
                uv,
                    warped_uv,
                octave_freq,
                override_seed ^ 0x1u,
                hue_distrib,
                octave_index,
                time_value,
                speed,
                spline_order,
                aspect_ratio,
            );
        } else {
            hue_value = wrap_unit(layer_color.x * hue_range + hue_rotation);
        }
        if (saturation_distrib != 0u) {
            saturation_value = sample_distribution_value(
                uv,
                    warped_uv,
                octave_freq,
                override_seed ^ 0x2u,
                saturation_distrib,
                octave_index,
                time_value,
                speed,
                spline_order,
                aspect_ratio,
            );
        } else {
            saturation_value = layer_color.y;
        }
        saturation_value = saturation_value * saturation_scale;
        let brightness_freq_has_override : bool = brightness_freq_override.x > 0.0 || brightness_freq_override.y > 0.0;
        let brightness_noise_required : bool = brightness_distrib != 0u && brightness_distrib != DISTRIB_NONE;
        var brightness_value : f32;
        if (brightness_noise_required) {
            var brightness_freq_vec : vec2<f32> = octave_freq;
            if (brightness_freq_has_override) {
                brightness_freq_vec = max(brightness_freq_override, vec2<f32>(1.0, 1.0));
            }
            brightness_value = sample_distribution_value(
                uv,
                warped_uv,
                brightness_freq_vec,
                override_seed ^ 0x3u,
                brightness_distrib,
                octave_index,
                time_value,
                speed,
                spline_order,
                aspect_ratio,
            );
        } else {
            brightness_value = apply_distribution(layer_color.z, distrib, uv, warped_uv, aspect_ratio);
        }

        var rgb_color : vec3<f32>;
        if (color_space == COLOR_SPACE_GRAYSCALE) {
            if (ridges_enabled) {
                brightness_value = ridge_transform(brightness_value);
            }
            if (sin_amount != 0.0) {
                brightness_value = sin(brightness_value * sin_amount) * 0.5 + 0.5;
            }
            brightness_value = clamp(brightness_value, 0.0, 1.0);
            rgb_color = vec3<f32>(brightness_value, brightness_value, brightness_value);
        } else if (color_space == COLOR_SPACE_RGB) {
            var ridged_color = layer_color;
            if (ridges_enabled) {
                ridged_color = vec3<f32>(
                    ridge_transform(layer_color.x),
                    ridge_transform(layer_color.y),
                    ridge_transform(layer_color.z)
                );
            }
            // Convert to HSV to apply hue rotation and saturation, then back to RGB
            let hsv = rgb_to_hsv(ridged_color);
            let rotated_hue = wrap_unit(hsv.x * hue_range + hue_rotation);
            let adjusted_hsv = vec3<f32>(rotated_hue, hsv.y * saturation_scale, hsv.z);
            rgb_color = clamp(hsv_to_rgb(adjusted_hsv), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));
        } else if (color_space == COLOR_SPACE_OKLAB) {
            var ridged_color = layer_color;
            if (ridges_enabled) {
                ridged_color = vec3<f32>(
                    ridge_transform(layer_color.x),
                    ridge_transform(layer_color.y),
                    ridge_transform(layer_color.z)
                );
            }
            let oklab_color : vec3<f32> = vec3<f32>(
                ridged_color.x,
                ridged_color.y * -0.509 + 0.276,
                ridged_color.z * -0.509 + 0.198,
            );
            let rgb = oklab_to_srgb(oklab_color);
            // Convert to HSV to apply hue rotation and saturation, then back to RGB
            let hsv = rgb_to_hsv(rgb);
            let rotated_hue = wrap_unit(hsv.x * hue_range + hue_rotation);
            let adjusted_hsv = vec3<f32>(rotated_hue, hsv.y * saturation_scale, hsv.z);
            rgb_color = clamp(hsv_to_rgb(adjusted_hsv), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));
        } else {
            if (ridges_enabled) {
                brightness_value = ridge_transform(brightness_value);
            }
            if (sin_amount != 0.0) {
                brightness_value = sin(brightness_value * sin_amount) * 0.5 + 0.5;
            }
            brightness_value = clamp(brightness_value, 0.0, 1.0);
            rgb_color = hsv_to_rgb(vec3<f32>(hue_value, saturation_value, brightness_value));
        }

        // Alpha channel handling:
        // - If with_alpha_output is true, use independent c3 noise value (with distribution applied)
        // - Otherwise, use brightness_value for octave blending
        var layer_alpha : f32;
        if (with_alpha_output) {
            // Apply distribution to alpha channel just like Python does in value.values()
            layer_alpha = apply_distribution(c3, distrib, uv, warped_uv, aspect_ratio);
        } else {
            layer_alpha = brightness_value;
        }
        
        if (mask_enabled) {
            layer_alpha = layer_alpha * mask_value.w;
        }

        var layer_rgba : vec4<f32> = vec4<f32>(rgb_color, layer_alpha);

        if (octave_blending == OCTAVE_BLENDING_REDUCE_MAX) {
            accum = max(accum, layer_rgba);
        } else if (octave_blending == OCTAVE_BLENDING_ALPHA) {
            accum = combine_alpha(accum, layer_rgba);
        } else {
            let weight : f32 = pow(0.5, f32(octave_index + 1u));
            accum = accum + layer_rgba * weight;
        }
    }

    var final_color : vec4<f32> = accum;
    if (!with_alpha_output) {
        final_color.w = 1.0;
    } else {
        // Premultiply RGB by alpha for proper blending
        final_color = vec4<f32>(
            final_color.rgb * final_color.a,
            final_color.a
        );
    }

    final_color = clamp(final_color, vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(1.0, 1.0, 1.0, 1.0));

    update_normalization(final_color, with_alpha_output, &normalization_state);

    textureStore(
        output_texture,
        vec2<i32>(i32(global_id.x), i32(global_id.y)),
        final_color,
    );
}

