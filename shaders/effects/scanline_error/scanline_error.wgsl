// Scanline error glitch effect.
//
// Mirrors ``effects.scanline_error`` in ``noisemaker/effects.py``. Animated
// exponential noise bands modulate both a horizontal displacement field and an
// additive white-noise overlay. The displacement uses the normalized value-map
// of the combined error signal as in the Python reference, approximated here by
// directly mixing the error contributions. The shader writes RGBA results to the
// output buffer with deterministic wrapping semantics for the displaced lookup.

const TAU : f32 = 6.283185307179586;

struct ScanlineErrorParams {
    size : vec4<f32>,       // (width, height, channels, unused)
    time_speed : vec4<f32>, // (time, speed, unused, unused)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ScanlineErrorParams;

fn as_u32(value : f32) -> u32 {
    if (value <= 0.0) {
        return 0u;
    }
    return u32(value + 0.5);
}

fn clamp_01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn sanitize_channel_count(value : f32) -> u32 {
    let rounded : i32 = i32(round(value));
    if (rounded <= 1) {
        return 1u;
    }
    if (rounded >= 4) {
        return 4u;
    }
    return u32(rounded);
}

fn wrap_coord(coord : i32, limit : i32) -> i32 {
    if (limit <= 0) {
        return 0;
    }
    var wrapped : i32 = coord % limit;
    if (wrapped < 0) {
        wrapped = wrapped + limit;
    }
    return wrapped;
}

fn read_channel(value : vec4<f32>, channel : u32) -> f32 {
    switch channel {
        case 0u: { return value.x; }
        case 1u: { return value.y; }
        case 2u: { return value.z; }
        default: { return value.w; }
    }
}

fn mod289_vec3(x : vec3<f32>) -> vec3<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn mod289_vec4(x : vec4<f32>) -> vec4<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn permute(x : vec4<f32>) -> vec4<f32> {
    return mod289_vec4(((x * 34.0) + 1.0) * x);
}

fn taylor_inv_sqrt(r : vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * r;
}

fn simplex_noise(v : vec3<f32>) -> f32 {
    let c : vec2<f32> = vec2<f32>(1.0 / 6.0, 1.0 / 3.0);
    let d : vec4<f32> = vec4<f32>(0.0, 0.5, 1.0, 2.0);

    let i0 : vec3<f32> = floor(v + dot(v, vec3<f32>(c.y)));
    let x0 : vec3<f32> = v - i0 + dot(i0, vec3<f32>(c.x));

    let step1 : vec3<f32> = step(vec3<f32>(x0.y, x0.z, x0.x), x0);
    let l : vec3<f32> = vec3<f32>(1.0) - step1;
    let i1 : vec3<f32> = min(step1, vec3<f32>(l.z, l.x, l.y));
    let i2 : vec3<f32> = max(step1, vec3<f32>(l.z, l.x, l.y));

    let x1 : vec3<f32> = x0 - i1 + vec3<f32>(c.x);
    let x2 : vec3<f32> = x0 - i2 + vec3<f32>(c.y);
    let x3 : vec3<f32> = x0 - vec3<f32>(d.y);

    let i = mod289_vec3(i0);
    let p = permute(permute(permute(
        i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0))
        + i.y + vec4<f32>(0.0, i1.y, i2.y, 1.0))
        + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0));

    let n_ : f32 = 0.14285714285714285;
    let ns : vec3<f32> = n_ * vec3<f32>(d.w, d.y, d.z) - vec3<f32>(d.x, d.z, d.x);

    let j : vec4<f32> = p - 49.0 * floor(p * ns.z * ns.z);
    let x_ : vec4<f32> = floor(j * ns.z);
    let y_ : vec4<f32> = floor(j - 7.0 * x_);

    let x = x_ * ns.x + ns.y;
    let y = y_ * ns.x + ns.y;
    let h = 1.0 - abs(x) - abs(y);

    let b0 : vec4<f32> = vec4<f32>(x.x, x.y, y.x, y.y);
    let b1 : vec4<f32> = vec4<f32>(x.z, x.w, y.z, y.w);

    let s0 : vec4<f32> = floor(b0) * 2.0 + 1.0;
    let s1 : vec4<f32> = floor(b1) * 2.0 + 1.0;
    let sh : vec4<f32> = -step(h, vec4<f32>(0.0));

    let a0 : vec4<f32> = vec4<f32>(b0.x, b0.z, b0.y, b0.w)
        + vec4<f32>(s0.x, s0.z, s0.y, s0.w) * vec4<f32>(sh.x, sh.x, sh.y, sh.y);
    let a1 : vec4<f32> = vec4<f32>(b1.x, b1.z, b1.y, b1.w)
        + vec4<f32>(s1.x, s1.z, s1.y, s1.w) * vec4<f32>(sh.z, sh.z, sh.w, sh.w);

    let g0 : vec3<f32> = vec3<f32>(a0.x, a0.y, h.x);
    let g1 : vec3<f32> = vec3<f32>(a0.z, a0.w, h.y);
    let g2 : vec3<f32> = vec3<f32>(a1.x, a1.y, h.z);
    let g3 : vec3<f32> = vec3<f32>(a1.z, a1.w, h.w);

    let norm : vec4<f32> = taylor_inv_sqrt(vec4<f32>(
        dot(g0, g0),
        dot(g1, g1),
        dot(g2, g2),
        dot(g3, g3)
    ));

    let g0n : vec3<f32> = g0 * norm.x;
    let g1n : vec3<f32> = g1 * norm.y;
    let g2n : vec3<f32> = g2 * norm.z;
    let g3n : vec3<f32> = g3 * norm.w;

    let m0 : f32 = max(0.6 - dot(x0, x0), 0.0);
    let m1 : f32 = max(0.6 - dot(x1, x1), 0.0);
    let m2 : f32 = max(0.6 - dot(x2, x2), 0.0);
    let m3 : f32 = max(0.6 - dot(x3, x3), 0.0);

    let m0sq : f32 = m0 * m0;
    let m1sq : f32 = m1 * m1;
    let m2sq : f32 = m2 * m2;
    let m3sq : f32 = m3 * m3;

    return 42.0 * (
        m0sq * m0sq * dot(g0n, x0)
        + m1sq * m1sq * dot(g1n, x1)
        + m2sq * m2sq * dot(g2n, x2)
        + m3sq * m3sq * dot(g3n, x3)
    );
}

fn periodic_value(time : f32, value : f32) -> f32 {
    return sin((time - value) * TAU) * 0.5 + 0.5;
}

fn normalized_coord(coord : vec2<u32>, dims : vec2<f32>) -> vec2<f32> {
    let width_safe : f32 = max(dims.x, 1.0);
    let height_safe : f32 = max(dims.y, 1.0);
    return vec2<f32>(
        (f32(coord.x) + 0.5) / width_safe,
        (f32(coord.y) + 0.5) / height_safe
    );
}

fn compute_simplex_value(
    coord : vec2<f32>,
    freq : vec2<f32>,
    time : f32,
    speed : f32,
    offset : vec3<f32>
) -> f32 {
    let freq_x : f32 = max(freq.x, 1.0);
    let freq_y : f32 = max(freq.y, 1.0);
    let angle : f32 = cos(time * TAU) * speed;
    let sample : vec3<f32> = vec3<f32>(
        coord.x * freq_x + offset.x,
        coord.y * freq_y + offset.y,
        angle + offset.z
    );
    return simplex_noise(sample);
}

fn compute_value_noise(
    coord : vec2<f32>,
    freq : vec2<f32>,
    time : f32,
    speed : f32,
    base_seed : vec3<f32>,
    time_seed : vec3<f32>
) -> f32 {
    let base_noise : f32 = compute_simplex_value(coord, freq, time, speed, base_seed);
    var value : f32 = clamp_01(base_noise * 0.5 + 0.5);

    if (speed != 0.0 && time != 0.0) {
        let time_noise_raw : f32 = compute_simplex_value(coord, freq, 0.0, 1.0, time_seed);
        let time_value : f32 = clamp_01(time_noise_raw * 0.5 + 0.5);
        let scaled_time : f32 = periodic_value(time, time_value) * speed;
        value = periodic_value(scaled_time, value);
    }

    return clamp_01(value);
}

fn compute_exponential_noise(
    coord : vec2<f32>,
    freq : vec2<f32>,
    time : f32,
    speed : f32,
    base_seed : vec3<f32>,
    time_seed : vec3<f32>
) -> f32 {
    let base : f32 = compute_value_noise(coord, freq, time, speed, base_seed, time_seed);
    return pow(base, 4.0);
}

const BASE_SEED_LINE : vec3<f32> = vec3<f32>(37.0, 91.0, 53.0);
const TIME_SEED_LINE : vec3<f32> = vec3<f32>(
    BASE_SEED_LINE.x + 97.0,
    BASE_SEED_LINE.y + 59.0,
    BASE_SEED_LINE.z + 131.0
);
const BASE_SEED_SWERVE : vec3<f32> = vec3<f32>(11.0, 73.0, 29.0);
const TIME_SEED_SWERVE : vec3<f32> = vec3<f32>(
    BASE_SEED_SWERVE.x + 89.0,
    BASE_SEED_SWERVE.y + 41.0,
    BASE_SEED_SWERVE.z + 149.0
);
const BASE_SEED_WHITE : vec3<f32> = vec3<f32>(67.0, 29.0, 149.0);
const TIME_SEED_WHITE : vec3<f32> = vec3<f32>(
    BASE_SEED_WHITE.x + 113.0,
    BASE_SEED_WHITE.y + 53.0,
    BASE_SEED_WHITE.z + 173.0
);

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = max(as_u32(params.size.x), 1u);
    let height : u32 = max(as_u32(params.size.y), 1u);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let channel_count : u32 = sanitize_channel_count(params.size.z);
    let width_f : f32 = f32(width);
    let height_f : f32 = f32(height);
    let dims : vec2<f32> = vec2<f32>(width_f, height_f);

    let coord_norm : vec2<f32> = normalized_coord(gid.xy, dims);
    let freq_line : vec2<f32> = vec2<f32>(
        max(floor(width_f * 0.5), 1.0),
        max(floor(height_f * 0.5), 1.0)
    );
    let swerve_height : f32 = max(floor(height_f * 0.01), 1.0);
    let freq_swerve : vec2<f32> = vec2<f32>(1.0, swerve_height);
    let swerve_coord : vec2<f32> = vec2<f32>(0.0, coord_norm.y);

    let time_value : f32 = params.time_speed.x;
    let speed_value : f32 = params.time_speed.y;

    var line_noise : f32 = compute_exponential_noise(
        coord_norm,
        freq_line,
        time_value,
        speed_value * 10.0,
        BASE_SEED_LINE,
        TIME_SEED_LINE
    );
    line_noise = max(line_noise - 0.5, 0.0);

    var swerve_noise : f32 = compute_exponential_noise(
        swerve_coord,
        freq_swerve,
        time_value,
        speed_value,
        BASE_SEED_SWERVE,
        TIME_SEED_SWERVE
    );
    swerve_noise = max(swerve_noise - 0.5, 0.0);

    let line_weighted : f32 = line_noise * swerve_noise;
    let swerve_weight : f32 = swerve_noise * 2.0;

    let white_base : f32 = compute_value_noise(
        coord_norm,
        freq_line,
        time_value,
        speed_value * 100.0,
        BASE_SEED_WHITE,
        TIME_SEED_WHITE
    );
    let white_weighted : f32 = white_base * swerve_weight;

    let combined_error : f32 = clamp_01(line_weighted + white_weighted);
    let shift_amount : f32 = combined_error * width_f * 0.025;
    let shift_pixels : i32 = i32(floor(shift_amount));
    let sample_x : i32 = wrap_coord(i32(gid.x) - shift_pixels, i32(width));

    let texel : vec4<f32> = textureLoad(input_texture, vec2<i32>(sample_x, i32(gid.y)), 0);
    let additive : f32 = clamp(line_weighted * white_weighted * 4.0, 0.0, 4.0);

    let base_index : u32 = (gid.y * width + gid.x) * channel_count;
    var channel : u32 = 0u;
    loop {
        if (channel >= channel_count) {
            break;
        }

        let original : f32 = read_channel(texel, min(channel, 3u));
        if (channel == 3u) {
            output_buffer[base_index + channel] = original;
        } else {
            output_buffer[base_index + channel] = clamp_01(original + additive);
        }

        channel = channel + 1u;
    }
}
