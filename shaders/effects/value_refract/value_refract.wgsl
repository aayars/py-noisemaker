// Value Refract: replicates Noisemaker's value_refract effect by generating a value
// distribution map and using it as the refractive driver for the source texture.
const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;

struct ValueRefractParams {
    size_freq : vec4<f32>, // (width, height, channels, freq)
    displacement_time_speed_distrib : vec4<f32>, // (displacement, time, speed, distrib)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ValueRefractParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp_01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
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

fn wrap_float(value : f32, limit : f32) -> f32 {
    if (limit == 0.0) {
        return 0.0;
    }
    let div : f32 = floor(value / limit);
    var result : f32 = value - div * limit;
    if (result < 0.0) {
        result = result + limit;
    }
    return result;
}

fn freq_for_shape(base_freq : f32, width : f32, height : f32) -> vec2<f32> {
    let safe_freq : f32 = max(base_freq, 1.0);
    let safe_width : f32 = max(width, 1.0);
    let safe_height : f32 = max(height, 1.0);

    if (abs(safe_width - safe_height) < 1e-5) {
        return vec2<f32>(safe_freq, safe_freq);
    }

    if (safe_height < safe_width) {
        let second : f32 = floor(safe_freq * safe_width / safe_height);
        return vec2<f32>(safe_freq, max(second, 1.0));
    }

    let first : f32 = floor(safe_freq * safe_height / safe_width);
    return vec2<f32>(max(first, 1.0), safe_freq);
}

fn normalized_sine(value : f32) -> f32 {
    return sin(value) * 0.5 + 0.5;
}

fn periodic_value(time : f32, value : f32) -> f32 {
    return normalized_sine((time - value) * TAU);
}

fn rounded_speed(speed : f32) -> f32 {
    if (speed > 0.0) {
        return floor(1.0 + speed);
    }
    return ceil(-1.0 + speed);
}

fn mod289_vec3(x : vec3<f32>) -> vec3<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn mod289_vec4(x : vec4<f32>) -> vec4<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn permute_vec4(x : vec4<f32>) -> vec4<f32> {
    return mod289_vec4(((x * 34.0) + 1.0) * x);
}

fn taylor_inv_sqrt(r : vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * r;
}

fn simplex_noise(v : vec3<f32>) -> f32 {
    let C : vec2<f32> = vec2<f32>(1.0 / 6.0, 1.0 / 3.0);
    let D : vec4<f32> = vec4<f32>(0.0, 0.5, 1.0, 2.0);

    let i0 : vec3<f32> = floor(v + dot(v, vec3<f32>(C.y)));
    let x0 : vec3<f32> = v - i0 + dot(i0, vec3<f32>(C.x));

    let step1 : vec3<f32> = step(vec3<f32>(x0.y, x0.z, x0.x), x0);
    let l : vec3<f32> = vec3<f32>(1.0) - step1;
    let i1 : vec3<f32> = min(step1, vec3<f32>(l.z, l.x, l.y));
    let i2 : vec3<f32> = max(step1, vec3<f32>(l.z, l.x, l.y));

    let x1 : vec3<f32> = x0 - i1 + vec3<f32>(C.x);
    let x2 : vec3<f32> = x0 - i2 + vec3<f32>(C.y);
    let x3 : vec3<f32> = x0 - vec3<f32>(D.y);

    let i = mod289_vec3(i0);
    let p = permute_vec4(
        permute_vec4(permute_vec4(i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0)) + i.y
            + vec4<f32>(0.0, i1.y, i2.y, 1.0))
        + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0)
    );

    let n_ : f32 = 0.14285714285714285;
    let ns : vec3<f32> = n_ * vec3<f32>(D.w, D.y, D.z) - vec3<f32>(D.x, D.z, D.x);

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

    let m0_sq : f32 = m0 * m0;
    let m1_sq : f32 = m1 * m1;
    let m2_sq : f32 = m2 * m2;
    let m3_sq : f32 = m3 * m3;

    return 42.0 * (
        m0_sq * m0_sq * dot(g0n, x0)
        + m1_sq * m1_sq * dot(g1n, x1)
        + m2_sq * m2_sq * dot(g2n, x2)
        + m3_sq * m3_sq * dot(g3n, x3)
    );
}

struct DistributionInfo {
    metric : i32,
    sdf_sides : f32,
};

fn distribution_info(distribution : i32) -> DistributionInfo {
    // Defaults to center circle (euclidean) if unknown.
    var metric : i32 = 1;
    var sdf_sides : f32 = 0.0;

    switch distribution {
        case 20: { // center_circle
            metric = 1;
        }
        case 21: { // center_diamond
            metric = 2;
        }
        case 23: { // center_triangle
            metric = 101;
        }
        case 24: { // center_square
            metric = 3;
        }
        case 25: { // center_pentagon
            metric = 201;
            sdf_sides = 5.0;
        }
        case 26: { // center_hexagon
            metric = 102;
        }
        case 27: { // center_heptagon
            metric = 201;
            sdf_sides = 7.0;
        }
        case 28: { // center_octagon
            metric = 4;
        }
        case 29: { // center_nonagon
            metric = 201;
            sdf_sides = 9.0;
        }
        case 30: { // center_decagon
            metric = 201;
            sdf_sides = 10.0;
        }
        case 31: { // center_hendecagon
            metric = 201;
            sdf_sides = 11.0;
        }
        case 32: { // center_dodecagon
            metric = 201;
            sdf_sides = 12.0;
        }
        default: {
            // leave defaults
        }
    }

    return DistributionInfo(metric, sdf_sides);
}

fn compute_distance(a : f32, b : f32, metric : i32, sdf_sides : f32) -> f32 {
    switch metric {
        case 1: { // euclidean
            return sqrt(a * a + b * b);
        }
        case 2: { // manhattan
            return abs(a) + abs(b);
        }
        case 3: { // chebyshev
            return max(abs(a), abs(b));
        }
        case 4: { // octagram
            let combo : f32 = (abs(a) + abs(b)) / sqrt(2.0);
            return max(combo, max(abs(a), abs(b)));
        }
        case 101: { // triangular
            return max(abs(a) - b * 0.5, b);
        }
        case 102: { // hexagram
            let pos : f32 = max(abs(a) - b * 0.5, b);
            let neg : f32 = max(abs(a) - b * -0.5, b * -1.0);
            return max(pos, neg);
        }
        case 201: { // sdf polygon
            if (sdf_sides <= 0.0) {
                return sqrt(a * a + b * b);
            }
            let angle : f32 = atan2(a, -b) + PI;
            let r : f32 = TAU / sdf_sides;
            let k : f32 = floor(0.5 + angle / r);
            let diff : f32 = k * r - angle;
            return cos(diff) * sqrt(a * a + b * b);
        }
        default: {
            return sqrt(a * a + b * b);
        }
    }
}

fn compute_center_value(
    coord : vec2<u32>,
    width : f32,
    height : f32,
    freq : vec2<f32>,
    distribution : i32,
    time : f32,
    speed_phase : f32
) -> f32 {
    let info : DistributionInfo = distribution_info(distribution);
    let width_safe : f32 = max(width, 1.0);
    let height_safe : f32 = max(height, 1.0);
    let center : vec2<f32> = vec2<f32>(width_safe * 0.5, height_safe * 0.5);
    let pos : vec2<f32> = vec2<f32>(f32(coord.x) + 0.5, f32(coord.y) + 0.5);
    let offset : vec2<f32> = vec2<f32>(pos.x - center.x, pos.y - center.y);
    let norm : vec2<f32> = vec2<f32>(offset.x / width_safe, offset.y / height_safe);

    let dist : f32 = compute_distance(norm.x, norm.y, info.metric, info.sdf_sides);
    let freq_scale : f32 = max(freq.x, freq.y);
    let angle : f32 = dist * TAU * freq_scale - TAU * time * speed_phase;
    return clamp_01(normalized_sine(angle));
}

fn compute_noise_value(
    coord : vec2<u32>,
    width : f32,
    height : f32,
    freq : vec2<f32>,
    time : f32,
    speed : f32,
    is_exp : bool
) -> f32 {
    let width_safe : f32 = max(width, 1.0);
    let height_safe : f32 = max(height, 1.0);
    let uv : vec2<f32> = vec2<f32>(
        (f32(coord.x) / width_safe) * max(freq.x, 1.0),
        (f32(coord.y) / height_safe) * max(freq.y, 1.0)
    );

    let angle : f32 = time * TAU;
    let z_base : f32 = cos(angle) * speed;
    let base_seed : vec3<f32> = vec3<f32>(17.0, 29.0, 47.0);
    let base_noise : f32 = simplex_noise(vec3<f32>(
        uv.x + base_seed.x,
        uv.y + base_seed.y,
        z_base + base_seed.z
    ));
    var value : f32 = clamp(base_noise * 0.5 + 0.5, 0.0, 1.0);

    if (speed != 0.0 && time != 0.0) {
        let time_seed : vec3<f32> = vec3<f32>(71.0, 113.0, 191.0);
        let time_noise : f32 = simplex_noise(vec3<f32>(
            uv.x + time_seed.x,
            uv.y + time_seed.y,
            time_seed.z
        ));
        let time_value : f32 = clamp(time_noise * 0.5 + 0.5, 0.0, 1.0);
        let scaled_time : f32 = periodic_value(time, time_value) * speed;
        value = clamp_01(periodic_value(scaled_time, value));
    }

    if (is_exp) {
        value = pow(value, 4.0);
    }

    return clamp_01(value);
}

fn generate_distribution_value(
    coord : vec2<u32>,
    width : f32,
    height : f32,
    freq : vec2<f32>,
    distribution : i32,
    time : f32,
    speed : f32,
    center_phase : f32
) -> f32 {
    if (distribution >= 20 && distribution < 40) {
        return compute_center_value(coord, width, height, freq, distribution, time, center_phase);
    }

    switch distribution {
        case 5: { // ones
            return 1.0;
        }
        case 6: { // mids
            return 0.5;
        }
        case 7: { // zeros
            return 0.0;
        }
        case 10: { // column_index
            return clamp_01((f32(coord.x) + 0.5) / max(width, 1.0));
        }
        case 11: { // row_index
            return clamp_01((f32(coord.y) + 0.5) / max(height, 1.0));
        }
        default: {
            let is_exp : bool = distribution == 2;
            return compute_noise_value(coord, width, height, freq, time, speed, is_exp);
        }
    }
}

fn sample_texture_bilinear(x : f32, y : f32, width : u32, height : u32) -> vec4<f32> {
    let width_f : f32 = f32(width);
    let height_f : f32 = f32(height);

    let wrapped_x : f32 = wrap_float(x, width_f);
    let wrapped_y : f32 = wrap_float(y, height_f);

    var x0 : i32 = i32(floor(wrapped_x));
    var y0 : i32 = i32(floor(wrapped_y));

    let width_i : i32 = i32(width);
    let height_i : i32 = i32(height);

    if (x0 < 0) {
        x0 = 0;
    } else if (x0 >= width_i) {
        x0 = width_i - 1;
    }

    if (y0 < 0) {
        y0 = 0;
    } else if (y0 >= height_i) {
        y0 = height_i - 1;
    }

    let x1 : i32 = wrap_coord(x0 + 1, width_i);
    let y1 : i32 = wrap_coord(y0 + 1, height_i);

    let fx : f32 = clamp(wrapped_x - f32(x0), 0.0, 1.0);
    let fy : f32 = clamp(wrapped_y - f32(y0), 0.0, 1.0);

    let tex00 : vec4<f32> = textureLoad(input_texture, vec2<i32>(x0, y0), 0);
    let tex10 : vec4<f32> = textureLoad(input_texture, vec2<i32>(x1, y0), 0);
    let tex01 : vec4<f32> = textureLoad(input_texture, vec2<i32>(x0, y1), 0);
    let tex11 : vec4<f32> = textureLoad(input_texture, vec2<i32>(x1, y1), 0);

    let mix_x0 : vec4<f32> = mix(tex00, tex10, vec4<f32>(fx));
    let mix_x1 : vec4<f32> = mix(tex01, tex11, vec4<f32>(fx));
    return mix(mix_x0, mix_x1, vec4<f32>(fy));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width_u : u32 = max(as_u32(params.size_freq.x), 1u);
    let height_u : u32 = max(as_u32(params.size_freq.y), 1u);
    if (gid.x >= width_u || gid.y >= height_u) {
        return;
    }

    let width_f : f32 = params.size_freq.x;
    let height_f : f32 = params.size_freq.y;
    let channel_count : u32 = max(as_u32(params.size_freq.z), 1u);
    let freq_param : f32 = params.size_freq.w;

    let displacement : f32 = params.displacement_time_speed_distrib.x;
    let time : f32 = params.displacement_time_speed_distrib.y;
    let speed : f32 = params.displacement_time_speed_distrib.z;
    let distribution_id : i32 = i32(round(params.displacement_time_speed_distrib.w));

    let freq_vec : vec2<f32> = freq_for_shape(freq_param, width_f, height_f);
    let phase_speed : f32 = rounded_speed(speed);
    let value_sample : f32 = generate_distribution_value(
        gid.xy,
        width_f,
        height_f,
        freq_vec,
        distribution_id,
        time,
        speed,
        phase_speed
    );

    let angle_value : f32 = value_sample * TAU;
    let ref_x : f32 = clamp_01(cos(angle_value) * 0.5 + 0.5);
    let ref_y : f32 = clamp_01(sin(angle_value) * 0.5 + 0.5);

    let offset_x : f32 = (ref_x * 2.0 - 1.0) * displacement * width_f;
    let offset_y : f32 = (ref_y * 2.0 - 1.0) * displacement * height_f;

    let sample_x : f32 = f32(gid.x) + offset_x;
    let sample_y : f32 = f32(gid.y) + offset_y;

    let sampled : vec4<f32> = sample_texture_bilinear(sample_x, sample_y, width_u, height_u);

    let base_index : u32 = (gid.y * width_u + gid.x) * channel_count;
    var channel : u32 = 0u;
    loop {
        if (channel >= channel_count) {
            break;
        }
        let component : f32 = sampled[min(channel, 3u)];
        output_buffer[base_index + channel] = component;
        channel = channel + 1u;
    }
}
