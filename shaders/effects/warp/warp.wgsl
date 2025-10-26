// Warp effect: multi-octave displacement using simplex noise or a supplied warp map.
// Mirrors noisemaker.effects.warp, emitting offsets that refract the input texture.

const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;

struct WarpParams {
    dims_freq : vec4<f32>,             // (width, height, channels, freq)
    octave_disp_spline_map : vec4<f32>,  // (octaves, displacement, spline_order, warp_map_flag)
    signed_time_speed_pad : vec4<f32>,   // (signed_range, time, speed, padding)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : WarpParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(value, 0.0));
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
    if (limit <= 0.0) {
        return 0.0;
    }
    var result : f32 = value - floor(value / limit) * limit;
    if (result < 0.0) {
        result = result + limit;
    }
    return result;
}

fn srgb_to_linear(value : f32) -> f32 {
    if (value <= 0.04045) {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn cube_root(value : f32) -> f32 {
    if (value == 0.0) {
        return 0.0;
    }
    let sign_value : f32 = select(-1.0, 1.0, value >= 0.0);
    return sign_value * pow(abs(value), 1.0 / 3.0);
}

fn oklab_l_component(rgb : vec3<f32>) -> f32 {
    let r : f32 = srgb_to_linear(clamp_01(rgb.x));
    let g : f32 = srgb_to_linear(clamp_01(rgb.y));
    let b : f32 = srgb_to_linear(clamp_01(rgb.z));

    let l : f32 = 0.4121656120 * r + 0.5362752080 * g + 0.0514575653 * b;
    let m : f32 = 0.2118591070 * r + 0.6807189584 * g + 0.1074065790 * b;
    let s : f32 = 0.0883097947 * r + 0.2818474174 * g + 0.6302613616 * b;

    let l_c : f32 = cube_root(l);
    let m_c : f32 = cube_root(m);
    let s_c : f32 = cube_root(s);

    return clamp_01(0.2104542553 * l_c + 0.7936177850 * m_c - 0.0040720468 * s_c);
}

fn value_map_from_texel(texel : vec4<f32>, channel_count : u32) -> f32 {
    if (channel_count <= 2u) {
        return clamp_01(texel.x);
    }
    return oklab_l_component(vec3<f32>(texel.x, texel.y, texel.z));
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

fn normalized_sine(value : f32) -> f32 {
    return sin(value) * 0.5 + 0.5;
}

fn periodic_value(time : f32, value : f32) -> f32 {
    return normalized_sine((time - value) * TAU);
}

fn mod_289_vec3(x : vec3<f32>) -> vec3<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn mod_289_vec4(x : vec4<f32>) -> vec4<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn permute(x : vec4<f32>) -> vec4<f32> {
    return mod_289_vec4(((x * 34.0) + 1.0) * x);
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

    let i = mod_289_vec3(i0);
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
        m0sq * m0sq * dot(g0n, x0) +
        m1sq * m1sq * dot(g1n, x1) +
        m2sq * m2sq * dot(g2n, x2) +
        m3sq * m3sq * dot(g3n, x3)
    );
}

fn simplex_value(
    coord : vec2<u32>,
    width : f32,
    height : f32,
    freq : vec2<f32>,
    time : f32,
    speed : f32,
    base_seed : vec3<f32>,
    time_seed : vec3<f32>
) -> f32 {
    let width_safe : f32 = max(width, 1.0);
    let height_safe : f32 = max(height, 1.0);
    let sample : vec2<f32> = vec2<f32>(
        (f32(coord.x) / width_safe) * max(freq.x, 1.0),
        (f32(coord.y) / height_safe) * max(freq.y, 1.0)
    );

    let angle : f32 = time * TAU;
    let z_base : f32 = cos(angle) * speed;
    let base_noise : f32 = simplex_noise(vec3<f32>(
        sample.x + base_seed.x,
        sample.y + base_seed.y,
        z_base + base_seed.z
    ));
    var value : f32 = clamp(base_noise * 0.5 + 0.5, 0.0, 1.0);

    if (speed != 0.0 && time != 0.0) {
        let time_noise : f32 = simplex_noise(vec3<f32>(
            sample.x + time_seed.x,
            sample.y + time_seed.y,
            time_seed.z
        ));
        let time_value : f32 = clamp(time_noise * 0.5 + 0.5, 0.0, 1.0);
        let scaled_time : f32 = periodic_value(time, time_value) * speed;
        value = clamp_01(periodic_value(scaled_time, value));
    }

    return clamp_01(value);
}

fn compute_warp_reference(
    coord : vec2<u32>,
    width : f32,
    height : f32,
    freq : vec2<f32>,
    time : f32,
    speed : f32,
    warp_map_enabled : bool,
    channel_count : u32
) -> vec2<f32> {
    if (warp_map_enabled) {
        let texel : vec4<f32> = textureLoad(
            input_texture,
            vec2<i32>(i32(coord.x), i32(coord.y)),
            0
        );
        let map_value : f32 = value_map_from_texel(texel, channel_count);
        let angle : f32 = map_value * TAU;
        let ref_x : f32 = clamp_01(cos(angle) * 0.5 + 0.5);
        let ref_y : f32 = clamp_01(sin(angle) * 0.5 + 0.5);
        return vec2<f32>(ref_x, ref_y);
    }

    let base_seed_x : vec3<f32> = vec3<f32>(17.0, 29.0, 47.0);
    let time_seed_x : vec3<f32> = vec3<f32>(71.0, 113.0, 191.0);
    let base_seed_y : vec3<f32> = vec3<f32>(23.0, 31.0, 53.0);
    let time_seed_y : vec3<f32> = vec3<f32>(79.0, 131.0, 197.0);

    let ref_x : f32 = simplex_value(
        coord,
        width,
        height,
        freq,
        time,
        speed,
        base_seed_x,
        time_seed_x
    );
    let ref_y : f32 = simplex_value(
        coord,
        width,
        height,
        freq,
        time,
        speed,
        base_seed_y,
        time_seed_y
    );
    return vec2<f32>(ref_x, ref_y);
}

fn displacement_offset(
    reference : vec2<f32>,
    signed_range : bool,
    displacement : f32,
    width : f32,
    height : f32
) -> vec2<f32> {
    var ref_vec : vec2<f32> = vec2<f32>(clamp_01(reference.x), clamp_01(reference.y));
    if (signed_range) {
        ref_vec = ref_vec * 2.0 - vec2<f32>(1.0, 1.0);
    }

    var offset : vec2<f32> = vec2<f32>(ref_vec.x * displacement * width, ref_vec.y * displacement * height);
    if (!signed_range) {
        offset = offset * 2.0;
    }
    return offset;
}

fn apply_spline(value : f32, order : i32) -> f32 {
    let clamped : f32 = clamp(value, 0.0, 1.0);
    if (order == 2) {
        return 0.5 - cos(clamped * PI) * 0.5;
    }
    return clamped;
}

fn cubic_interpolate(
    a : vec4<f32>,
    b : vec4<f32>,
    c : vec4<f32>,
    d : vec4<f32>,
    t : f32
) -> vec4<f32> {
    let t2 : f32 = t * t;
    let t3 : f32 = t2 * t;
    let a0 : vec4<f32> = d - c - a + b;
    let a1 : vec4<f32> = a - b - a0;
    let a2 : vec4<f32> = c - a;
    let a3 : vec4<f32> = b;
    return a0 * t3 + a1 * t2 + a2 * t + a3;
}

fn sample_nearest(coord : vec2<f32>, width : i32, height : i32) -> vec4<f32> {
    let x : i32 = wrap_coord(i32(round(coord.x)), width);
    let y : i32 = wrap_coord(i32(round(coord.y)), height);
    return textureLoad(input_texture, vec2<i32>(x, y), 0);
}

fn sample_bilinear(coord : vec2<f32>, width : i32, height : i32, order : i32) -> vec4<f32> {
    var x0 : i32 = i32(floor(coord.x));
    var y0 : i32 = i32(floor(coord.y));

    if (x0 < 0) {
        x0 = 0;
    } else if (x0 >= width) {
        x0 = width - 1;
    }

    if (y0 < 0) {
        y0 = 0;
    } else if (y0 >= height) {
        y0 = height - 1;
    }

    let x1 : i32 = wrap_coord(x0 + 1, width);
    let y1 : i32 = wrap_coord(y0 + 1, height);

    let fx : f32 = clamp(coord.x - f32(x0), 0.0, 1.0);
    let fy : f32 = clamp(coord.y - f32(y0), 0.0, 1.0);

    let tx : f32 = apply_spline(fx, order);
    let ty : f32 = apply_spline(fy, order);

    let tex00 : vec4<f32> = textureLoad(input_texture, vec2<i32>(x0, y0), 0);
    let tex10 : vec4<f32> = textureLoad(input_texture, vec2<i32>(x1, y0), 0);
    let tex01 : vec4<f32> = textureLoad(input_texture, vec2<i32>(x0, y1), 0);
    let tex11 : vec4<f32> = textureLoad(input_texture, vec2<i32>(x1, y1), 0);

    let mix_x0 : vec4<f32> = mix(tex00, tex10, vec4<f32>(tx));
    let mix_x1 : vec4<f32> = mix(tex01, tex11, vec4<f32>(tx));
    return mix(mix_x0, mix_x1, vec4<f32>(ty));
}

fn sample_bicubic(coord : vec2<f32>, width : i32, height : i32) -> vec4<f32> {
    let base_x : i32 = i32(floor(coord.x));
    let base_y : i32 = i32(floor(coord.y));

    var columns : array<vec4<f32>, 4>;
    var m : i32 = -1;
    loop {
        if (m >= 3) {
            break;
        }
        var row : array<vec4<f32>, 4>;
        var n : i32 = -1;
        loop {
            if (n >= 3) {
                break;
            }
            let sx : i32 = wrap_coord(base_x + n, width);
            let sy : i32 = wrap_coord(base_y + m, height);
            row[n + 1] = textureLoad(input_texture, vec2<i32>(sx, sy), 0);
            n = n + 1;
        }
        columns[m + 1] = cubic_interpolate(
            row[0],
            row[1],
            row[2],
            row[3],
            clamp(coord.x - floor(coord.x), 0.0, 1.0)
        );
        m = m + 1;
    }

    let frac_y : f32 = clamp(coord.y - floor(coord.y), 0.0, 1.0);
    return cubic_interpolate(columns[0], columns[1], columns[2], columns[3], frac_y);
}

fn sample_with_order(coord : vec2<f32>, width : u32, height : u32, order : i32) -> vec4<f32> {
    let width_f : f32 = f32(width);
    let height_f : f32 = f32(height);
    let wrapped : vec2<f32> = vec2<f32>(
        wrap_float(coord.x, width_f),
        wrap_float(coord.y, height_f)
    );
    let width_i : i32 = i32(width);
    let height_i : i32 = i32(height);

    if (order <= 0) {
        return sample_nearest(wrapped, width_i, height_i);
    }
    if (order >= 3) {
        return sample_bicubic(wrapped, width_i, height_i);
    }
    return sample_bilinear(wrapped, width_i, height_i, order);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.dims_freq.x);
    let height : u32 = as_u32(params.dims_freq.y);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let width_f : f32 = max(params.dims_freq.x, 1.0);
    let height_f : f32 = max(params.dims_freq.y, 1.0);
    let channel_count : u32 = max(as_u32(params.dims_freq.z), 1u);
    let freq_param : f32 = params.dims_freq.w;

    let octave_count : i32 = max(i32(round(params.octave_disp_spline_map.x)), 0);
    let displacement_base : f32 = params.octave_disp_spline_map.y;
    let spline_order : i32 = i32(round(params.octave_disp_spline_map.z));
    let warp_map_enabled : bool = params.octave_disp_spline_map.w > 0.5;

    let signed_range : bool = params.signed_time_speed_pad.x > 0.5;
    let time : f32 = params.signed_time_speed_pad.y;
    let speed : f32 = params.signed_time_speed_pad.z;

    let freq_shape : vec2<f32> = freq_for_shape(freq_param, width_f, height_f);

    var sample_coord : vec2<f32> = vec2<f32>(f32(gid.x), f32(gid.y));

    if (octave_count > 0 && displacement_base != 0.0) {
        let base_coord : vec2<u32> = gid.xy;
        var octave : i32 = 1;
        loop {
            if (octave > octave_count) {
                break;
            }

            let multiplier : f32 = pow(2.0, f32(octave));
            let freq_scaled : vec2<f32> = freq_shape * 0.5 * multiplier;
            let freq_floored : vec2<f32> = vec2<f32>(
                max(floor(freq_scaled.x), 1.0),
                max(floor(freq_scaled.y), 1.0)
            );

            if (freq_floored.x >= width_f || freq_floored.y >= height_f) {
                break;
            }

            let reference : vec2<f32> = compute_warp_reference(
                base_coord,
                width_f,
                height_f,
                freq_floored,
                time,
                speed,
                warp_map_enabled,
                channel_count
            );
            let displacement_scale : f32 = displacement_base / multiplier;
            let offsets : vec2<f32> = displacement_offset(
                reference,
                signed_range,
                displacement_scale,
                width_f,
                height_f
            );

            sample_coord = sample_coord + offsets;
            sample_coord = vec2<f32>(
                wrap_float(sample_coord.x, width_f),
                wrap_float(sample_coord.y, height_f)
            );

            octave = octave + 1;
        }
    }

    let sampled : vec4<f32> = sample_with_order(sample_coord, width, height, spline_order);

    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * channel_count;
    var channel : u32 = 0u;
    loop {
        if (channel >= channel_count) {
            break;
        }
        let component : f32 = sampled[min(channel, 3u)];
        output_buffer[base_index + channel] = clamp_01(component);
        channel = channel + 1u;
    }
}
