// Clouds effect: multi-pass translation of noisemaker.effects.clouds.

const CHANNEL_COUNT : u32 = 4u;
const CONTROL_OCTAVES : u32 = 8u;
const WARP_OCTAVES : u32 = 2u;
const WARP_SUB_OCTAVES : u32 = 3u;
const WARP_DISPLACEMENT : f32 = 0.125;
const SHADE_PRE_SCALE : f32 = 2.5;
const SHADE_SCALE : f32 = 1.0;
const BLUR_RADIUS : i32 = 6;
const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;

const CONTROL_BASE_SEED : vec3<f32> = vec3<f32>(17.0, 29.0, 47.0);
const CONTROL_TIME_SEED : vec3<f32> = vec3<f32>(71.0, 113.0, 191.0);
const WARP_BASE_SEED : vec3<f32> = vec3<f32>(23.0, 37.0, 59.0);
const WARP_TIME_SEED : vec3<f32> = vec3<f32>(83.0, 127.0, 211.0);

const TRIPLE_GAUSS_KERNEL : array<f32, 13> = array<f32, 13>(
    0.0002441406,
    0.0029296875,
    0.0161132812,
    0.0537109375,
    0.1208496094,
    0.1933593750,
    0.2255859375,
    0.1933593750,
    0.1208496094,
    0.0537109375,
    0.0161132812,
    0.0029296875,
    0.0002441406,
);

struct CloudsParams {
    size_time : vec4<f32>,   // width, height, channels, time
    anim_down : vec4<f32>,   // speed, downsample_width, downsample_height, pre_scale
    inv_offset : vec4<f32>,  // inv_downsample_width, inv_downsample_height, shade_offset_x, shade_offset_y
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : CloudsParams;
@group(0) @binding(3) var<storage, read_write> downsample_buffer : array<f32>;
@group(0) @binding(4) var<storage, read_write> stats_buffer : array<f32>;

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn wrap_component(value : f32, size : f32) -> f32 {
    if (size <= 0.0) {
        return 0.0;
    }
    let wrapped : f32 = value - floor(value / size) * size;
    if (wrapped < 0.0) {
        return wrapped + size;
    }
    return wrapped;
}

fn wrap_coord(coord : vec2<f32>, dims : vec2<f32>) -> vec2<f32> {
    return vec2<f32>(wrap_component(coord.x, dims.x), wrap_component(coord.y, dims.y));
}

fn wrap_index(value : i32, limit : i32) -> i32 {
    if (limit <= 0) {
        return 0;
    }
    var wrapped : i32 = value % limit;
    if (wrapped < 0) {
        wrapped = wrapped + limit;
    }
    return wrapped;
}

fn freq_for_shape(base_freq : f32, dims : vec2<f32>) -> vec2<f32> {
    // Python takes [height, width] and returns [freq_y, freq_x]
    // Shader dims are (width, height), so dims.y = height, dims.x = width
    let width : f32 = max(dims.x, 1.0);
    let height : f32 = max(dims.y, 1.0);
    if (abs(width - height) < 0.5) {
        return vec2<f32>(base_freq, base_freq);
    }
    if (height < width) {
        return vec2<f32>(base_freq, base_freq * width / height);
    }
    return vec2<f32>(base_freq * height / width, base_freq);
}

fn ridge_transform(value : f32) -> f32 {
    return 1.0 - abs(value * 2.0 - 1.0);
}
fn normalized_sine(value : f32) -> f32 {
    return sin(value) * 0.5 + 0.5;
}

fn periodic_value(value : f32, phase : f32) -> f32 {
    return normalized_sine((value - phase) * TAU);
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

fn animated_simplex_value(
    coord : vec2<f32>,
    dims : vec2<f32>,
    freq : vec2<f32>,
    time_value : f32,
    speed_value : f32,
    base_seed : vec3<f32>,
    time_seed : vec3<f32>
) -> f32 {
    let angle : f32 = TAU * time_value;
    let z_coord : f32 = cos(angle) * speed_value;

    let scale : vec2<f32> = freq / dims;
    let scaled_coord : vec2<f32> = coord * scale;

    let base_noise : f32 = simplex_noise(vec3<f32>(
        scaled_coord.x + base_seed.x,
        scaled_coord.y + base_seed.y,
        z_coord
    ));

    if (speed_value == 0.0 || time_value == 0.0) {
        return base_noise;
    }

    let time_noise : f32 = simplex_noise(vec3<f32>(
        scaled_coord.x + time_seed.x,
        scaled_coord.y + time_seed.y,
        1.0
    ));

    let scaled_time : f32 = periodic_value(time_value, time_noise) * speed_value;
    return periodic_value(scaled_time, base_noise);
}

fn seeded_base_frequency(dims : vec2<f32>) -> f32 {
    let hash_val : f32 = fract(sin(dot(dims, vec2<f32>(12.9898, 78.233))) * 43758.5453123);
    return floor(hash_val * 3.0) + 2.0;
}
fn simplex_multires_value(
    coord : vec2<f32>,
    dims : vec2<f32>,
    base_freq : vec2<f32>,
    time_value : f32,
    speed_value : f32,
    octaves : u32,
    ridged : bool,
    base_seed : vec3<f32>,
    time_seed : vec3<f32>
) -> f32 {
    let safe_dims : vec2<f32> = vec2<f32>(max(dims.x, 1.0), max(dims.y, 1.0));

    var accum : f32 = 0.0;

    for (var octave : u32 = 1u; octave <= octaves; octave = octave + 1u) {
        let multiplier : f32 = pow(2.0, f32(octave));
        let octave_freq : vec2<f32> = vec2<f32>(
            base_freq.x * 0.5 * multiplier,
            base_freq.y * 0.5 * multiplier
        );

        if (octave_freq.x > safe_dims.x && octave_freq.y > safe_dims.y) {
            break;
        }

        let seed_offset : vec3<f32> = vec3<f32>(
            f32(octave) * 37.0,
            f32(octave) * 53.0,
            f32(octave) * 19.0
        );
        let time_offset : vec3<f32> = vec3<f32>(
            f32(octave) * 41.0,
            f32(octave) * 23.0,
            f32(octave) * 61.0
        );

        let sample_value : f32 = animated_simplex_value(
            coord,
            safe_dims,
            octave_freq,
            time_value + f32(octave) * 0.07,
            speed_value,
            base_seed + seed_offset,
            time_seed + time_offset
        );

        var layer : f32 = sample_value;
        if (ridged) {
            layer = ridge_transform(layer);
        }

        // Python: tensor += layer / multiplier
        let amplitude : f32 = 1.0 / multiplier;
        accum = accum + layer * amplitude;
    }

    return accum;
}

fn warp_coordinate(
    coord : vec2<f32>,
    dims : vec2<f32>,
    time_value : f32,
    speed_value : f32
) -> vec2<f32> {
    var warped : vec2<f32> = coord;
    let base_freq : vec2<f32> = freq_for_shape(3.0, dims);

    var octave : u32 = 0u;
    loop {
        if (octave >= WARP_OCTAVES) {
            break;
        }

        let freq_scale : vec2<f32> = base_freq * pow(2.0, f32(octave));
        let flow_x : f32 = simplex_multires_value(
            warped,
            dims,
            freq_scale,
            time_value + f32(octave) * 0.21,
            speed_value,
            WARP_SUB_OCTAVES,
            false,
            WARP_BASE_SEED + vec3<f32>(f32(octave) * 13.0, f32(octave) * 17.0, f32(octave) * 19.0),
            WARP_TIME_SEED + vec3<f32>(f32(octave) * 23.0, f32(octave) * 29.0, f32(octave) * 31.0)
        );

        let flow_y : f32 = simplex_multires_value(
            wrap_coord(warped + vec2<f32>(0.5, 0.5), dims),
            dims,
            freq_scale,
            time_value + f32(octave) * 0.37,
            speed_value,
            WARP_SUB_OCTAVES,
            false,
            WARP_BASE_SEED + vec3<f32>(f32(octave) * 19.0, f32(octave) * 23.0, f32(octave) * 29.0) + vec3<f32>(11.0, 7.0, 5.0),
            WARP_TIME_SEED + vec3<f32>(f32(octave) * 31.0, f32(octave) * 37.0, f32(octave) * 41.0) + vec3<f32>(13.0, 19.0, 17.0)
        );

        let offset_vec : vec2<f32> = vec2<f32>(flow_x * 2.0 - 1.0, flow_y * 2.0 - 1.0);
        let displacement : f32 = WARP_DISPLACEMENT / pow(2.0, f32(octave));
        warped = wrap_coord(warped + offset_vec * displacement * dims, dims);

        octave = octave + 1u;
    }

    return warped;
}

fn control_value_at(
    coord : vec2<f32>,
    dims : vec2<f32>,
    time_value : f32,
    speed_value : f32
) -> f32 {
    let base_freq_value : f32 = seeded_base_frequency(dims);
    let freq_vec : vec2<f32> = freq_for_shape(base_freq_value, dims);
    // Python clouds calls warp without time/speed, so we use 0.0, 1.0
    let warped_coord : vec2<f32> = warp_coordinate(coord, dims, 0.0, 1.0);
    return simplex_multires_value(
        warped_coord,
        dims,
        freq_vec,
        time_value,
        speed_value,
        CONTROL_OCTAVES,
        true,
        CONTROL_BASE_SEED,
        CONTROL_TIME_SEED
    );
}

fn normalize_control(raw_value : f32, min_value : f32, max_value : f32) -> f32 {
    let delta : f32 = max(max_value - min_value, 1e-6);
    return clamp((raw_value - min_value) / delta, 0.0, 1.0);
}

fn combined_from_normalized(control_norm : f32) -> f32 {
    let scaled : f32 = control_norm * 2.0;
    if (scaled < 1.0) {
        return clamp(1.0 - scaled, 0.0, 1.0);
    }
    return 0.0;
}

fn combined_from_raw(raw_value : f32, min_value : f32, max_value : f32) -> f32 {
    let control_norm : f32 = normalize_control(raw_value, min_value, max_value);
    return combined_from_normalized(control_norm);
}

fn read_channel(coord : vec2<i32>, size : vec2<i32>, channel : u32) -> f32 {
    let width : i32 = max(size.x, 1);
    let height : i32 = max(size.y, 1);
    let safe_x : i32 = wrap_index(coord.x, width);
    let safe_y : i32 = wrap_index(coord.y, height);
    let base_index : u32 = (u32(safe_y) * u32(width) + u32(safe_x)) * CHANNEL_COUNT + channel;
    return downsample_buffer[base_index];
}

fn cubic_interpolate_scalar(a : f32, b : f32, c : f32, d : f32, t : f32) -> f32 {
    let t2 : f32 = t * t;
    let t3 : f32 = t2 * t;
    let a0 : f32 = d - c - a + b;
    let a1 : f32 = a - b - a0;
    let a2 : f32 = c - a;
    let a3 : f32 = b;
    return a0 * t3 + a1 * t2 + a2 * t + a3;
}

fn sample_channel_bicubic(uv : vec2<f32>, size : vec2<i32>, channel : u32) -> f32 {
    let width : i32 = max(size.x, 1);
    let height : i32 = max(size.y, 1);
    let scale : vec2<f32> = vec2<f32>(f32(width), f32(height));
    let base_coord : vec2<f32> = uv * scale - vec2<f32>(0.5, 0.5);

    let ix : i32 = i32(floor(base_coord.x));
    let iy : i32 = i32(floor(base_coord.y));
    let fx : f32 = clamp(base_coord.x - floor(base_coord.x), 0.0, 1.0);
    let fy : f32 = clamp(base_coord.y - floor(base_coord.y), 0.0, 1.0);

    var column : array<f32, 4>;
    var row : array<f32, 4>;

    var m : i32 = -1;
    loop {
        if (m > 2) {
            break;
        }

        var n : i32 = -1;
        loop {
            if (n > 2) {
                break;
            }

            let sample_coord : vec2<i32> = vec2<i32>(
                wrap_index(ix + n, width),
                wrap_index(iy + m, height)
            );
            row[u32(n + 1)] = read_channel(sample_coord, size, channel);
            n = n + 1;
        }

        column[u32(m + 1)] = cubic_interpolate_scalar(row[0], row[1], row[2], row[3], fx);
        m = m + 1;
    }

    let value : f32 = cubic_interpolate_scalar(column[0], column[1], column[2], column[3], fy);
    return clamp(value, 0.0, 1.0);
}

fn blur_shade(coord : vec2<i32>, size : vec2<i32>, offset : vec2<i32>, min_value : f32, max_value : f32) -> f32 {
    let width : i32 = max(size.x, 1);
    let height : i32 = max(size.y, 1);
    
    // Python does: shaded = offset(combined) then shaded *= 2.5 then blur(shaded)
    // We need to blur the OFFSET+BOOSTED values
    // Each sample in the blur reads from (coord_in_blur_space + global_offset)
    
    var accum : f32 = 0.0;
    var dy : i32 = -BLUR_RADIUS;
    loop {
        if (dy > BLUR_RADIUS) {
            break;
        }

        let weight_y : f32 = TRIPLE_GAUSS_KERNEL[u32(dy + BLUR_RADIUS)];
        var row_accum : f32 = 0.0;
        var dx : i32 = -BLUR_RADIUS;
        loop {
            if (dx > BLUR_RADIUS) {
                break;
            }

            let weight_x : f32 = TRIPLE_GAUSS_KERNEL[u32(dx + BLUR_RADIUS)];
            // Read from shifted position: (coord + blur_delta + global_offset)
            let sample_coord : vec2<i32> = vec2<i32>(
                wrap_index(coord.x + dx + offset.x, width),
                wrap_index(coord.y + dy + offset.y, height)
            );
            let control_raw : f32 = read_channel(sample_coord, size, 2u);
            let combined : f32 = combined_from_raw(control_raw, min_value, max_value);
            let boosted : f32 = min(combined * 2.5, 1.0);
            row_accum = row_accum + boosted * weight_x;

            dx = dx + 1;
        }

        accum = accum + row_accum * weight_y;
        dy = dy + 1;
    }

    return clamp01(accum);
}

fn sample_texture_bilinear(uv : vec2<f32>, tex_size : vec2<i32>) -> vec4<f32> {
    let width : f32 = f32(tex_size.x);
    let height : f32 = f32(tex_size.y);
    
    let coord : vec2<f32> = vec2<f32>(uv.x * width - 0.5, uv.y * height - 0.5);
    let coord_floor : vec2<i32> = vec2<i32>(i32(floor(coord.x)), i32(floor(coord.y)));
    let fract_part : vec2<f32> = vec2<f32>(coord.x - floor(coord.x), coord.y - floor(coord.y));
    
    let x0 : i32 = wrap_index(coord_floor.x, tex_size.x);
    let y0 : i32 = wrap_index(coord_floor.y, tex_size.y);
    let x1 : i32 = wrap_index(coord_floor.x + 1, tex_size.x);
    let y1 : i32 = wrap_index(coord_floor.y + 1, tex_size.y);
    
    let p00 : vec4<f32> = textureLoad(input_texture, vec2<i32>(x0, y0), 0);
    let p10 : vec4<f32> = textureLoad(input_texture, vec2<i32>(x1, y0), 0);
    let p01 : vec4<f32> = textureLoad(input_texture, vec2<i32>(x0, y1), 0);
    let p11 : vec4<f32> = textureLoad(input_texture, vec2<i32>(x1, y1), 0);
    
    let p0 : vec4<f32> = mix(p00, p10, fract_part.x);
    let p1 : vec4<f32> = mix(p01, p11, fract_part.x);
    
    return mix(p0, p1, fract_part.y);
}

fn sobel_gradient(uv : vec2<f32>, size : vec2<i32>) -> vec2<f32> {
    let width : i32 = max(size.x, 1);
    let height : i32 = max(size.y, 1);

    // First, blur the input (matching Python's sobel_operator)
    var blurred_value : f32 = 0.0;
    for (var i : i32 = -1; i <= 1; i = i + 1) {
        for (var j : i32 = -1; j <= 1; j = j + 1) {
            let sample_uv : vec2<f32> = uv + vec2<f32>(f32(j) / f32(width), f32(i) / f32(height));
            let texel : vec4<f32> = sample_texture_bilinear(sample_uv, size);
            let luminance : f32 = (texel.r + texel.g + texel.b) / 3.0;
            blurred_value = blurred_value + luminance;
        }
    }
    blurred_value = blurred_value / 9.0;

    // Sobel kernels
    let x_kernel : mat3x3<f32> = mat3x3<f32>(
        vec3<f32>(-1.0, 0.0, 1.0),
        vec3<f32>(-2.0, 0.0, 2.0),
        vec3<f32>(-1.0, 0.0, 1.0)
    );

    let y_kernel : mat3x3<f32> = mat3x3<f32>(
        vec3<f32>(-1.0, -2.0, -1.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(1.0, 2.0, 1.0)
    );

    var gx : f32 = 0.0;
    var gy : f32 = 0.0;

    for (var i : i32 = -1; i <= 1; i = i + 1) {
        for (var j : i32 = -1; j <= 1; j = j + 1) {
            let sample_uv : vec2<f32> = uv + vec2<f32>(f32(j) / f32(width), f32(i) / f32(height));
            let texel : vec4<f32> = sample_texture_bilinear(sample_uv, size);
            let value : f32 = (texel.r + texel.g + texel.b) / 3.0;

            gx = gx + value * x_kernel[i + 1][j + 1];
            gy = gy + value * y_kernel[i + 1][j + 1];
        }
    }

    return vec2<f32>(gx, gy);
}

fn shadow(original_texel: vec4<f32>, uv : vec2<f32>, size : vec2<i32>, alpha : f32) -> vec4<f32> {
    // Get Sobel gradients
    let gradient : vec2<f32> = sobel_gradient(uv, size);
    
    // Calculate Euclidean distance and normalize (simplified - no global normalization)
    let distance : f32 = sqrt(gradient.x * gradient.x + gradient.y * gradient.y);
    let normalized_distance : f32 = clamp(distance, 0.0, 1.0);
    
    // Apply sharpen effect (simplified - just boost the contrast)
    var shade : f32 = normalized_distance;
    shade = clamp((shade - 0.5) * 1.5 + 0.5, 0.0, 1.0);
    
    // Create highlight by squaring
    let highlight : f32 = shade * shade;
    
    // Apply shadow formula: shade = (1.0 - ((1.0 - tensor) * (1.0 - highlight))) * shade
    let shadowed : vec3<f32> = (vec3<f32>(1.0) - ((vec3<f32>(1.0) - original_texel.rgb) * (1.0 - highlight))) * shade;
    
    // Blend with original
    return vec4<f32>(mix(original_texel.rgb, shadowed, alpha), original_texel.a);
}

@compute @workgroup_size(8, 8, 1)
fn downsample_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let down_width : i32 = max(i32(round(params.anim_down.y)), 1);
    let down_height : i32 = max(i32(round(params.anim_down.z)), 1);
    if (gid.x >= u32(down_width) || gid.y >= u32(down_height)) {
        return;
    }

    let dims : vec2<f32> = vec2<f32>(f32(down_width), f32(down_height));
    let coord : vec2<f32> = vec2<f32>(f32(gid.x), f32(gid.y));
    let time_value : f32 = params.size_time.w;
    let speed_value : f32 = params.anim_down.x;

    let control_raw : f32 = control_value_at(coord, dims, time_value, speed_value);
    let base_index : u32 = (gid.y * u32(down_width) + gid.x) * CHANNEL_COUNT;
    downsample_buffer[base_index + 0u] = 0.0;
    downsample_buffer[base_index + 1u] = 0.0;
    downsample_buffer[base_index + 2u] = control_raw;
    downsample_buffer[base_index + 3u] = 1.0;
}

@compute @workgroup_size(8, 8, 1)
fn shade_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let down_width : i32 = max(i32(round(params.anim_down.y)), 1);
    let down_height : i32 = max(i32(round(params.anim_down.z)), 1);
    if (gid.x >= u32(down_width) || gid.y >= u32(down_height)) {
        return;
    }

    let size_i : vec2<i32> = vec2<i32>(down_width, down_height);
    let coord_i : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let offset_i : vec2<i32> = vec2<i32>(
        i32(round(params.inv_offset.z)),
        i32(round(params.inv_offset.w))
    );

    let min_value : f32 = stats_buffer[0];
    let max_value : f32 = stats_buffer[1];
    let control_raw : f32 = read_channel(coord_i, size_i, 2u);
    let combined : f32 = combined_from_raw(control_raw, min_value, max_value);
    let shade : f32 = blur_shade(coord_i, size_i, offset_i, min_value, max_value);
    let base_index : u32 = (gid.y * u32(down_width) + gid.x) * CHANNEL_COUNT;
    downsample_buffer[base_index + 0u] = combined;
    downsample_buffer[base_index + 1u] = shade;
    downsample_buffer[base_index + 2u] = control_raw;
    downsample_buffer[base_index + 3u] = 1.0;
}

// Compute true min and max of combined values for normalization
@compute @workgroup_size(1, 1, 1)
fn normalize_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    // Only single invocation
    if (gid.x != 0u || gid.y != 0u) { return; }
    let down_w : i32 = max(i32(round(params.anim_down.y)), 1);
    let down_h : i32 = max(i32(round(params.anim_down.z)), 1);
    var minv : f32 = 1e30;
    var maxv : f32 = -1e30;
    // Iterate all downsample buffer texels
    for (var yy : i32 = 0; yy < down_h; yy = yy + 1) {
        for (var xx : i32 = 0; xx < down_w; xx = xx + 1) {
            let base_index : u32 = (u32(yy) * u32(down_w) + u32(xx)) * CHANNEL_COUNT;
            let v : f32 = downsample_buffer[base_index + 2u];
            minv = min(minv, v);
            maxv = max(maxv, v);
        }
    }
    stats_buffer[0] = minv;
    stats_buffer[1] = maxv;
}

@compute @workgroup_size(8, 8, 1)
fn upsample_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : i32 = max(i32(round(params.size_time.x)), 1);
    let height : i32 = max(i32(round(params.size_time.y)), 1);
    if (gid.x >= u32(width) || gid.y >= u32(height)) {
        return;
    }

    let down_size_i : vec2<i32> = vec2<i32>(
        max(i32(round(params.anim_down.y)), 1),
        max(i32(round(params.anim_down.z)), 1)
    );

    let uv : vec2<f32> = vec2<f32>(
        (f32(gid.x) + 0.5) / max(params.size_time.x, 1.0),
        (f32(gid.y) + 0.5) / max(params.size_time.y, 1.0)
    );

    // Combined is already 0-1 from blend_layers
    let combined_value : f32 = clamp01(sample_channel_bicubic(uv, down_size_i, 0u));
    
    // Sample and soften shade mask
    var shade_mask : f32 = sample_channel_bicubic(uv, down_size_i, 1u);
    // reduce harshness and boost low values
    let shade_factor : f32 = smoothstep(0.0, 0.5, shade_mask * 0.75);

    let texel : vec4<f32> = textureLoad(
        input_texture,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        0,
    );

    let shaded_color : vec3<f32> = mix(texel.xyz, vec3<f32>(0.0), vec3<f32>(shade_factor));
    let lit_color : vec4<f32> = vec4<f32>(mix(shaded_color, vec3<f32>(1.0), vec3<f32>(combined_value)), clamp(mix(texel.w, 1.0, combined_value), 0.0, 1.0));

    let final_texel : vec4<f32> = shadow(lit_color, uv, vec2<i32>(width, height), 0.5);

    let pixel_index : u32 = (gid.y * u32(width) + gid.x) * CHANNEL_COUNT;
    output_buffer[pixel_index + 0u] = final_texel.x;
    output_buffer[pixel_index + 1u] = final_texel.y;
    output_buffer[pixel_index + 2u] = final_texel.z;
    output_buffer[pixel_index + 3u] = final_texel.w;
}
