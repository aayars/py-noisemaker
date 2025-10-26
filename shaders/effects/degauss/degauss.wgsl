// Degauss: simulate a CRT-style degaussing wobble by lens-warping
// each color channel independently. Based on the Python
// implementation in effects.degauss(), which repeatedly invokes
// lens_warp() with simplex noise-derived displacements.

const TAU : f32 = 6.28318530717958647692;

struct DegaussParams {
    dims0 : vec4<f32>, // (width, height, displacement, time)
    dims1 : vec4<f32>, // (speed, _pad0, _pad1, _pad2)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : DegaussParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(value, 0.0));
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
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
    let p = permute(permute(permute(
        i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0))
        + i.y + vec4<f32>(0.0, i1.y, i2.y, 1.0))
        + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0));

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

fn compute_noise_value(
    coord : vec2<u32>,
    width : f32,
    height : f32,
    freq : vec2<f32>,
    time : f32,
    speed : f32,
    channel : u32,
) -> f32 {
    let width_safe : f32 = max(width, 1.0);
    let height_safe : f32 = max(height, 1.0);
    let freq_x : f32 = max(freq.y, 1.0);
    let freq_y : f32 = max(freq.x, 1.0);

    let uv : vec2<f32> = vec2<f32>(
        (f32(coord.x) / width_safe) * freq_x,
        (f32(coord.y) / height_safe) * freq_y
    );

    let angle : f32 = time * TAU;
    let z_base : f32 = cos(angle) * speed;
    let channel_offset : f32 = f32(channel) * 37.0;
    let base_seed : vec3<f32> = vec3<f32>(
        17.0 + channel_offset,
        29.0 + channel_offset * 1.3,
        47.0 + channel_offset * 1.7
    );

    let base_noise : f32 = simplex_noise(vec3<f32>(
        uv.x + base_seed.x,
        uv.y + base_seed.y,
        z_base + base_seed.z
    ));

    var value : f32 = clamp(base_noise * 0.5 + 0.5, 0.0, 1.0);

    if (speed != 0.0 && time != 0.0) {
        let time_seed : vec3<f32> = vec3<f32>(
            base_seed.x + 54.0,
            base_seed.y + 82.0,
            base_seed.z + 124.0
        );
        let time_noise : f32 = simplex_noise(vec3<f32>(
            uv.x + time_seed.x,
            uv.y + time_seed.y,
            time_seed.z
        ));
        let time_value : f32 = clamp(time_noise * 0.5 + 0.5, 0.0, 1.0);
        let scaled_time : f32 = periodic_value(time, time_value) * speed;
        value = clamp01(periodic_value(scaled_time, value));
    }

    return clamp01(value);
}

fn singularity_mask(uv : vec2<f32>, width : f32, height : f32) -> f32 {
    if (width <= 0.0 || height <= 0.0) {
        return 0.0;
    }

    let delta : vec2<f32> = abs(uv - vec2<f32>(0.5, 0.5));
    let aspect : f32 = width / height;
    let scaled : vec2<f32> = vec2<f32>(delta.x * aspect, delta.y);
    let max_radius : f32 = length(vec2<f32>(aspect * 0.5, 0.5));
    if (max_radius <= 0.0) {
        return 0.0;
    }

    let normalized : f32 = clamp(length(scaled) / max_radius, 0.0, 1.0);
    let masked : f32 = sqrt(normalized);
    return pow(masked, 5.0);
}

fn sample_bilinear(pos : vec2<f32>, width : f32, height : f32) -> vec4<f32> {
    let width_f : f32 = max(width, 1.0);
    let height_f : f32 = max(height, 1.0);

    let wrapped_x : f32 = wrap_float(pos.x, width_f);
    let wrapped_y : f32 = wrap_float(pos.y, height_f);

    var x0 : i32 = i32(floor(wrapped_x));
    var y0 : i32 = i32(floor(wrapped_y));

    let width_i : i32 = i32(max(width, 1.0));
    let height_i : i32 = i32(max(height, 1.0));

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

    let x1 : i32 = wrap_index(x0 + 1, width_i);
    let y1 : i32 = wrap_index(y0 + 1, height_i);

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

fn warped_channel_value(
    channel : u32,
    coord : vec2<u32>,
    base_pos : vec2<f32>,
    width : f32,
    height : f32,
    freq : vec2<f32>,
    displacement : f32,
    mask : f32,
    time : f32,
    speed : f32,
) -> f32 {
    let noise_value : f32 = compute_noise_value(coord, width, height, freq, time, speed, channel);
    let centered : f32 = (noise_value * 2.0 - 1.0) * mask;
    let angle : f32 = centered * TAU;
    let offset : vec2<f32> = vec2<f32>(cos(angle), sin(angle)) * displacement * vec2<f32>(width, height);
    let sample : vec4<f32> = sample_bilinear(base_pos + offset, width, height);

    switch channel {
        case 0u: {
            return clamp01(sample.x);
        }
        case 1u: {
            return clamp01(sample.y);
        }
        default: {
            return clamp01(sample.z);
        }
    }
}

fn store_pixel(base_index : u32, value : vec4<f32>) {
    output_buffer[base_index + 0u] = value.x;
    output_buffer[base_index + 1u] = value.y;
    output_buffer[base_index + 2u] = value.z;
    output_buffer[base_index + 3u] = value.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.dims0.x);
    let height : u32 = as_u32(params.dims0.y);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * 4u;
    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let original : vec4<f32> = textureLoad(input_texture, coords, 0);

    let displacement : f32 = params.dims0.z;
    if (displacement == 0.0) {
        store_pixel(base_index, original);
        return;
    }

    let width_f : f32 = params.dims0.x;
    let height_f : f32 = params.dims0.y;
    let uv : vec2<f32> = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5, 0.5))
        / vec2<f32>(max(width_f, 1.0), max(height_f, 1.0));
    let mask : f32 = singularity_mask(uv, width_f, height_f);
    if (mask <= 0.0) {
        store_pixel(base_index, original);
        return;
    }

    let freq : vec2<f32> = freq_for_shape(2.0, width_f, height_f);
    let base_pos : vec2<f32> = vec2<f32>(f32(gid.x), f32(gid.y));
    let coord : vec2<u32> = gid.xy;

    let time : f32 = params.dims0.w;
    let speed : f32 = params.dims1.x;

    let red : f32 = warped_channel_value(
        0u,
        coord,
        base_pos,
        width_f,
        height_f,
        freq,
        displacement,
        mask,
        time,
        speed,
    );
    let green : f32 = warped_channel_value(
        1u,
        coord,
        base_pos,
        width_f,
        height_f,
        freq,
        displacement,
        mask,
        time,
        speed,
    );
    let blue : f32 = warped_channel_value(
        2u,
        coord,
        base_pos,
        width_f,
        height_f,
        freq,
        displacement,
        mask,
        time,
        speed,
    );
    let alpha : f32 = clamp01(original.w);

    store_pixel(base_index, vec4<f32>(red, green, blue, alpha));
}
