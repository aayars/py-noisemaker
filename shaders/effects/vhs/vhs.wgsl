// Bad VHS tracking effect replicating noisemaker.effects.vhs.
// Generates scanline noise, blends it with the source image, and horizontally shifts rows
// based on the noisy gradient to mimic faulty VHS tracking.

const TAU : f32 = 6.28318530717958647692;
const CHANNEL_COUNT : u32 = 4u;

struct VHSParams {
    size : vec4<f32>,    // (width, height, channels, unused)
    motion : vec4<f32>,  // (time, speed, unused, unused)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : VHSParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp_01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn wrap_coord(value : i32, limit : i32) -> i32 {
    if (limit <= 0) {
        return 0;
    }

    var wrapped : i32 = value % limit;
    if (wrapped < 0) {
        wrapped = wrapped + limit;
    }

    return wrapped;
}

fn freq_for_shape(base_freq : f32, width : f32, height : f32) -> vec2<f32> {
    if (base_freq <= 0.0) {
        return vec2<f32>(1.0, 1.0);
    }

    if (abs(width - height) < 1e-5) {
        return vec2<f32>(base_freq, base_freq);
    }

    if (height < width) {
        let ratio : f32 = width / max(height, 1.0);
        return vec2<f32>(base_freq * ratio, base_freq);
    }

    let ratio : f32 = height / max(width, 1.0);
    return vec2<f32>(base_freq, base_freq * ratio);
}

fn normalized_coord(x : f32, y : f32, width : f32, height : f32) -> vec2<f32> {
    let width_safe : f32 = max(width, 1.0);
    let height_safe : f32 = max(height, 1.0);
    return vec2<f32>((x + 0.5) / width_safe, (y + 0.5) / height_safe);
}

fn periodic_value(time : f32, value : f32) -> f32 {
    return sin((time - value) * TAU) * 0.5 + 0.5;
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

    let i : vec3<f32> = mod289_vec3(i0);
    let p : vec4<f32> = permute(permute(permute(
        i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0))
        + i.y + vec4<f32>(0.0, i1.y, i2.y, 1.0))
        + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0));

    let n_ : f32 = 0.14285714285714285;
    let ns : vec3<f32> = n_ * vec3<f32>(D.w, D.y, D.z) - vec3<f32>(D.x, D.z, D.x);

    let j : vec4<f32> = p - 49.0 * floor(p * ns.z * ns.z);
    let x_ : vec4<f32> = floor(j * ns.z);
    let y_ : vec4<f32> = floor(j - 7.0 * x_);

    let x : vec4<f32> = x_ * ns.x + ns.y;
    let y : vec4<f32> = y_ * ns.x + ns.y;
    let h : vec4<f32> = 1.0 - abs(x) - abs(y);

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

fn compute_simplex_value(coord : vec2<f32>, freq : vec2<f32>, time : f32,
                         speed : f32, offset : vec3<f32>) -> f32 {
    let freq_x : f32 = max(freq.x, 1.0);
    let freq_y : f32 = max(freq.y, 1.0);
    let uv : vec3<f32> = vec3<f32>(
        coord.x * freq_x + offset.x,
        coord.y * freq_y + offset.y,
        cos(time * TAU) * speed + offset.z
    );

    return simplex_noise(uv);
}

fn compute_value_noise(coord : vec2<f32>, freq : vec2<f32>, time : f32, speed : f32,
                       base_offset : vec3<f32>, time_offset : vec3<f32>) -> f32 {
    let base_noise : f32 = compute_simplex_value(coord, freq, time, speed, base_offset);
    var value : f32 = clamp_01(base_noise * 0.5 + 0.5);

    if (speed != 0.0 && time != 0.0) {
        let time_noise_raw : f32 = compute_simplex_value(coord, freq, 0.0, 1.0, time_offset);
        let time_value : f32 = clamp_01(time_noise_raw * 0.5 + 0.5);
        let scaled_time : f32 = periodic_value(time, time_value) * speed;
        value = periodic_value(scaled_time, value);
    }

    return clamp_01(value);
}

fn compute_grad_value(coord : vec2<f32>, freq : vec2<f32>, time : f32, speed : f32) -> f32 {
    // Emulate the Python/JS pipeline that builds the gradient noise from a 5x1 lattice.
    // Sample at a fixed x coordinate so each scanline shares the same base value.
    // Only the vertical axis varies, yielding horizontal bars.
    let column_coord : vec2<f32> = vec2<f32>(0.0, coord.y);
    let column_freq : vec2<f32> = vec2<f32>(1.0, freq.y);
    let base : f32 = compute_value_noise(
        column_coord,
        column_freq,
        time,
        speed,
        vec3<f32>(17.0, 29.0, 47.0),
        vec3<f32>(71.0, 113.0, 191.0)
    );

    var g : f32 = max(base - 0.5, 0.0);
    g = min(g * 2.0, 1.0);
    return g;
}

fn compute_scan_noise(coord : vec2<f32>, freq : vec2<f32>, time : f32, speed : f32) -> f32 {
    return compute_value_noise(
        coord,
        freq,
        time,
        speed,
        vec3<f32>(37.0, 59.0, 83.0),
        vec3<f32>(131.0, 173.0, 211.0)
    );
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.size.x);
    let height : u32 = as_u32(params.size.y);
    if (width == 0u || height == 0u) {
        return;
    }
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let base_index : u32 = (gid.y * width + gid.x) * CHANNEL_COUNT;

    let width_f : f32 = max(params.size.x, 1.0);
    let height_f : f32 = max(params.size.y, 1.0);
    let time : f32 = params.motion.x;
    let speed : f32 = params.motion.y;

    // Match the Python/JS implementations where the gradient noise varies across rows,
    // producing horizontal offsets per scanline. That corresponds to five samples along
    // the vertical axis and a single column horizontally.
    let grad_freq : vec2<f32> = vec2<f32>(1.0, 5.0);
    let scan_base_freq : f32 = floor(height_f * 0.5) + 1.0;
    let scan_freq : vec2<f32> = freq_for_shape(scan_base_freq, width_f, height_f);

    let dest_coord_norm : vec2<f32> = normalized_coord(f32(gid.x), f32(gid.y), width_f, height_f);
    let grad_dest : f32 = compute_grad_value(dest_coord_norm, grad_freq, time, speed);
    let scan_dest : f32 = compute_scan_noise(dest_coord_norm, scan_freq, time, speed * 100.0);

    let shift_amount : i32 = i32(floor(scan_dest * width_f * grad_dest * grad_dest));
    let src_x : i32 = wrap_coord(i32(gid.x) - shift_amount, i32(width));
    let src_coord : vec2<i32> = vec2<i32>(src_x, i32(gid.y));

    let src_coord_norm : vec2<f32> = normalized_coord(f32(src_x), f32(gid.y), width_f, height_f);
    let grad_source : f32 = compute_grad_value(src_coord_norm, grad_freq, time, speed);
    let scan_source : f32 = compute_scan_noise(src_coord_norm, scan_freq, time, speed * 100.0);

    let src_texel : vec4<f32> = textureLoad(input_texture, src_coord, 0);
    let noise_color : vec4<f32> = vec4<f32>(scan_source);
    let blended : vec4<f32> = mix(src_texel, noise_color, vec4<f32>(grad_source));

    for (var channel : u32 = 0u; channel < CHANNEL_COUNT; channel = channel + 1u) {
        output_buffer[base_index + channel] = blended[channel];
    }
}
