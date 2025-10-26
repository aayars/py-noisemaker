// Wobble effect: offsets the entire frame using simplex noise-driven jitter.

const TAU : f32 = 6.28318530717958647692;
const CHANNEL_COUNT : u32 = 4u;

struct WobbleParams {
    dims_time : vec4<f32>,   // (width, height, channels, time)
    speed_pad : vec4<f32>,   // (speed, _pad0, _pad1, _pad2)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : WobbleParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn wrap_index(coord : i32, offset : i32, size : i32) -> i32 {
    if (size <= 0) {
        return 0;
    }
    let sum : i32 = coord + offset;
    let modulo : i32 = sum % size;
    if (modulo < 0) {
        return modulo + size;
    }
    return modulo;
}

fn store_texel(base_index : u32, texel : vec4<f32>) {
    output_buffer[base_index + 0u] = texel.x;
    output_buffer[base_index + 1u] = texel.y;
    output_buffer[base_index + 2u] = texel.z;
    output_buffer[base_index + 3u] = texel.w;
}

fn mod_289_vec3(value : vec3<f32>) -> vec3<f32> {
    let divisor : vec3<f32> = vec3<f32>(289.0);
    let quotient : vec3<f32> = floor(value / divisor);
    return value - quotient * divisor;
}

fn mod_289_vec4(value : vec4<f32>) -> vec4<f32> {
    let divisor : vec4<f32> = vec4<f32>(289.0);
    let quotient : vec4<f32> = floor(value / divisor);
    return value - quotient * divisor;
}

fn permute(value : vec4<f32>) -> vec4<f32> {
    let scale : vec4<f32> = vec4<f32>(34.0);
    let offset : vec4<f32> = vec4<f32>(1.0);
    return mod_289_vec4(((value * scale) + offset) * value);
}

fn taylor_inv_sqrt(value : vec4<f32>) -> vec4<f32> {
    let numerator : vec4<f32> = vec4<f32>(1.79284291400159);
    let denominator : vec4<f32> = vec4<f32>(0.85373472095314);
    return numerator - denominator * value;
}

fn simplex_noise(v : vec3<f32>) -> f32 {
    let c : vec2<f32> = vec2<f32>(1.0 / 6.0, 1.0 / 3.0);
    let d : vec4<f32> = vec4<f32>(0.0, 0.5, 1.0, 2.0);

    let c_y : vec3<f32> = vec3<f32>(c.y);
    let c_x : vec3<f32> = vec3<f32>(c.x);
    let i0 : vec3<f32> = floor(v + vec3<f32>(dot(v, c_y)));
    let x0 : vec3<f32> = v - i0 + vec3<f32>(dot(i0, c_x));

    let step1 : vec3<f32> = step(vec3<f32>(x0.y, x0.z, x0.x), x0);
    let l : vec3<f32> = vec3<f32>(1.0) - step1;
    let i1 : vec3<f32> = min(step1, vec3<f32>(l.z, l.x, l.y));
    let i2 : vec3<f32> = max(step1, vec3<f32>(l.z, l.x, l.y));

    let x1 : vec3<f32> = x0 - i1 + c_x;
    let x2 : vec3<f32> = x0 - i2 + c_y;
    let x3 : vec3<f32> = x0 - vec3<f32>(d.y);

    let i : vec3<f32> = mod_289_vec3(i0);
    let permute0 : vec4<f32> = permute(vec4<f32>(i.z) + vec4<f32>(0.0, i1.z, i2.z, 1.0));
    let permute1 : vec4<f32> = permute(permute0 + vec4<f32>(i.y) + vec4<f32>(0.0, i1.y, i2.y, 1.0));
    let p : vec4<f32> = permute(permute1 + vec4<f32>(i.x) + vec4<f32>(0.0, i1.x, i2.x, 1.0));

    let n : f32 = 0.14285714285714285;
    let ns : vec3<f32> = vec3<f32>(n) * vec3<f32>(d.w, d.y, d.z) - vec3<f32>(d.x, d.z, d.x);

    let ns_zz : f32 = ns.z * ns.z;
    let j : vec4<f32> = p - vec4<f32>(49.0) * floor(p * vec4<f32>(ns_zz));
    let x_ : vec4<f32> = floor(j * vec4<f32>(ns.z));
    let y_ : vec4<f32> = floor(j - vec4<f32>(7.0) * x_);

    let x : vec4<f32> = x_ * vec4<f32>(ns.x) + vec4<f32>(ns.y);
    let y : vec4<f32> = y_ * vec4<f32>(ns.x) + vec4<f32>(ns.y);
    let h : vec4<f32> = vec4<f32>(1.0) - abs(x) - abs(y);

    let b0 : vec4<f32> = vec4<f32>(x.x, x.y, y.x, y.y);
    let b1 : vec4<f32> = vec4<f32>(x.z, x.w, y.z, y.w);

    let s0 : vec4<f32> = floor(b0) * vec4<f32>(2.0) + vec4<f32>(1.0);
    let s1 : vec4<f32> = floor(b1) * vec4<f32>(2.0) + vec4<f32>(1.0);
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
    let g0n : vec3<f32> = g0 * vec3<f32>(norm.x);
    let g1n : vec3<f32> = g1 * vec3<f32>(norm.y);
    let g2n : vec3<f32> = g2 * vec3<f32>(norm.z);
    let g3n : vec3<f32> = g3 * vec3<f32>(norm.w);

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

fn simplex_random(time : f32, speed : f32, seed : vec3<f32>) -> f32 {
    let angle : f32 = time * TAU;
    let z : f32 = cos(angle) * speed + seed.x;
    let w : f32 = sin(angle) * speed + seed.y;
    let noise_value : f32 = simplex_noise(vec3<f32>(z, w, seed.z));
    return clamp(noise_value * 0.5 + 0.5, 0.0, 1.0);
}

fn compute_offset(time : f32, speed : f32, dimension : f32, seed : vec3<f32>) -> i32 {
    if (dimension <= 0.0) {
        return 0;
    }
    let random_value : f32 = simplex_random(time, speed, seed);
    let scaled : f32 = random_value * dimension;
    return i32(floor(scaled));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let width : u32 = as_u32(params.dims_time.x);
    let height : u32 = as_u32(params.dims_time.y);

    if (global_id.x >= width || global_id.y >= height) {
        return;
    }

    if (width == 0u || height == 0u) {
        return;
    }

    let half_speed : f32 = params.speed_pad.x * 0.5;
    let x_offset : i32 = compute_offset(
        params.dims_time.w,
        half_speed,
        params.dims_time.x,
        vec3<f32>(17.0, 29.0, 11.0)
    );
    let y_offset : i32 = compute_offset(
        params.dims_time.w,
        half_speed,
        params.dims_time.y,
        vec3<f32>(41.0, 23.0, 7.0)
    );

    let wrapped_x : i32 = wrap_index(i32(global_id.x), x_offset, i32(width));
    let wrapped_y : i32 = wrap_index(i32(global_id.y), y_offset, i32(height));

    let texel : vec4<f32> = textureLoad(input_texture, vec2<i32>(wrapped_x, wrapped_y), 0);

    let pixel_index : u32 = global_id.y * width + global_id.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;
    store_texel(base_index, texel);
}
