// GPU recreation of the "fibers" overlay from the Python effects module.
// The CPU implementation constructs a chaotic worm mask over a coarse
// simplex noise field and blends in a bright high-frequency texture. The
// shader mirrors the same sequence of steps:
//   1. Generate the low-frequency mask noise.
//   2. Derive a chaotic flow field (similar to WormBehavior.chaotic) and
//      integrate along it to approximate the worm scatter.
//   3. Synthesize high-frequency brightness noise.
//   4. Perform four blend passes, matching `value.blend` with an alpha of
//      `mask * 0.5`.

struct FibersParams {
    size : vec4<f32>,       // (width, height, channel count, unused)
    time_speed : vec4<f32>, // (time, speed, unused, unused)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : FibersParams;

const TAU : f32 = 6.28318530717958647692;

fn to_dimension(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp_01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn channel_count(raw : f32) -> u32 {
    let count : u32 = to_dimension(raw);
    return clamp(count, 1u, 4u);
}

fn lerp(a : f32, b : f32, t : f32) -> f32 {
    return a + (b - a) * t;
}

fn hash_11(x : f32) -> f32 {
    return fract(sin(x * 43758.5453123) * 43758.5453123);
}

fn freq_for_shape(base_freq : f32, width : f32, height : f32) -> vec2<f32> {
    let freq : f32 = max(base_freq, 1.0);
    if (abs(width - height) < 1e-5) {
        return vec2<f32>(freq, freq);
    }
    if (height < width && height > 0.0) {
        return vec2<f32>(freq * width / height, freq);
    }
    if (width > 0.0) {
        return vec2<f32>(freq, freq * height / width);
    }
    return vec2<f32>(freq, freq);
}

fn wrap_float(value : f32, limit : f32) -> f32 {
    if (limit <= 0.0) {
        return 0.0;
    }
    let ratio : f32 = floor(value / limit);
    var wrapped : f32 = value - ratio * limit;
    if (wrapped < 0.0) {
        wrapped = wrapped + limit;
    }
    return wrapped;
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
    let p = permute_vec4(permute_vec4(permute_vec4(
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

    let a0 : vec4<f32> = vec4<f32>(b0.x, b0.z, b0.y, b0.w) + vec4<f32>(s0.x, s0.z, s0.y, s0.w)
        * vec4<f32>(sh.x, sh.x, sh.y, sh.y);
    let a1 : vec4<f32> = vec4<f32>(b1.x, b1.z, b1.y, b1.w) + vec4<f32>(s1.x, s1.z, s1.y, s1.w)
        * vec4<f32>(sh.z, sh.z, sh.w, sh.w);

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

    return 42.0 * (m0sq * m0sq * dot(g0n, x0)
        + m1sq * m1sq * dot(g1n, x1)
        + m2sq * m2sq * dot(g2n, x2)
        + m3sq * m3sq * dot(g3n, x3));
}

fn animated_simplex(
    pos : vec2<f32>,
    width : f32,
    height : f32,
    base_freq : f32,
    seed : vec3<f32>,
    time_value : f32,
    speed_value : f32,
) -> f32 {
    let freq : vec2<f32> = freq_for_shape(base_freq, width, height);
    let inv_width : f32 = select(0.0, 1.0 / width, width > 0.0);
    let inv_height : f32 = select(0.0, 1.0 / height, height > 0.0);
    let uv : vec2<f32> = vec2<f32>((pos.x + 0.5) * inv_width, (pos.y + 0.5) * inv_height);
    let scaled : vec2<f32> = vec2<f32>(uv.x * freq.x, uv.y * freq.y);
    let temporal_offset : f32 = time_value * speed_value;
    let sample : vec3<f32> = vec3<f32>(
        scaled.x + seed.x,
        scaled.y + seed.y,
        temporal_offset + seed.z
    );
    let noise : f32 = simplex_noise(sample);
    return clamp(noise * 0.5 + 0.5, 0.0, 1.0);
}

fn worm_mask(
    pos : vec2<f32>,
    width : f32,
    height : f32,
    layer : u32,
    time_value : f32,
    speed_value : f32,
) -> f32 {
    let seed_base : f32 = 37.0 + f32(layer) * 41.0;
    let base_noise : f32 = animated_simplex(
        pos,
        width,
        height,
        4.0,
        vec3<f32>(seed_base * 0.17, seed_base * 0.29, seed_base * 0.41),
        time_value,
        speed_value,
    );

    let kink_rand : f32 = hash_11(seed_base * 1.13 + 13.0);
    let kink : f32 = 5.0 + floor(kink_rand * 6.0);
    let angle : f32 = base_noise * TAU * kink;
    let direction : vec2<f32> = vec2<f32>(sin(angle), cos(angle));

    let min_dim : f32 = max(min(width, height), 1.0);
    let iterations_raw : u32 = max(u32(floor(sqrt(min_dim))), 1u);
    let step_count : u32 = max(iterations_raw, 3u);
    let max_dim : f32 = max(width, height);
    let stride_base : f32 = 0.75 * (max_dim / 1024.0);
    let stride_rand : f32 = hash_11(seed_base * 3.73 + 17.0);
    let stride : f32 = stride_base * lerp(1.0 - 0.125, 1.0 + 0.125, stride_rand);

    let density_rand : f32 = hash_11(seed_base * 5.91 + 23.0);
    let density : f32 = 0.05 + density_rand * 0.00125;

    let span_steps : f32 = select(1.0, f32(step_count - 1u), step_count > 1u);
    let lateral_seed : f32 = seed_base * 7.17;

    var accum : f32 = 0.0;
    var weight_sum : f32 = 0.0;

    for (var i : u32 = 0u; i < step_count; i = i + 1u) {
        let denom : f32 = select(1.0, f32(step_count - 1u), step_count > 1u);
        let t : f32 = f32(i) / denom;
        let signed : f32 = t * 2.0 - 1.0;
        let exposure : f32 = 1.0 - abs(signed);
        let jitter : f32 = hash_11(lateral_seed + f32(i) * 0.937) - 0.5;
        let lateral : vec2<f32> = direction.yx * vec2<f32>(1.0, -1.0) * jitter * stride * 0.75;
        let offset_from_center : f32 = (f32(i) - span_steps * 0.5) * stride;
        let sample_pos : vec2<f32> = pos + direction * offset_from_center + lateral;
        let wrapped : vec2<f32> = vec2<f32>(
            wrap_float(sample_pos.x, width),
            wrap_float(sample_pos.y, height)
        );
        let sample_noise : f32 = animated_simplex(
            wrapped,
            width,
            height,
            4.0,
            vec3<f32>(seed_base * 0.47, seed_base * 0.61, seed_base * 0.83),
            time_value,
            speed_value,
        );
        accum = accum + sample_noise * exposure;
        weight_sum = weight_sum + exposure;
    }

    let normalized : f32 = select(0.0, accum / weight_sum, weight_sum > 0.0);
    let scaled : f32 = clamp(normalized * density * f32(step_count), 0.0, 1.0);
    return sqrt(scaled);
}

fn brightness_noise(
    pos : vec2<f32>,
    width : f32,
    height : f32,
    layer : u32,
    time_value : f32,
    speed_value : f32,
) -> vec4<f32> {
    let seed_base : f32 = 71.0 + f32(layer) * 53.0;
    let r : f32 = animated_simplex(
        pos,
        width,
        height,
        128.0,
        vec3<f32>(seed_base * 0.17, seed_base * 0.19, seed_base * 0.23),
        time_value,
        speed_value,
    );
    let g : f32 = animated_simplex(
        pos,
        width,
        height,
        128.0,
        vec3<f32>(seed_base * 0.31, seed_base * 0.37, seed_base * 0.41),
        time_value,
        speed_value,
    );
    let b : f32 = animated_simplex(
        pos,
        width,
        height,
        128.0,
        vec3<f32>(seed_base * 0.53, seed_base * 0.59, seed_base * 0.61),
        time_value,
        speed_value,
    );
    let a : f32 = animated_simplex(
        pos,
        width,
        height,
        128.0,
        vec3<f32>(seed_base * 0.73, seed_base * 0.79, seed_base * 0.83),
        time_value,
        speed_value,
    );
    return clamp(vec4<f32>(r, g, b, a), vec4<f32>(0.0), vec4<f32>(1.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = to_dimension(params.size.x);
    let height : u32 = to_dimension(params.size.y);
    if (width == 0u || height == 0u) {
        return;
    }
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let channels : u32 = channel_count(params.size.z);
    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let base_sample : vec4<f32> = textureLoad(input_texture, coords, 0);
    let base_alpha : f32 = base_sample.w;

    let width_f : f32 = f32(width);
    let height_f : f32 = f32(height);
    let pos : vec2<f32> = vec2<f32>(f32(gid.x), f32(gid.y));

    let time_value : f32 = params.time_speed.x;
    let speed_value : f32 = params.time_speed.y;

    var accum : vec4<f32> = base_sample;

    for (var layer : u32 = 0u; layer < 4u; layer = layer + 1u) {
        let mask_value : f32 = worm_mask(pos, width_f, height_f, layer, time_value, speed_value);
        let blend_alpha : f32 = clamp_01(mask_value * 0.5);
        if (blend_alpha <= 0.0) {
            continue;
        }

        let bright : vec4<f32> = brightness_noise(
            pos,
            width_f,
            height_f,
            layer,
            time_value,
            speed_value,
        );

        if (channels > 0u) {
            accum.x = lerp(accum.x, bright.x, blend_alpha);
        }
        if (channels > 1u) {
            accum.y = lerp(accum.y, bright.y, blend_alpha);
        }
        if (channels > 2u) {
            accum.z = lerp(accum.z, bright.z, blend_alpha);
        }
    }

    accum.w = base_alpha;

    let base_index : u32 = (gid.y * width + gid.x) * 4u;
    output_buffer[base_index + 0u] = clamp_01(accum.x);
    output_buffer[base_index + 1u] = clamp_01(accum.y);
    output_buffer[base_index + 2u] = clamp_01(accum.z);
    output_buffer[base_index + 3u] = clamp_01(base_alpha);
}
