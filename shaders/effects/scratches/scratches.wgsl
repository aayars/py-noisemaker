// Scratches overlay effect.
//
// Mirrors the Python implementation found in ``noisemaker/effects.py`` where
// four layers of animated value noise are run through the ``worms`` effect with
// randomized parameters, then brightened and composited back onto the source
// tensor. Each layer uses deterministic pseudo-random seeds so the output is
// temporally stable for a given ``time``/``speed`` pair while still animating.

const TAU : f32 = 6.28318530717958647692;

struct ScratchesParams {
    size : vec4<f32>,       // (width, height, channels, unused)
    time_speed : vec4<f32>, // (time, speed, unused, unused)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ScratchesParams;

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn hash11(x : f32) -> f32 {
    return fract(sin(x * 43758.5453123) * 43758.5453123);
}

fn hash13(p : vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453123);
}

fn random_float(seed : vec3<f32>) -> f32 {
    return hash13(seed);
}

fn random_int(seed : vec3<f32>, min_value : i32, max_value : i32) -> i32 {
    let low : i32 = min(min_value, max_value);
    let high : i32 = max(min_value, max_value);
    let span : i32 = (high - low) + 1;
    if (span <= 1) {
        return low;
    }
    let value : f32 = floor(random_float(seed) * f32(span));
    return low + clamp(i32(value), 0, span - 1);
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
        m0sq * m0sq * dot(g0n, x0) +
        m1sq * m1sq * dot(g1n, x1) +
        m2sq * m2sq * dot(g2n, x2) +
        m3sq * m3sq * dot(g3n, x3)
    );
}

fn freq_for_shape(base_freq : f32, width : f32, height : f32) -> vec2<f32> {
    let freq_value : f32 = max(base_freq, 1.0);
    if (abs(width - height) < 1e-5) {
        return vec2<f32>(freq_value, freq_value);
    }
    if (height < width && height > 0.0) {
        return vec2<f32>(freq_value * width / height, freq_value);
    }
    if (width > 0.0) {
        return vec2<f32>(freq_value, freq_value * height / width);
    }
    return vec2<f32>(freq_value, freq_value);
}

fn animated_value_noise(
    pos : vec2<f32>,
    freq : vec2<f32>,
    seed : vec3<f32>,
    time_value : f32,
    speed_value : f32
) -> f32 {
    let scaled : vec2<f32> = vec2<f32>(pos.x * freq.x, pos.y * freq.y);
    let phase : f32 = time_value * TAU * speed_value;
    let sample : vec3<f32> = vec3<f32>(scaled.x + seed.x, scaled.y + seed.y, phase + seed.z);
    let noise : f32 = simplex_noise(sample);
    return clamp(noise * 0.5 + 0.5, 0.0, 1.0);
}

fn worm_mask(
    uv : vec2<f32>,
    freq : vec2<f32>,
    width : f32,
    height : f32,
    seed : f32,
    behavior : u32,
    density : f32,
    duration : f32,
    kink : f32,
    stride : f32,
    stride_deviation : f32,
    time_value : f32,
    speed_value : f32
) -> f32 {
    let max_dim : f32 = max(max(width, height), 1.0);
    let min_dim : f32 = max(min(width, height), 1.0);
    let stride_base : f32 = stride / max_dim;
    let stride_jitter : f32 = stride_deviation / max_dim;
    let iteration_count : u32 = clamp(as_u32(sqrt(min_dim) * duration), 3u, 72u);
    let worm_count : u32 = clamp(as_u32(8.0 + density * 24.0), 2u, 48u);

    let align_factor : f32 = select(0.9, 0.6, behavior == 3u);
    let jitter_amount : f32 = select(0.12, 0.35, behavior == 3u);

    var result : f32 = 0.0;
    var worm_index : u32 = 0u;
    loop {
        if (worm_index >= worm_count) {
            break;
        }

        let base_seed : f32 = seed + f32(worm_index) * 41.31;
        worm_index = worm_index + 1u;

        let offset_seed : vec3<f32> = vec3<f32>(base_seed + 3.1, base_seed + 7.9, base_seed + 11.3);
        let start_offset : vec2<f32> = (vec2<f32>(
            random_float(offset_seed),
            random_float(offset_seed + vec3<f32>(5.7, 2.3, 9.1))
        ) - vec2<f32>(0.5)) * (0.2 + density * 0.8);
        var pos : vec2<f32> = fract(uv + start_offset);
        var heading : f32 = hash11(base_seed + 13.7) * TAU;

        var local_max : f32 = 0.0;
        var iteration : u32 = 0u;
        loop {
            if (iteration >= iteration_count) {
                break;
            }

            let step_seed : vec3<f32> = vec3<f32>(base_seed + f32(iteration) * 17.0);
            let flow_angle : f32 = animated_value_noise(
                pos,
                freq,
                step_seed,
                time_value,
                speed_value
            ) * TAU * kink;

            heading = mix(heading, flow_angle, align_factor);

            let jitter_noise : f32 = animated_value_noise(
                pos,
                freq * 1.7,
                step_seed + vec3<f32>(23.0, 47.0, 59.0),
                time_value,
                speed_value
            ) - 0.5;
            heading = heading + jitter_noise * jitter_amount * TAU;

            let stride_noise : f32 = animated_value_noise(
                pos,
                freq * 1.3,
                step_seed + vec3<f32>(101.0, 131.0, 197.0),
                time_value,
                speed_value
            ) - 0.5;
            let travel : f32 = stride_base + stride_noise * stride_jitter;
            pos = fract(pos + vec2<f32>(cos(heading), sin(heading)) * travel);

            let exposure : f32 = select(
                1.0,
                1.0 - abs(1.0 - (f32(iteration) / max(f32(iteration_count - 1u), 1.0)) * 2.0),
                iteration_count > 1u
            );

            let coverage : f32 = animated_value_noise(
                pos,
                freq,
                step_seed + vec3<f32>(211.0, 223.0, 251.0),
                time_value,
                speed_value
            );
            local_max = max(local_max, coverage * exposure);

            iteration = iteration + 1u;
        }

        result = max(result, local_max);
    }

    return clamp01(result);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let base_color : vec4<f32> = textureLoad(input_texture, coords, 0);

    let width_f : f32 = max(f32(dims.x), 1.0);
    let height_f : f32 = max(f32(dims.y), 1.0);
    let uv : vec2<f32> = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5, 0.5))
        / vec2<f32>(width_f, height_f);

    let time_value : f32 = params.time_speed.x;
    let speed_value : f32 = params.time_speed.y;

    var scratch_rgb : vec3<f32> = base_color.xyz;
    var layer_index : u32 = 0u;
    loop {
        if (layer_index >= 4u) {
            break;
        }

        let layer_seed : f32 = floor(time_value * 61.0 + speed_value * 37.0)
            + f32(layer_index) * 97.0 + 1.0;
        layer_index = layer_index + 1u;

        let freq_seed : vec3<f32> = vec3<f32>(
            layer_seed + 3.0,
            layer_seed + 7.0,
            layer_seed + 11.0
        );
        let freq_a : i32 = random_int(freq_seed, 2, 4);
        let freq_vec_a : vec2<f32> = freq_for_shape(f32(freq_a), width_f, height_f);

        let behavior_choice : u32 = select(
            1u,
            3u,
            random_int(freq_seed + vec3<f32>(19.0, 23.0, 29.0), 0, 1) == 1
        );
        let density : f32 = 0.25 + random_float(freq_seed + vec3<f32>(31.0, 37.0, 43.0)) * 0.25;
        let duration : f32 = 2.0 + random_float(freq_seed + vec3<f32>(47.0, 53.0, 59.0)) * 2.0;
        let kink : f32 = 0.125 + random_float(freq_seed + vec3<f32>(61.0, 67.0, 73.0)) * 0.125;

        let base_mask : f32 = animated_value_noise(
            uv,
            freq_vec_a,
            freq_seed,
            time_value,
            speed_value
        );

        let wormed_mask : f32 = worm_mask(
            uv,
            freq_vec_a,
            width_f,
            height_f,
            layer_seed,
            behavior_choice,
            density,
            duration,
            kink,
            0.75,
            0.5,
            time_value,
            speed_value
        );

        var mask_value : f32 = clamp01(mix(base_mask, wormed_mask, 0.8));

        let freq_b : i32 = random_int(freq_seed + vec3<f32>(79.0, 83.0, 89.0), 2, 4);
        let freq_vec_b : vec2<f32> = freq_for_shape(f32(freq_b), width_f, height_f);
        let subtract_noise : f32 = animated_value_noise(
            uv,
            freq_vec_b,
            freq_seed + vec3<f32>(97.0, 101.0, 103.0),
            time_value,
            speed_value
        );
        mask_value = max(mask_value - subtract_noise * 2.0, 0.0);

        let layer_contrib : f32 = clamp(mask_value * 8.0, 0.0, 1.0);
        scratch_rgb = max(scratch_rgb, vec3<f32>(layer_contrib));
        scratch_rgb = min(scratch_rgb, vec3<f32>(1.0));
    }

    let width_u : u32 = dims.x;
    let index : u32 = (gid.y * width_u + gid.x) * 4u;
    output_buffer[index + 0u] = scratch_rgb.x;
    output_buffer[index + 1u] = scratch_rgb.y;
    output_buffer[index + 2u] = scratch_rgb.z;
    output_buffer[index + 3u] = base_color.w;
}
