// Fibers final combine pass.
// Reuses the worms low-level pipelines to paint a chaotic mask into
// `worm_texture`, then synthesizes animated brightness noise that is blended
// back over the input image. The heavy lifting (worm simulation) lives in the
// dedicated worms shaders; this pass is only responsible for layering the
// brightness streaks according to that mask.

struct FibersParams {
    width : f32,
    height : f32,
    channel_count : f32,
    mask_scale : f32,
    time : f32,
    speed : f32,
    seed : f32,
    _pad0 : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var worm_texture : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(3) var<uniform> params : FibersParams;

const TAU : f32 = 6.28318530717958647692;
const CHANNEL_COUNT : u32 = 4u;
const GOLDEN_ANGLE : f32 = 2.39996322972865332223;

fn to_dimension(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
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

    let g = step(vec3<f32>(x0.y, x0.z, x0.x), x0);
    let l = vec3<f32>(1.0) - g;
    let i1 = min(g, vec3<f32>(l.z, l.x, l.y));
    let i2 = max(g, vec3<f32>(l.z, l.x, l.y));

    let x1 = x0 - i1 + vec3<f32>(c.x);
    let x2 = x0 - i2 + vec3<f32>(c.y);
    let x3 = x0 - vec3<f32>(d.y);

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

    let g0n = g0 * norm.x;
    let g1n = g1 * norm.y;
    let g2n = g2 * norm.z;
    let g3n = g3 * norm.w;

    let m0 : f32 = max(0.6 - dot(x0, x0), 0.0);
    let m1 : f32 = max(0.6 - dot(x1, x1), 0.0);
    let m2 : f32 = max(0.6 - dot(x2, x2), 0.0);
    let m3 : f32 = max(0.6 - dot(x3, x3), 0.0);

    let m0sq = m0 * m0;
    let m1sq = m1 * m1;
    let m2sq = m2 * m2;
    let m3sq = m3 * m3;

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

fn brightness_noise(
    pos : vec2<f32>,
    width : f32,
    height : f32,
    seed_offset : f32,
    time_value : f32,
    speed_value : f32,
) -> vec3<f32> {
    let base : f32 = 71.0 + seed_offset * 53.0;
    let r = animated_simplex(pos, width, height, 128.0, vec3<f32>(base * 0.17, base * 0.23, base * 0.31), time_value, speed_value);
    let g = animated_simplex(pos, width, height, 128.0, vec3<f32>(base * 0.41, base * 0.47, base * 0.53), time_value, speed_value);
    let b = animated_simplex(pos, width, height, 128.0, vec3<f32>(base * 0.59, base * 0.61, base * 0.67), time_value, speed_value);
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn mask_strength(mask : vec4<f32>) -> f32 {
    let rgb_strength : f32 = max(mask.x, max(mask.y, mask.z));
    return max(rgb_strength, mask.w);
}

fn clamp_coords(coords : vec2<i32>, dims : vec2<i32>) -> vec2<i32> {
    let max_x : i32 = max(dims.x - 1, 0);
    let max_y : i32 = max(dims.y - 1, 0);
    return vec2<i32>(
        clamp(coords.x, 0, max_x),
        clamp(coords.y, 0, max_y)
    );
}

fn worm_mask_sample(coords : vec2<i32>) -> vec4<f32> {
    let worm_sample : vec4<f32> = textureLoad(worm_texture, coords, 0);
    return clamp(worm_sample, vec4<f32>(0.0), vec4<f32>(1.0));
}

fn worm_mask_accumulated(
    coords : vec2<i32>,
    dims : vec2<i32>,
    time_value : f32,
    speed_value : f32,
) -> vec4<f32> {
    // Sample the worm texture directly and normalize
    let worm_sample : vec4<f32> = worm_mask_sample(coords);
    let sqrt_rgb : vec3<f32> = sqrt(clamp(worm_sample.xyz, vec3<f32>(0.0), vec3<f32>(1.0)));
    let sqrt_alpha : f32 = sqrt(clamp(worm_sample.w, 0.0, 1.0));
    // Boost alpha so fresh worm trails reach full opacity quickly, aging out older ones
    let boosted_alpha : f32 = clamp(sqrt_alpha * 100.0, 0.0, 1.0);
    return vec4<f32>(sqrt_rgb, boosted_alpha);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = to_dimension(params.width);
    let height : u32 = to_dimension(params.height);
    if (width == 0u || height == 0u) {
        return;
    }
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;

    let base_sample : vec4<f32> = textureLoad(input_texture, coords, 0);

    let mask_dims_u : vec2<u32> = textureDimensions(worm_texture, 0);
    let mask_dims : vec2<i32> = vec2<i32>(i32(mask_dims_u.x), i32(mask_dims_u.y));
    let speed_value : f32 = max(params.speed, 0.0);
    let time_value : f32 = params.time;
    let base_mask : vec4<f32> = worm_mask_accumulated(coords, mask_dims, time_value, speed_value);
    let mask_power : f32 = mask_strength(base_mask);

    var accum : vec3<f32> = base_sample.xyz;
    let pos : vec2<f32> = vec2<f32>(f32(gid.x), f32(gid.y));
    let width_f : f32 = f32(width);
    let height_f : f32 = f32(height);

    // Python: tensor = value.blend(tensor, brightness, mask * 0.5)
    // Each layer uses the same worm mask sampled at the current pixel,
    // but with different brightness noise seeds and temporal offsets.
    for (var layer : u32 = 0u; layer < 4u; layer = layer + 1u) {
        let layer_seed : f32 = params.seed + f32(layer) * 17.0;
        
        // Sample brightness noise with per-layer temporal variation
        let brightness : vec3<f32> = brightness_noise(
            pos,
            width_f,
            height_f,
            layer_seed,
            time_value + f32(layer) * 0.13,
            1.0 + speed_value * 0.5,
        );
        
        // Use the worm mask directly (Python: mask * 0.5)
        let mask_blend : vec3<f32> = clamp(base_mask.xyz * 0.5, vec3<f32>(0.0), vec3<f32>(1.0));
        
        // Linear blend: accum = mix(accum, brightness, mask)
        accum = mix(accum, brightness, mask_blend);
    }

    // Blend the accumulated layers back to the source
    let final_rgb : vec3<f32> = clamp(accum, vec3<f32>(0.0), vec3<f32>(1.0));

    output_buffer[base_index + 0u] = final_rgb.x;
    output_buffer[base_index + 1u] = final_rgb.y;
    output_buffer[base_index + 2u] = final_rgb.z;
    output_buffer[base_index + 3u] = base_sample.w;
}
