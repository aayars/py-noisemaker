// Chromatic aberration effect mirroring Noisemaker's aberration() implementation.
// Applies a hue jitter, offsets RGB channels horizontally, and blends offsets
// toward the image center with a cosine falloff mask.

const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;

struct AberrationParams {
    size : vec4<f32>,      // (width, height, channels, displacement)
    anim : vec4<f32>,      // (time, speed, unused, unused)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : AberrationParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp_01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn wrap_unit(value : f32) -> f32 {
    let wrapped : f32 = value - floor(value);
    if (wrapped < 0.0) {
        return wrapped + 1.0;
    }
    return wrapped;
}

fn mod289_vec3(x : vec3<f32>) -> vec3<f32> {
    return x - floor(x / 289.0) * 289.0;
}

fn mod289_vec4(x : vec4<f32>) -> vec4<f32> {
    return x - floor(x / 289.0) * 289.0;
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

    let i : vec3<f32> = mod289_vec3(i0);
    let p : vec4<f32> = permute(
        permute(
            permute(i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0))
            + i.y + vec4<f32>(0.0, i1.y, i2.y, 1.0)
        )
        + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0)
    );

    let n_ : f32 = 0.14285714285714285;
    let ns : vec3<f32> = n_ * vec3<f32>(d.w, d.y, d.z) - vec3<f32>(d.x, d.z, d.x);

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

fn slow_motion_rate(speed : f32) -> f32 {
    let clamped_speed : f32 = max(speed, 0.0);
    let normalized : f32 = clamp(clamped_speed, 0.0, 1.0);
    let extended : f32 = max(clamped_speed - 1.0, 0.0);
    return normalized * 0.2 + extended * 0.05;
}

fn gentle_noise(time : f32, speed : f32, offset : vec3<f32>) -> f32 {
    let rate : f32 = slow_motion_rate(speed);
    let phase : f32 = time * TAU * rate;
    let sample : vec3<f32> = offset + vec3<f32>(
        phase * 0.11,
        phase * 0.17,
        phase * 0.23
    );
    let noise_value : f32 = simplex_noise(sample);
    return clamp(noise_value * 0.5 + 0.5, 0.0, 1.0);
}

fn blend_linear(a : f32, b : f32, t : f32) -> f32 {
    return a * (1.0 - t) + b * t;
}

fn blend_cosine(a : f32, b : f32, g : f32) -> f32 {
    let weight : f32 = (1.0 - cos(g * PI)) * 0.5;
    return a * (1.0 - weight) + b * weight;
}

fn clamp_index(value : f32, max_index : f32) -> u32 {
    if (max_index <= 0.0) {
        return 0u;
    }
    let clamped_value : f32 = clamp(value, 0.0, max_index);
    return u32(clamped_value);
}

fn aberration_mask(width : f32, height : f32, x : f32, y : f32) -> f32 {
    if (width <= 0.0 || height <= 0.0) {
        return 0.0;
    }
    let px : f32 = x + 0.5;
    let py : f32 = y + 0.5;
    let half_w : f32 = width * 0.5;
    let half_h : f32 = height * 0.5;
    let dx : f32 = (px - half_w) / width;
    let dy : f32 = (py - half_h) / height;
    let max_dx : f32 = abs((half_w - 0.5) / width);
    let max_dy : f32 = abs((half_h - 0.5) / height);
    let max_dist : f32 = sqrt(max_dx * max_dx + max_dy * max_dy);
    if (max_dist <= 0.0) {
        return 0.0;
    }
    let dist : f32 = sqrt(dx * dx + dy * dy);
    let normalized : f32 = clamp(dist / max_dist, 0.0, 1.0);
    return pow(normalized, 3.0);
}

fn rgb_to_hsv(rgb : vec3<f32>) -> vec3<f32> {
    let c_max : f32 = max(max(rgb.x, rgb.y), rgb.z);
    let c_min : f32 = min(min(rgb.x, rgb.y), rgb.z);
    let delta : f32 = c_max - c_min;

    var hue : f32 = 0.0;
    if (delta > 0.0) {
        if (c_max == rgb.x) {
            var segment : f32 = (rgb.y - rgb.z) / delta;
            if (segment < 0.0) {
                segment = segment + 6.0;
            }
            hue = segment;
        } else if (c_max == rgb.y) {
            hue = ((rgb.z - rgb.x) / delta) + 2.0;
        } else {
            hue = ((rgb.x - rgb.y) / delta) + 4.0;
        }
        hue = wrap_unit(hue / 6.0);
    }

    let saturation : f32 = select(0.0, delta / c_max, c_max != 0.0);
    return vec3<f32>(hue, saturation, c_max);
}

fn hsv_to_rgb(hsv : vec3<f32>) -> vec3<f32> {
    let h : f32 = hsv.x;
    let s : f32 = hsv.y;
    let v : f32 = hsv.z;

    let dh : f32 = h * 6.0;
    let r_comp : f32 = clamp_01(abs(dh - 3.0) - 1.0);
    let g_comp : f32 = clamp_01(-abs(dh - 2.0) + 2.0);
    let b_comp : f32 = clamp_01(-abs(dh - 4.0) + 2.0);

    let one_minus_s : f32 = 1.0 - s;
    let sr : f32 = s * r_comp;
    let sg : f32 = s * g_comp;
    let sb : f32 = s * b_comp;

    let r : f32 = clamp_01((one_minus_s + sr) * v);
    let g : f32 = clamp_01((one_minus_s + sg) * v);
    let b : f32 = clamp_01((one_minus_s + sb) * v);

    return vec3<f32>(r, g, b);
}

fn adjust_hue(rgb : vec3<f32>, amount : f32) -> vec3<f32> {
    var hsv : vec3<f32> = rgb_to_hsv(rgb);
    hsv.x = wrap_unit(hsv.x + amount);
    hsv.y = clamp_01(hsv.y);
    hsv.z = clamp_01(hsv.z);
    return clamp(vec3<f32>(hsv_to_rgb(hsv)), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn sample_shifted(coords : vec2<i32>, hue_shift : f32) -> vec4<f32> {
    let texel : vec4<f32> = textureLoad(input_texture, coords, 0);
    let adjusted_rgb : vec3<f32> = adjust_hue(texel.xyz, hue_shift);
    return vec4<f32>(adjusted_rgb, texel.w);
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

    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * 4u;

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let center_sample : vec4<f32> = textureLoad(input_texture, coords, 0);

    let channel_count : u32 = as_u32(params.size.z);
    if (channel_count < 3u) {
        output_buffer[base_index + 0u] = center_sample.x;
        output_buffer[base_index + 1u] = center_sample.y;
        output_buffer[base_index + 2u] = center_sample.z;
        output_buffer[base_index + 3u] = center_sample.w;
        return;
    }

    let width_f : f32 = max(params.size.x, 1.0);
    let height_f : f32 = max(params.size.y, 1.0);
    let x_float : f32 = f32(gid.x);
    let y_float : f32 = f32(gid.y);
    let width_minus_one : f32 = max(width_f - 1.0, 0.0);

    var gradient : f32 = 0.0;
    if (width > 1u) {
        gradient = x_float / width_minus_one;
    }

    let time_value : f32 = params.anim.x;
    let speed_value : f32 = params.anim.y;
    let speed_weight : f32 = clamp(speed_value, 0.0, 1.0);
    let base_noise : f32 = gentle_noise(time_value, speed_value, vec3<f32>(17.0, 29.0, 11.0));
    let random_factor : f32 = blend_linear(0.5, base_noise, speed_weight);

    let hue_noise : f32 = gentle_noise(time_value + 0.37, speed_value, vec3<f32>(23.0, 47.0, 19.0));
    let hue_shift : f32 = (hue_noise - 0.5) * 0.06;

    let displacement_raw : f32 = width_f * params.size.w * random_factor;
    let displacement_pixels : f32 = trunc(displacement_raw);

    let mask_value : f32 = aberration_mask(width_f, height_f, x_float, y_float);

    var red_offset : f32 = min(x_float + displacement_pixels, width_minus_one);
    red_offset = blend_linear(red_offset, x_float, gradient);
    red_offset = blend_cosine(x_float, red_offset, mask_value);
    let red_x : u32 = clamp_index(red_offset, width_minus_one);

    var green_offset : f32 = x_float;
    green_offset = blend_cosine(x_float, green_offset, mask_value);
    let green_x : u32 = clamp_index(green_offset, width_minus_one);

    var blue_offset : f32 = max(x_float - displacement_pixels, 0.0);
    blue_offset = blend_linear(x_float, blue_offset, gradient);
    blue_offset = blend_cosine(x_float, blue_offset, mask_value);
    let blue_x : u32 = clamp_index(blue_offset, width_minus_one);

    let red_sample : vec4<f32> = sample_shifted(vec2<i32>(i32(red_x), i32(gid.y)), hue_shift);
    let green_sample : vec4<f32> = sample_shifted(vec2<i32>(i32(green_x), i32(gid.y)), hue_shift);
    let blue_sample : vec4<f32> = sample_shifted(vec2<i32>(i32(blue_x), i32(gid.y)), hue_shift);

    let combined_rgb : vec3<f32> = vec3<f32>(red_sample.x, green_sample.y, blue_sample.z);
    let restored_rgb : vec3<f32> = adjust_hue(combined_rgb, -hue_shift);

    output_buffer[base_index + 0u] = clamp_01(restored_rgb.x);
    output_buffer[base_index + 1u] = clamp_01(restored_rgb.y);
    output_buffer[base_index + 2u] = clamp_01(restored_rgb.z);
    output_buffer[base_index + 3u] = center_sample.w;
}
