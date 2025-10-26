// False Color effect.
// Reproduces Noisemaker's false_color by generating a procedural color lookup table
// and remapping luminance-driven coordinates into that palette.

const CHANNEL_COUNT : u32 = 4u;
const TAU : f32 = 6.283185307179586;
const F32_MAX : f32 = 0x1.fffffep+127;
const F32_MIN : f32 = -0x1.fffffep+127;

struct FalseColorParams {
    width : f32,
    height : f32,
    channels : f32,
    horizontal : f32,
    displacement : f32,
    time : f32,
    speed : f32,
    _pad0 : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : FalseColorParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn sanitized_channel_count(channel_value : f32) -> u32 {
    let rounded : i32 = i32(round(channel_value));
    if (rounded <= 1) {
        return 1u;
    }
    if (rounded >= 4) {
        return CHANNEL_COUNT;
    }
    return u32(rounded);
}

fn wrap_coord(value : i32, extent : i32) -> i32 {
    if (extent <= 0) {
        return 0;
    }
    var wrapped : i32 = value % extent;
    if (wrapped < 0) {
        wrapped = wrapped + extent;
    }
    return wrapped;
}

fn srgb_to_linear(value : f32) -> f32 {
    if (value <= 0.04045) {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn cbrt(value : f32) -> f32 {
    if (value == 0.0) {
        return 0.0;
    }
    let sign_value : f32 = select(-1.0, 1.0, value >= 0.0);
    return sign_value * pow(abs(value), 1.0 / 3.0);
}

fn oklab_l_component(rgb : vec3<f32>) -> f32 {
    let r : f32 = srgb_to_linear(clamp01(rgb.x));
    let g : f32 = srgb_to_linear(clamp01(rgb.y));
    let b : f32 = srgb_to_linear(clamp01(rgb.z));

    let l : f32 = 0.4121656120 * r + 0.5362752080 * g + 0.0514575653 * b;
    let m : f32 = 0.2118591070 * r + 0.6807189584 * g + 0.1074065790 * b;
    let s : f32 = 0.0883097947 * r + 0.2818474174 * g + 0.6302613616 * b;

    let l_c : f32 = cbrt(l);
    let m_c : f32 = cbrt(m);
    let s_c : f32 = cbrt(s);

    return clamp01(0.2104542553 * l_c + 0.7936177850 * m_c - 0.0040720468 * s_c);
}

fn value_map_component(texel : vec4<f32>, channel_count : u32) -> f32 {
    if (channel_count <= 2u) {
        return clamp01(texel.x);
    }
    return oklab_l_component(vec3<f32>(texel.x, texel.y, texel.z));
}

fn freq_for_shape(base_freq : f32, dims : vec2<f32>) -> vec2<f32> {
    let width : f32 = max(dims.x, 1.0);
    let height : f32 = max(dims.y, 1.0);

    if (height == width) {
        return vec2<f32>(base_freq, base_freq);
    }

    if (height < width) {
        let freq_y : f32 = base_freq;
        let freq_x : f32 = max(1.0, floor(base_freq * width / height));
        return vec2<f32>(freq_x, freq_y);
    }

    let freq_x : f32 = base_freq;
    let freq_y : f32 = max(1.0, floor(base_freq * height / width));
    return vec2<f32>(freq_x, freq_y);
}

fn mod289_vec3(value : vec3<f32>) -> vec3<f32> {
    return value - floor(value * (1.0 / 289.0)) * 289.0;
}

fn mod289_vec4(value : vec4<f32>) -> vec4<f32> {
    return value - floor(value * (1.0 / 289.0)) * 289.0;
}

fn permute4(value : vec4<f32>) -> vec4<f32> {
    return mod289_vec4(((value * 34.0) + 1.0) * value);
}

fn taylor_inv_sqrt4(value : vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * value;
}

fn simplex_noise(coord : vec3<f32>) -> f32 {
    let c : vec2<f32> = vec2<f32>(1.0 / 6.0, 1.0 / 3.0);
    let d : vec4<f32> = vec4<f32>(0.0, 0.5, 1.0, 2.0);

    let i0 : vec3<f32> = floor(coord + dot(coord, vec3<f32>(c.y)));
    let x0 : vec3<f32> = coord - i0 + dot(i0, vec3<f32>(c.x));

    let step1 : vec3<f32> = step(vec3<f32>(x0.y, x0.z, x0.x), x0);
    let l : vec3<f32> = vec3<f32>(1.0) - step1;
    let i1 : vec3<f32> = min(step1, vec3<f32>(l.z, l.x, l.y));
    let i2 : vec3<f32> = max(step1, vec3<f32>(l.z, l.x, l.y));

    let x1 : vec3<f32> = x0 - i1 + vec3<f32>(c.x);
    let x2 : vec3<f32> = x0 - i2 + vec3<f32>(c.y);
    let x3 : vec3<f32> = x0 - vec3<f32>(d.y);

    let i = mod289_vec3(i0);
    let p = permute4(permute4(permute4(
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

    let norm : vec4<f32> = taylor_inv_sqrt4(vec4<f32>(
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

fn normalized_sine(value : f32) -> f32 {
    return (sin(value) + 1.0) * 0.5;
}

fn periodic_value(time_value : f32, sample_value : f32) -> f32 {
    return normalized_sine((time_value - sample_value) * TAU);
}

const SIMPLEX_OFFSETS : array<vec3<f32>, 4> = array<vec3<f32>, 4>(
    vec3<f32>(37.0, 17.0, 53.0),
    vec3<f32>(71.0, 29.0, 97.0),
    vec3<f32>(113.0, 47.0, 151.0),
    vec3<f32>(157.0, 67.0, 211.0)
);

const TIME_SIMPLEX_OFFSETS : array<vec3<f32>, 4> = array<vec3<f32>, 4>(
    vec3<f32>(193.0, 131.0, 271.0),
    vec3<f32>(233.0, 163.0, 313.0),
    vec3<f32>(271.0, 197.0, 353.0),
    vec3<f32>(313.0, 229.0, 397.0)
);

fn sample_simplex(coord : vec3<f32>) -> f32 {
    return simplex_noise(coord) * 0.5 + 0.5;
}

fn generate_clut_color(
    coord : vec2<i32>,
    freq : vec2<f32>,
    dims : vec2<f32>,
    time_value : f32,
    speed_value : f32
) -> vec4<f32> {
    let width : f32 = max(dims.x, 1.0);
    let height : f32 = max(dims.y, 1.0);
    let uv : vec2<f32> = (vec2<f32>(f32(coord.x), f32(coord.y)) + vec2<f32>(0.5, 0.5))
        / vec2<f32>(width, height);
    let scaled : vec2<f32> = vec2<f32>(uv.x * freq.x, uv.y * freq.y);

    let angle : f32 = time_value * TAU;
    let z_base : f32 = cos(angle) * speed_value;
    let base : vec3<f32> = vec3<f32>(scaled.x, scaled.y, z_base);
    let time_base : vec3<f32> = vec3<f32>(scaled.x, scaled.y, 1.0);
    let animate : bool = (speed_value != 0.0) && (time_value != 0.0);

    var base_noise : array<f32, CHANNEL_COUNT>;
    var time_noise : array<f32, CHANNEL_COUNT>;

    for (var channel : u32 = 0u; channel < CHANNEL_COUNT; channel = channel + 1u) {
        let offset : vec3<f32> = SIMPLEX_OFFSETS[channel];
        base_noise[channel] = clamp01(sample_simplex(base + offset));

        if (animate) {
            let time_offset : vec3<f32> = TIME_SIMPLEX_OFFSETS[channel];
            time_noise[channel] = clamp01(sample_simplex(time_base + time_offset));
        }
    }

    var values : array<f32, CHANNEL_COUNT>;
    for (var channel : u32 = 0u; channel < CHANNEL_COUNT; channel = channel + 1u) {
        var value : f32 = base_noise[channel];
        if (animate) {
            let scaled_time : f32 = periodic_value(time_value, time_noise[channel]) * speed_value;
            value = periodic_value(scaled_time, value);
        }
        values[channel] = clamp01(value);
    }

    // Force alpha to 1.0 - do not generate transparency
    return vec4<f32>(values[0], values[1], values[2], 1.0);
}

fn update_color_range(color : vec4<f32>, min_value : ptr<function, f32>, max_value : ptr<function, f32>) {
    *min_value = min(*min_value, color.x);
    *max_value = max(*max_value, color.x);
    *min_value = min(*min_value, color.y);
    *max_value = max(*max_value, color.y);
    *min_value = min(*min_value, color.z);
    *max_value = max(*max_value, color.z);
    *min_value = min(*min_value, color.w);
    *max_value = max(*max_value, color.w);
}

fn write_pixel(base_index : u32, color : vec4<f32>) {
    output_buffer[base_index + 0u] = color.x;
    output_buffer[base_index + 1u] = color.y;
    output_buffer[base_index + 2u] = color.z;
    output_buffer[base_index + 3u] = color.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.width);
    let height : u32 = as_u32(params.height);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let channel_count : u32 = sanitized_channel_count(params.channels);
    let displacement : f32 = params.displacement;
    let horizontal : bool = params.horizontal >= 0.5;
    let time_value : f32 = params.time;
    let speed_value : f32 = params.speed;

    let width_i : i32 = i32(width);
    let height_i : i32 = i32(height);
    let max_x_offset : f32 = f32(max(width_i - 1, 0));
    let max_y_offset : f32 = f32(max(height_i - 1, 0));
    let dims : vec2<f32> = vec2<f32>(f32(width), f32(height));
    let freq : vec2<f32> = freq_for_shape(2.0, dims);

    let coord : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let texel : vec4<f32> = textureLoad(input_texture, coord, 0);
    let reference_raw : f32 = value_map_component(texel, channel_count);
    let normalized : f32 = clamp01(reference_raw);
    let reference : f32 = normalized * displacement;

    // Python reference: x_index always gets offset
    let offset_x : i32 = i32(reference * max_x_offset);
    let sample_x : i32 = wrap_coord(coord.x + offset_x, width_i);

    // Python reference: y_index gets offset only when !horizontal
    var sample_y : i32 = coord.y;
    if (horizontal) {
        // When horizontal=true, y is just the current row (no offset)
        sample_y = coord.y;
    } else {
        // When horizontal=false, y gets offset
        let offset_y : i32 = i32(reference * max_y_offset);
        sample_y = wrap_coord(coord.y + offset_y, height_i);
    }

    let clut_color : vec4<f32> = generate_clut_color(vec2<i32>(sample_x, sample_y), freq, dims, time_value, speed_value);

    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;
    write_pixel(base_index, clut_color);
}
