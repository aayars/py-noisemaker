// Ripple displaces pixels using angular offsets derived from a value map.
// Mirrors noisemaker.effects.ripple.

const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;

const INTERPOLATION_CONSTANT : u32 = 0u;
const INTERPOLATION_LINEAR : u32 = 1u;
const INTERPOLATION_COSINE : u32 = 2u;
const INTERPOLATION_BICUBIC : u32 = 3u;

struct RippleParams {
    dimensions_freq : vec4<f32>, // (width, height, channels, freq)
    effect : vec4<f32>,          // (displacement, kink, spline_order, time)
    animation : vec4<f32>,       // (speed, unused, unused, unused)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var output_texture : texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params : RippleParams;
@group(0) @binding(3) var reference_texture : texture_2d<f32>;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
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

fn wrap_index(coord : i32, limit : i32) -> i32 {
    return wrap_coord(coord, limit);
}

fn freq_for_shape(base_freq : f32, width : f32, height : f32) -> vec2<f32> {
    let freq_value : f32 = max(base_freq, 1.0);
    if (abs(width - height) < 1.0e-5) {
        return vec2<f32>(freq_value, freq_value);
    }
    if (height < width && height > 0.0) {
        return vec2<f32>(freq_value, freq_value * width / height);
    }
    if (width > 0.0) {
        return vec2<f32>(freq_value * height / width, freq_value);
    }
    return vec2<f32>(freq_value, freq_value);
}

fn srgb_to_linear(value : f32) -> f32 {
    if (value <= 0.04045) {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn oklab_l_component(rgb : vec3<f32>) -> f32 {
    let r_lin : f32 = srgb_to_linear(rgb.x);
    let g_lin : f32 = srgb_to_linear(rgb.y);
    let b_lin : f32 = srgb_to_linear(rgb.z);

    let l_val : f32 = 0.4121656120 * r_lin + 0.5362752080 * g_lin + 0.0514575653 * b_lin;
    let m_val : f32 = 0.2118591070 * r_lin + 0.6807189584 * g_lin + 0.1074065790 * b_lin;
    let s_val : f32 = 0.0883097947 * r_lin + 0.2818474174 * g_lin + 0.6302613616 * b_lin;

    let l_cbrt : f32 = pow(max(l_val, 0.0), 1.0 / 3.0);
    let m_cbrt : f32 = pow(max(m_val, 0.0), 1.0 / 3.0);
    let s_cbrt : f32 = pow(max(s_val, 0.0), 1.0 / 3.0);

    return 0.2104542553 * l_cbrt + 0.7936177850 * m_cbrt - 0.0040720468 * s_cbrt;
}

fn value_map_component(texel : vec4<f32>) -> f32 {
    let eps : f32 = 1.0e-5;
    let diff_xy : f32 = abs(texel.x - texel.y);
    let diff_xz : f32 = abs(texel.x - texel.z);
    let alpha : f32 = clamp_01(texel.w);
    let alpha_multiplier : f32 = select(1.0, alpha, alpha < 0.999);

    if (diff_xy < eps && diff_xz < eps) {
        return clamp_01(texel.x * alpha_multiplier);
    }

    let rgb : vec3<f32> = clamp(texel.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    let lum : f32 = clamp_01(oklab_l_component(rgb));
    return clamp_01(lum * alpha_multiplier);
}

fn hash_21(p : vec2<i32>) -> f32 {
    let pf : vec2<f32> = vec2<f32>(f32(p.x), f32(p.y));
    let dot_val : f32 = dot(pf, vec2<f32>(127.1, 311.7));
    return fract(sin(dot_val) * 43758.5453);
}

fn cosine_mix(a : f32, b : f32, t : f32) -> f32 {
    let weight : f32 = (1.0 - cos(clamp(t, 0.0, 1.0) * PI)) * 0.5;
    return mix(a, b, weight);
}

fn cubic_mix(a : f32, b : f32, c : f32, d : f32, t : f32) -> f32 {
    let clamped : f32 = clamp(t, 0.0, 1.0);
    let t2 : f32 = clamped * clamped;
    let a0 : f32 = (d - c) - (a - b);
    let a1 : f32 = (a - b) - a0;
    let a2 : f32 = c - a;
    let a3 : f32 = b;
    return ((a0 * clamped) * t2) + (a1 * t2) + (a2 * clamped) + a3;
}

fn lattice_value(coord : vec2<i32>, freq : vec2<i32>) -> f32 {
    let wrapped_x : i32 = wrap_index(coord.x, max(freq.x, 1));
    let wrapped_y : i32 = wrap_index(coord.y, max(freq.y, 1));
    return hash_21(vec2<i32>(wrapped_x, wrapped_y));
}

fn sample_value_field(sample_pos : vec2<f32>, freq : vec2<i32>, spline_order : u32) -> f32 {
    let freq_x : i32 = max(freq.x, 1);
    let freq_y : i32 = max(freq.y, 1);
    let base_floor : vec2<f32> = floor(sample_pos);
    let base_coord : vec2<i32> = vec2<i32>(i32(base_floor.x), i32(base_floor.y));
    let frac : vec2<f32> = sample_pos - base_floor;

    let x0 : i32 = wrap_index(base_coord.x, freq_x);
    let y0 : i32 = wrap_index(base_coord.y, freq_y);

    if (spline_order == INTERPOLATION_CONSTANT) {
        return lattice_value(vec2<i32>(x0, y0), freq);
    }

    let x1 : i32 = wrap_index(x0 + 1, freq_x);
    let y1 : i32 = wrap_index(y0 + 1, freq_y);

    let v00 : f32 = lattice_value(vec2<i32>(x0, y0), freq);
    let v10 : f32 = lattice_value(vec2<i32>(x1, y0), freq);
    let v01 : f32 = lattice_value(vec2<i32>(x0, y1), freq);
    let v11 : f32 = lattice_value(vec2<i32>(x1, y1), freq);

    if (spline_order == INTERPOLATION_LINEAR) {
        let xa : f32 = mix(v00, v10, frac.x);
        let xb : f32 = mix(v01, v11, frac.x);
        return mix(xa, xb, frac.y);
    }

    if (spline_order == INTERPOLATION_COSINE) {
        let xa : f32 = cosine_mix(v00, v10, frac.x);
        let xb : f32 = cosine_mix(v01, v11, frac.x);
        return cosine_mix(xa, xb, frac.y);
    }

    var rows : array<f32, 4>;
    for (var i : i32 = -1; i <= 2; i = i + 1) {
        let sample_y : i32 = wrap_index(y0 + i, freq_y);
        var cols : array<f32, 4>;
        for (var j : i32 = -1; j <= 2; j = j + 1) {
            let sample_x : i32 = wrap_index(x0 + j, freq_x);
            cols[u32(j + 1)] = lattice_value(vec2<i32>(sample_x, sample_y), freq);
        }
        rows[u32(i + 1)] = cubic_mix(cols[0], cols[1], cols[2], cols[3], frac.x);
    }

    return cubic_mix(rows[0], rows[1], rows[2], rows[3], frac.y);
}

fn sanitize_spline_order(raw_value : f32) -> u32 {
    let rounded : i32 = i32(round(raw_value));
    if (rounded <= 0) {
        return INTERPOLATION_CONSTANT;
    }
    if (rounded == 1) {
        return INTERPOLATION_LINEAR;
    }
    if (rounded == 2) {
        return INTERPOLATION_COSINE;
    }
    return INTERPOLATION_BICUBIC;
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
    let p : vec4<f32> = permute(permute(permute(
        i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0))
        + i.y + vec4<f32>(0.0, i1.y, i2.y, 1.0))
        + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0));

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
        m0sq * m0sq * dot(g0n, x0) +
        m1sq * m1sq * dot(g1n, x1) +
        m2sq * m2sq * dot(g2n, x2) +
        m3sq * m3sq * dot(g3n, x3)
    );
}

fn simplex_random(time : f32, speed : f32) -> f32 {
    let angle : f32 = time * TAU;
    let z : f32 = cos(angle) * speed;
    let w : f32 = sin(angle) * speed;
    let noise_value : f32 = simplex_noise(vec3<f32>(z + 17.0, w + 29.0, 11.0));
    return clamp(noise_value * 0.5 + 0.5, 0.0, 1.0);
}

fn reference_value(coord : vec2<u32>, width : f32, height : f32, freq_param : f32, spline_order : u32) -> f32 {
    if (freq_param > 0.0) {
        let freq_vec : vec2<f32> = freq_for_shape(freq_param, width, height);
        let freq_int : vec2<i32> = vec2<i32>(
            max(i32(round(freq_vec.x)), 1),
            max(i32(round(freq_vec.y)), 1)
        );

        let width_safe : f32 = max(width, 1.0);
        let height_safe : f32 = max(height, 1.0);
        let uv : vec2<f32> = vec2<f32>(
            f32(coord.x) / width_safe,
            f32(coord.y) / height_safe
        );

        let sample_pos : vec2<f32> = vec2<f32>(
            uv.x * f32(freq_int.x),
            uv.y * f32(freq_int.y)
        );

        return clamp_01(sample_value_field(sample_pos, freq_int, spline_order));
    }

    let texel : vec4<f32> = textureLoad(reference_texture, vec2<i32>(i32(coord.x), i32(coord.y)), 0);
    return value_map_component(texel);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.dimensions_freq.x);
    let height : u32 = as_u32(params.dimensions_freq.y);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let width_f : f32 = max(params.dimensions_freq.x, 1.0);
    let height_f : f32 = max(params.dimensions_freq.y, 1.0);

    let displacement : f32 = params.effect.x;
    let kink : f32 = params.effect.y;
    let spline_order : u32 = sanitize_spline_order(params.effect.z);
    let time_value : f32 = params.effect.w;
    let speed_value : f32 = params.animation.x;
    let freq_param : f32 = params.dimensions_freq.w;

    let ref_value : f32 = reference_value(gid.xy, width_f, height_f, freq_param, spline_order);
    let random_factor : f32 = simplex_random(time_value, speed_value);

    let angle : f32 = ref_value * TAU * kink * random_factor;

    let offset_x : f32 = cos(angle) * displacement * width_f;
    let offset_y : f32 = sin(angle) * displacement * height_f;

    let sample_x : f32 = f32(gid.x) + offset_x;
    let sample_y : f32 = f32(gid.y) + offset_y;

    let x0 : i32 = i32(floor(sample_x));
    let y0 : i32 = i32(floor(sample_y));
    let x1 : i32 = x0 + 1;
    let y1 : i32 = y0 + 1;

    let frac_x : f32 = sample_x - f32(x0);
    let frac_y : f32 = sample_y - f32(y0);

    let width_i : i32 = i32(width);
    let height_i : i32 = i32(height);

    let x0w : i32 = wrap_coord(x0, width_i);
    let x1w : i32 = wrap_coord(x1, width_i);
    let y0w : i32 = wrap_coord(y0, height_i);
    let y1w : i32 = wrap_coord(y1, height_i);

    let coord00 : vec2<i32> = vec2<i32>(x0w, y0w);
    let coord10 : vec2<i32> = vec2<i32>(x1w, y0w);
    let coord01 : vec2<i32> = vec2<i32>(x0w, y1w);
    let coord11 : vec2<i32> = vec2<i32>(x1w, y1w);

    let c00 : vec4<f32> = textureLoad(input_texture, coord00, 0);
    let c10 : vec4<f32> = textureLoad(input_texture, coord10, 0);
    let c01 : vec4<f32> = textureLoad(input_texture, coord01, 0);
    let c11 : vec4<f32> = textureLoad(input_texture, coord11, 0);

    let mix_x0 : vec4<f32> = mix(c00, c10, vec4<f32>(frac_x));
    let mix_x1 : vec4<f32> = mix(c01, c11, vec4<f32>(frac_x));
    let interpolated : vec4<f32> = mix(mix_x0, mix_x1, vec4<f32>(frac_y));

    textureStore(output_texture, vec2<i32>(i32(gid.x), i32(gid.y)), interpolated);
}
