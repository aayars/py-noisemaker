// Refract: displacement driven by luminance-derived offsets, matching value.refract().

const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;
const FLOAT_EPSILON : f32 = 1e-5;

struct RefractParams {
    width : f32,
    height : f32,
    channel_count : f32,
    displacement : f32,
    warp : f32,
    spline_order : f32,
    derivative : f32,
    unused0 : f32,
    range : f32,
    time : f32,
    speed : f32,
    _pad0 : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : RefractParams;
@group(0) @binding(3) var reference_x_texture : texture_2d<f32>;
@group(0) @binding(4) var reference_y_texture : texture_2d<f32>;

fn bool_from_float(value : f32) -> bool {
    return value > 0.5;
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

fn srgb_to_linear_component(value : f32) -> f32 {
    if (value <= 0.04045) {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn cube_root(value : f32) -> f32 {
    if (value == 0.0) {
        return 0.0;
    }
    let sign_value : f32 = select(-1.0, 1.0, value >= 0.0);
    return sign_value * pow(abs(value), 1.0 / 3.0);
}

fn oklab_l_component(rgb : vec3<f32>) -> f32 {
    let r_lin : f32 = srgb_to_linear_component(clamp_01(rgb.x));
    let g_lin : f32 = srgb_to_linear_component(clamp_01(rgb.y));
    let b_lin : f32 = srgb_to_linear_component(clamp_01(rgb.z));

    let l : f32 = 0.4121656120 * r_lin + 0.5362752080 * g_lin + 0.0514575653 * b_lin;
    let m : f32 = 0.2118591070 * r_lin + 0.6807189584 * g_lin + 0.1074065790 * b_lin;
    let s : f32 = 0.0883097947 * r_lin + 0.2818474174 * g_lin + 0.6302613616 * b_lin;

    let l_c : f32 = cube_root(l);
    let m_c : f32 = cube_root(m);
    let s_c : f32 = cube_root(s);

    return 0.2104542553 * l_c + 0.7936177850 * m_c - 0.0040720468 * s_c;
}

fn value_map(texel : vec4<f32>, channel_count : u32, signed_range : bool) -> f32 {
    var value : f32 = texel.x;
    if (channel_count > 2u) {
        let rgb : vec3<f32> = vec3<f32>(
            clamp_01(texel.x),
            clamp_01(texel.y),
            clamp_01(texel.z)
        );
        value = oklab_l_component(rgb);
    }

    if (signed_range) {
        value = value * 2.0 - 1.0;
    }

    return value;
}

fn freq_for_shape(base_freq : f32, width : f32, height : f32) -> vec2<f32> {
    if (base_freq <= FLOAT_EPSILON) {
        return vec2<f32>(0.0, 0.0);
    }
    if (abs(width - height) < FLOAT_EPSILON) {
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
    let p = permute_vec4(
        permute_vec4(
            permute_vec4(i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0))
            + i.y + vec4<f32>(0.0, i1.y, i2.y, 1.0)
        )
        + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0)
    );

    let n_ : f32 = 0.14285714285714285;
    let ns : vec3<f32> = n_ * vec3<f32>(D.w, D.y, D.z) - vec3<f32>(D.x, D.z, D.x);

    let j : vec4<f32> = p - 49.0 * floor(p * ns.z * ns.z);
    let x_ : vec4<f32> = floor(j * ns.z);
    let y_ : vec4<f32> = floor(j - 7.0 * x_);

    let x = x_ * ns.x + ns.y;
    let y = y_ * ns.x + ns.y;
    let h = 1.0 - abs(x) - abs(y);

    let b0 : vec4<f32> = vec4<f32>(x.xy, y.xy);
    let b1 : vec4<f32> = vec4<f32>(x.zw, y.zw);

    let s0 : vec4<f32> = floor(b0) * 2.0 + 1.0;
    let s1 : vec4<f32> = floor(b1) * 2.0 + 1.0;
    let sh : vec4<f32> = -step(h, vec4<f32>(0.0));

    let a0 : vec4<f32> = b0.xzyw + s0.xzyw * sh.xxyy;
    let a1 : vec4<f32> = b1.xzyw + s1.xzyw * sh.zzww;

    let g0 : vec3<f32> = vec3<f32>(a0.xy, h.x);
    let g1 : vec3<f32> = vec3<f32>(a0.zw, h.y);
    let g2 : vec3<f32> = vec3<f32>(a1.xy, h.z);
    let g3 : vec3<f32> = vec3<f32>(a1.zw, h.w);

    let norm : vec4<f32> = taylor_inv_sqrt(
        vec4<f32>(dot(g0, g0), dot(g1, g1), dot(g2, g2), dot(g3, g3))
    );
    let g0n : vec3<f32> = g0 * norm.x;
    let g1n : vec3<f32> = g1 * norm.y;
    let g2n : vec3<f32> = g2 * norm.z;
    let g3n : vec3<f32> = g3 * norm.w;

    let m0 : f32 = max(0.6 - dot(x0, x0), 0.0);
    let m1 : f32 = max(0.6 - dot(x1, x1), 0.0);
    let m2 : f32 = max(0.6 - dot(x2, x2), 0.0);
    let m3 : f32 = max(0.6 - dot(x3, x3), 0.0);

    let m0_4 : f32 = m0 * m0 * m0 * m0;
    let m1_4 : f32 = m1 * m1 * m1 * m1;
    let m2_4 : f32 = m2 * m2 * m2 * m2;
    let m3_4 : f32 = m3 * m3 * m3 * m3;

    return 42.0 * (
        m0_4 * dot(g0n, x0) +
        m1_4 * dot(g1n, x1) +
        m2_4 * dot(g2n, x2) +
        m3_4 * dot(g3n, x3)
    );
}

fn remap_by_spline(value : f32, order : i32) -> f32 {
    let clamped : f32 = clamp(value, 0.0, 1.0);
    switch order {
        case 0: {
            return select(0.0, 1.0, clamped >= 0.5);
        }
        case 2: {
            return 0.5 - cos(clamped * PI) * 0.5;
        }
        case 3: {
            return clamped * clamped * (3.0 - 2.0 * clamped);
        }
        default: {
            return clamped;
        }
    }
}

fn generate_warp_value(
    coord : vec2<u32>,
    size : vec2<f32>,
    freq : vec2<f32>,
    time : f32,
    speed : f32,
    order : i32,
    seed_offset : f32
) -> f32 {
    let width_f : f32 = max(size.x, 1.0);
    let height_f : f32 = max(size.y, 1.0);
    let uv : vec2<f32> = vec2<f32>((f32(coord.x) + 0.5) / width_f, (f32(coord.y) + 0.5) / height_f);
    let freq_vec : vec2<f32> = max(freq, vec2<f32>(1.0));
    let offset : vec3<f32> = vec3<f32>(seed_offset, seed_offset * 1.37, seed_offset * 2.11);
    let noise_input : vec3<f32> = vec3<f32>(uv * freq_vec, time * speed) + offset;
    let noise_sample : f32 = simplex_noise(noise_input);
    let normalized : f32 = clamp(noise_sample * 0.5 + 0.5, 0.0, 1.0);
    return remap_by_spline(normalized, order);
}

fn store_texel(base_index : u32, texel : vec4<f32>) {
    output_buffer[base_index + 0u] = texel.x;
    output_buffer[base_index + 1u] = texel.y;
    output_buffer[base_index + 2u] = texel.z;
    output_buffer[base_index + 3u] = texel.w;
}

fn safe_channel_count(value : f32) -> u32 {
    let rounded : f32 = round(max(value, 0.0));
    return max(u32(max(rounded, 1.0)), 1u);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = max(u32(max(round(params.width), 0.0)), 1u);
    let height : u32 = max(u32(max(round(params.height), 0.0)), 1u);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let width_f : f32 = max(params.width, 1.0);
    let height_f : f32 = max(params.height, 1.0);
    let channel_count : u32 = safe_channel_count(params.channel_count);

    let displacement : f32 = max(params.displacement, 0.0);
    let range_scale : f32 = max(params.range, 0.0);
    let base_scale_x : f32 = displacement * range_scale * width_f;
    let base_scale_y : f32 = displacement * range_scale * height_f;

    if (base_scale_x <= FLOAT_EPSILON && base_scale_y <= FLOAT_EPSILON) {
        let coord : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
        let source : vec4<f32> = textureLoad(input_texture, coord, 0);
        let output_index : u32 = (gid.y * width + gid.x) * 4u;
        store_texel(output_index, source);
        return;
    }

    let warp_scalar : f32 = max(params.warp, 0.0);
    let spline_order : i32 = clamp(i32(round(params.spline_order)), 0, 3);
    let use_derivative : bool = bool_from_float(params.derivative);
    let quad_directional : bool = !use_derivative;
    let time : f32 = params.time;
    let speed : f32 = params.speed;

    var ref_value_x : f32;
    var ref_value_y : f32;

    if (use_derivative) {
        let coord_i : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
        let width_i : i32 = i32(width);
        let height_i : i32 = i32(height);

        let center : vec4<f32> = textureLoad(input_texture, coord_i, 0);
        let right : vec4<f32> = textureLoad(input_texture, vec2<i32>(wrap_coord(coord_i.x + 1, width_i), coord_i.y), 0);
        let down : vec4<f32> = textureLoad(input_texture, vec2<i32>(coord_i.x, wrap_coord(coord_i.y + 1, height_i)), 0);

        let deriv_x_texel : vec4<f32> = center - right;
        let deriv_y_texel : vec4<f32> = center - down;

        ref_value_x = value_map(deriv_x_texel, channel_count, false);
        ref_value_y = value_map(deriv_y_texel, channel_count, false);
    } else if (warp_scalar > FLOAT_EPSILON) {
        let freq_vec : vec2<f32> = freq_for_shape(warp_scalar, width_f, height_f);
        let size_vec : vec2<f32> = vec2<f32>(width_f, height_f);
        let warp_x : f32 = generate_warp_value(gid.xy, size_vec, freq_vec, time, speed, spline_order, 0.0);
        let warp_y : f32 = generate_warp_value(gid.xy, size_vec, freq_vec, time, speed, spline_order, 37.0);
        ref_value_x = warp_x * 2.0 - 1.0;
        ref_value_y = warp_y * 2.0 - 1.0;
    } else {
        let coord_i : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
        let texel_raw : vec4<f32> = textureLoad(input_texture, coord_i, 0);

        var ref_x : vec4<f32> = texel_raw;
        var ref_y : vec4<f32> = texel_raw;

        let angle_x : vec4<f32> = ref_x * vec4<f32>(TAU);
        let angle_y : vec4<f32> = ref_y * vec4<f32>(TAU);
        ref_x = clamp(cos(angle_x) * 0.5 + 0.5, vec4<f32>(0.0), vec4<f32>(1.0));
        ref_y = clamp(sin(angle_y) * 0.5 + 0.5, vec4<f32>(0.0), vec4<f32>(1.0));

        ref_value_x = value_map(ref_x, channel_count, true);
        ref_value_y = value_map(ref_y, channel_count, true);
    }

    var scale_x : f32 = base_scale_x;
    var scale_y : f32 = base_scale_y;
    if (!quad_directional) {
        scale_x = scale_x * 2.0;
        scale_y = scale_y * 2.0;
    }

    let sample_pos : vec2<f32> = vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(ref_value_x * scale_x, ref_value_y * scale_y);
    let sample_x : f32 = wrap_float(sample_pos.x, width_f);
    let sample_y : f32 = wrap_float(sample_pos.y, height_f);

    var x0 : i32 = i32(floor(sample_x));
    var y0 : i32 = i32(floor(sample_y));

    let width_i : i32 = i32(width);
    let height_i : i32 = i32(height);

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

    let x1 : i32 = wrap_coord(x0 + 1, width_i);
    let y1 : i32 = wrap_coord(y0 + 1, height_i);

    let fx : f32 = clamp(sample_x - f32(x0), 0.0, 1.0);
    let fy : f32 = clamp(sample_y - f32(y0), 0.0, 1.0);

    let tex00 : vec4<f32> = textureLoad(input_texture, vec2<i32>(x0, y0), 0);
    let tex10 : vec4<f32> = textureLoad(input_texture, vec2<i32>(x1, y0), 0);
    let tex01 : vec4<f32> = textureLoad(input_texture, vec2<i32>(x0, y1), 0);
    let tex11 : vec4<f32> = textureLoad(input_texture, vec2<i32>(x1, y1), 0);

    let mix_x0 : vec4<f32> = mix(tex00, tex10, vec4<f32>(fx));
    let mix_x1 : vec4<f32> = mix(tex01, tex11, vec4<f32>(fx));
    let result : vec4<f32> = mix(mix_x0, mix_x1, vec4<f32>(fy));

    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * 4u;
    store_texel(base_index, result);
}
