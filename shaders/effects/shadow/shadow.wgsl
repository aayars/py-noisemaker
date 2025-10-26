// Shadow effect compute shader.
//
// Mirrors noisemaker.effects.shadow: compute Sobel gradients from a value map,
// sharpen them, and use the highlight ramp to darken/brighten the source image.

const F32_MAX : f32 = 0x1.fffffep+127;
const F32_MIN : f32 = -0x1.fffffep+127;

const SOBEL_X : array<f32, 9> = array<f32, 9>(
    1.0, 0.0, -1.0,
    2.0, 0.0, -2.0,
    1.0, 0.0, -1.0,
);

const SOBEL_Y : array<f32, 9> = array<f32, 9>(
    1.0, 2.0, 1.0,
    0.0, 0.0, 0.0,
    -1.0, -2.0, -1.0,
);

const SHARPEN_KERNEL : array<f32, 9> = array<f32, 9>(
    0.0, -1.0, 0.0,
    -1.0, 5.0, -1.0,
    0.0, -1.0, 0.0,
);

const SHARPEN_BLEND : f32 = 0.5;

struct ShadowParams {
    size : vec4<f32>,    // (width, height, channels, alpha)
    anim : vec4<f32>,    // (time, speed, unused, unused)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ShadowParams;
@group(0) @binding(3) var reference_texture : texture_2d<f32>;

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
        return 4u;
    }
    return u32(rounded);
}

fn wrap_coord(value : i32, size : i32) -> i32 {
    if (size <= 0) {
        return 0;
    }
    var wrapped : i32 = value % size;
    if (wrapped < 0) {
        wrapped = wrapped + size;
    }
    return wrapped;
}

fn srgb_to_linear(value : f32) -> f32 {
    if (value <= 0.04045) {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn cube_root(value : f32) -> f32 {
    if (value < 0.0) {
        return -pow(-value, 1.0 / 3.0);
    }
    if (value == 0.0) {
        return 0.0;
    }
    return pow(value, 1.0 / 3.0);
}

fn oklab_l_component(rgb : vec3<f32>) -> f32 {
    let clamped : vec3<f32> = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let l : f32 = 0.4121656120 * srgb_to_linear(clamped.x)
        + 0.5362752080 * srgb_to_linear(clamped.y)
        + 0.0514575653 * srgb_to_linear(clamped.z);
    let m : f32 = 0.2118591070 * srgb_to_linear(clamped.x)
        + 0.6807189584 * srgb_to_linear(clamped.y)
        + 0.1074065790 * srgb_to_linear(clamped.z);
    let s : f32 = 0.0883097947 * srgb_to_linear(clamped.x)
        + 0.2818474174 * srgb_to_linear(clamped.y)
        + 0.6302613616 * srgb_to_linear(clamped.z);
    let l_c : f32 = cube_root(l);
    let m_c : f32 = cube_root(m);
    let s_c : f32 = cube_root(s);
    return 0.2104542553 * l_c + 0.7936177850 * m_c - 0.0040720468 * s_c;
}

fn mix_f32(a : f32, b : f32, t : f32) -> f32 {
    return mix(a, b, clamp01(t));
}

fn clamp_vec3(value : vec3<f32>) -> vec3<f32> {
    return clamp(value, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn rgb_to_hsv(rgb : vec3<f32>) -> vec3<f32> {
    let color : vec3<f32> = clamp_vec3(rgb);
    let max_val : f32 = max(max(color.x, color.y), color.z);
    let min_val : f32 = min(min(color.x, color.y), color.z);
    let delta : f32 = max_val - min_val;

    var hue : f32 = 0.0;
    if (delta != 0.0) {
        if (max_val == color.x) {
            hue = (color.y - color.z) / delta;
        } else if (max_val == color.y) {
            hue = 2.0 + (color.z - color.x) / delta;
        } else {
            hue = 4.0 + (color.x - color.y) / delta;
        }
        hue = hue / 6.0;
        if (hue < 0.0) {
            hue = hue + 1.0;
        }
    }

    var saturation : f32 = 0.0;
    if (max_val != 0.0) {
        saturation = delta / max_val;
    }
    return vec3<f32>(hue, saturation, max_val);
}

fn hsv_to_rgb(hsv : vec3<f32>) -> vec3<f32> {
    let h : f32 = hsv.x * 6.0;
    let s : f32 = clamp01(hsv.y);
    let v : f32 = clamp01(hsv.z);

    let sector : f32 = floor(h);
    let fraction : f32 = h - sector;

    let p : f32 = v * (1.0 - s);
    let q : f32 = v * (1.0 - fraction * s);
    let t : f32 = v * (1.0 - (1.0 - fraction) * s);

    switch i32(sector) {
        case 0: {
            return vec3<f32>(v, t, p);
        }
        case 1: {
            return vec3<f32>(q, v, p);
        }
        case 2: {
            return vec3<f32>(p, v, t);
        }
        case 3: {
            return vec3<f32>(p, q, v);
        }
        case 4: {
            return vec3<f32>(t, p, v);
        }
        default: {
            return vec3<f32>(v, p, q);
        }
    }
}

fn value_map_component(texel : vec4<f32>, channel_count : u32) -> f32 {
    if (channel_count <= 2u) {
        return texel.x;
    }
    return oklab_l_component(vec3<f32>(texel.x, texel.y, texel.z));
}

fn sample_reference_raw(x : i32, y : i32, width : i32, height : i32, channel_count : u32) -> f32 {
    let xi : i32 = wrap_coord(x, width);
    let yi : i32 = wrap_coord(y, height);
    let texel : vec4<f32> = textureLoad(reference_texture, vec2<i32>(xi, yi), 0);
    return value_map_component(texel, channel_count);
}

fn sample_reference_normalized(
    x : i32,
    y : i32,
    width : i32,
    height : i32,
    channel_count : u32,
    ref_min : f32,
    inv_ref_range : f32,
) -> f32 {
    let raw_value : f32 = sample_reference_raw(x, y, width, height, channel_count);
    if (inv_ref_range == 0.0) {
        return raw_value;
    }
    return clamp((raw_value - ref_min) * inv_ref_range, 0.0, 1.0);
}

fn compute_sobel(
    x : i32,
    y : i32,
    width : i32,
    height : i32,
    channel_count : u32,
    ref_min : f32,
    inv_ref_range : f32,
) -> f32 {
    var gx : f32 = 0.0;
    var gy : f32 = 0.0;
    var kernel_index : u32 = 0u;
    var ky : i32 = -1;
    loop {
        if (ky > 1) {
            break;
        }
        var kx : i32 = -1;
        loop {
            if (kx > 1) {
                break;
            }
            let sample_value : f32 = sample_reference_normalized(
                x + kx,
                y + ky,
                width,
                height,
                channel_count,
                ref_min,
                inv_ref_range,
            );
            gx = gx + sample_value * SOBEL_X[kernel_index];
            gy = gy + sample_value * SOBEL_Y[kernel_index];
            kernel_index = kernel_index + 1u;
            kx = kx + 1;
        }
        ky = ky + 1;
    }
    return sqrt(gx * gx + gy * gy);
}

fn normalize_value(value : f32, min_value : f32, inv_range : f32) -> f32 {
    if (inv_range == 0.0) {
        return value;
    }
    return clamp((value - min_value) * inv_range, 0.0, 1.0);
}

fn compute_sharpen(
    x : i32,
    y : i32,
    width : i32,
    height : i32,
    channel_count : u32,
    ref_min : f32,
    inv_ref_range : f32,
    shade_min : f32,
    inv_shade_range : f32,
) -> f32 {
    var accum : f32 = 0.0;
    var kernel_index : u32 = 0u;
    var ky : i32 = -1;
    loop {
        if (ky > 1) {
            break;
        }
        var kx : i32 = -1;
        loop {
            if (kx > 1) {
                break;
            }
            let weight : f32 = SHARPEN_KERNEL[kernel_index];
            if (weight != 0.0) {
                let neighbor_raw : f32 = compute_sobel(
                    x + kx,
                    y + ky,
                    width,
                    height,
                    channel_count,
                    ref_min,
                    inv_ref_range,
                );
                let neighbor_norm : f32 = normalize_value(neighbor_raw, shade_min, inv_shade_range);
                accum = accum + neighbor_norm * weight;
            }
            kernel_index = kernel_index + 1u;
            kx = kx + 1;
        }
        ky = ky + 1;
    }
    return accum;
}

fn shade_component(src_value : f32, final_shade : f32, highlight : f32) -> f32 {
    let dark : f32 = (1.0 - src_value) * (1.0 - highlight);
    let lit : f32 = 1.0 - dark;
    return clamp01(lit * final_shade);
}

fn write_pixel(base_index : u32, color : vec4<f32>) {
    output_buffer[base_index + 0u] = color.x;
    output_buffer[base_index + 1u] = color.y;
    output_buffer[base_index + 2u] = color.z;
    output_buffer[base_index + 3u] = color.w;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (gid.x != 0u || gid.y != 0u || gid.z != 0u) {
        return;
    }

    let width : u32 = as_u32(params.size.x);
    let height : u32 = as_u32(params.size.y);
    if (width == 0u || height == 0u) {
        return;
    }

    let width_i : i32 = i32(width);
    let height_i : i32 = i32(height);
    let channel_count : u32 = sanitized_channel_count(params.size.z);

    var ref_min : f32 = F32_MAX;
    var ref_max : f32 = F32_MIN;
    for (var y : u32 = 0u; y < height; y = y + 1u) {
        for (var x : u32 = 0u; x < width; x = x + 1u) {
            let raw_value : f32 = sample_reference_raw(i32(x), i32(y), width_i, height_i, channel_count);
            ref_min = min(ref_min, raw_value);
            ref_max = max(ref_max, raw_value);
        }
    }
    let ref_range : f32 = ref_max - ref_min;
    var inv_ref_range : f32 = 0.0;
    if (ref_range != 0.0) {
        inv_ref_range = 1.0 / ref_range;
    }

    var shade_min : f32 = F32_MAX;
    var shade_max : f32 = F32_MIN;
    for (var y : u32 = 0u; y < height; y = y + 1u) {
        for (var x : u32 = 0u; x < width; x = x + 1u) {
            let shade_raw : f32 = compute_sobel(i32(x), i32(y), width_i, height_i, channel_count, ref_min, inv_ref_range);
            shade_min = min(shade_min, shade_raw);
            shade_max = max(shade_max, shade_raw);
        }
    }
    let shade_range : f32 = shade_max - shade_min;
    var inv_shade_range : f32 = 0.0;
    if (shade_range != 0.0) {
        inv_shade_range = 1.0 / shade_range;
    }

    var sharpen_min : f32 = F32_MAX;
    var sharpen_max : f32 = F32_MIN;
    for (var y : u32 = 0u; y < height; y = y + 1u) {
        for (var x : u32 = 0u; x < width; x = x + 1u) {
            let sharpen_raw : f32 = compute_sharpen(
                i32(x),
                i32(y),
                width_i,
                height_i,
                channel_count,
                ref_min,
                inv_ref_range,
                shade_min,
                inv_shade_range,
            );
            sharpen_min = min(sharpen_min, sharpen_raw);
            sharpen_max = max(sharpen_max, sharpen_raw);
        }
    }
    let sharpen_range : f32 = sharpen_max - sharpen_min;
    var inv_sharpen_range : f32 = 0.0;
    if (sharpen_range != 0.0) {
        inv_sharpen_range = 1.0 / sharpen_range;
    }

    let alpha : f32 = clamp01(params.size.w);

    for (var y : u32 = 0u; y < height; y = y + 1u) {
        for (var x : u32 = 0u; x < width; x = x + 1u) {
            let coords : vec2<i32> = vec2<i32>(i32(x), i32(y));
            let src_color : vec4<f32> = textureLoad(input_texture, coords, 0);
            let base_alpha : f32 = clamp01(src_color.w);

            let shade_raw : f32 = compute_sobel(coords.x, coords.y, width_i, height_i, channel_count, ref_min, inv_ref_range);
            let shade_norm : f32 = normalize_value(shade_raw, shade_min, inv_shade_range);

            let sharpen_raw : f32 = compute_sharpen(
                coords.x,
                coords.y,
                width_i,
                height_i,
                channel_count,
                ref_min,
                inv_ref_range,
                shade_min,
                inv_shade_range,
            );
            let sharpen_norm : f32 = normalize_value(sharpen_raw, sharpen_min, inv_sharpen_range);

            let final_shade : f32 = mix_f32(shade_norm, sharpen_norm, SHARPEN_BLEND);
            let highlight : f32 = clamp01(final_shade * final_shade);

            let pixel_index : u32 = y * width + x;
            let base_index : u32 = pixel_index * 4u;

            if (channel_count == 1u) {
                let shade_value : f32 = shade_component(src_color.x, final_shade, highlight);
                let mixed : f32 = mix_f32(src_color.x, shade_value, alpha);
                let final_value : f32 = clamp01(mixed);
                write_pixel(base_index, vec4<f32>(final_value, final_value, final_value, base_alpha));
                continue;
            }

            if (channel_count == 2u) {
                let shade_value : f32 = shade_component(src_color.x, final_shade, highlight);
                let mixed : f32 = mix_f32(src_color.x, shade_value, alpha);
                let final_value : f32 = clamp01(mixed);
                let preserved_alpha : f32 = clamp01(src_color.y);
                write_pixel(base_index, vec4<f32>(final_value, final_value, final_value, preserved_alpha));
                continue;
            }

            let shade_r : f32 = shade_component(src_color.x, final_shade, highlight);
            let shade_g : f32 = shade_component(src_color.y, final_shade, highlight);
            let shade_b : f32 = shade_component(src_color.z, final_shade, highlight);

            let base_hsv : vec3<f32> = rgb_to_hsv(vec3<f32>(src_color.x, src_color.y, src_color.z));
            let shade_hsv : vec3<f32> = rgb_to_hsv(vec3<f32>(shade_r, shade_g, shade_b));
            let final_value : f32 = mix_f32(base_hsv.z, shade_hsv.z, alpha);
            let final_rgb : vec3<f32> = hsv_to_rgb(vec3<f32>(base_hsv.x, base_hsv.y, final_value));

            write_pixel(base_index, vec4<f32>(final_rgb.x, final_rgb.y, final_rgb.z, base_alpha));
        }
    }
}
