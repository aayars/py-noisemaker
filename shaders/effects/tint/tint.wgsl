// Tint effect: remap hue with deterministic RNG and blend with the source.
// Mirrors ``noisemaker.effects.tint``.

const CHANNEL_COUNT : u32 = 4u;
const ZERO_RGB : vec3<f32> = vec3<f32>(0.0);
const ONE_RGB : vec3<f32> = vec3<f32>(1.0);
const ONE_THIRD : f32 = 1.0 / 3.0;
const UINT32_SCALE : f32 = 1.0 / 4294967296.0;

struct TintParams {
    width : f32,
    height : f32,
    channels : f32,
    time : f32,
    speed : f32,
    alpha : f32,
    seed : f32,
    _pad0 : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : TintParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn positive_fract(value : f32) -> f32 {
    return value - floor(value);
}

fn write_pixel(base_index : u32, rgb : vec3<f32>, alpha : f32) {
    output_buffer[base_index + 0u] = rgb.x;
    output_buffer[base_index + 1u] = rgb.y;
    output_buffer[base_index + 2u] = rgb.z;
    output_buffer[base_index + 3u] = alpha;
}

fn rotate_left(value : u32, shift : u32) -> u32 {
    let amount : u32 = shift & 31u;
    return (value << amount) | (value >> (32u - amount));
}

fn seed_from_params(p : TintParams) -> u32 {
    let width_bits : u32 = bitcast<u32>(p.width);
    let height_bits : u32 = bitcast<u32>(p.height);
    let seed_bits : u32 = bitcast<u32>(p.seed);
    var hash : u32 = 0x12345678u ^ width_bits;
    hash = hash ^ rotate_left(height_bits ^ 0x9e3779b9u, 7u);
    hash = hash ^ rotate_left(seed_bits ^ 0xc2b2ae35u, 3u);
    return hash;
}

fn rng_next(state_ptr : ptr<function, u32>) -> f32 {
    var state : u32 = *(state_ptr);
    var t : u32 = state + 0x6d2b79f5u;
    t = (t ^ (t >> 15u)) * (t | 1u);
    t = t ^ (t + ((t ^ (t >> 7u)) * (t | 61u)));
    let masked : u32 = t & 0xffffffffu;
    *(state_ptr) = masked;
    let sample : u32 = (t ^ (t >> 14u)) & 0xffffffffu;
    return f32(sample) * UINT32_SCALE;
}

fn rgb_to_hsv(rgb : vec3<f32>) -> vec3<f32> {
    let r : f32 = rgb.x;
    let g : f32 = rgb.y;
    let b : f32 = rgb.z;
    let max_c : f32 = max(max(r, g), b);
    let min_c : f32 = min(min(r, g), b);
    let delta : f32 = max_c - min_c;

    var hue : f32 = 0.0;
    if (delta != 0.0) {
        if (max_c == r) {
            var raw : f32 = (g - b) / delta;
            raw = raw - floor(raw / 6.0) * 6.0;
            if (raw < 0.0) {
                raw = raw + 6.0;
            }
            hue = raw;
        } else if (max_c == g) {
            hue = (b - r) / delta + 2.0;
        } else {
            hue = (r - g) / delta + 4.0;
        }
    }

    hue = hue / 6.0;
    if (hue < 0.0) {
        hue = hue + 1.0;
    }

    var saturation : f32 = 0.0;
    if (max_c != 0.0) {
        saturation = delta / max_c;
    }

    return vec3<f32>(hue, saturation, max_c);
}

fn hsv_to_rgb(hsv : vec3<f32>) -> vec3<f32> {
    let h : f32 = hsv.x;
    let s : f32 = hsv.y;
    let v : f32 = hsv.z;
    let dh : f32 = h * 6.0;
    let dr : f32 = clamp01(abs(dh - 3.0) - 1.0);
    let dg : f32 = clamp01(-abs(dh - 2.0) + 2.0);
    let db : f32 = clamp01(-abs(dh - 4.0) + 2.0);
    let one_minus_s : f32 = 1.0 - s;
    let sr : f32 = s * dr;
    let sg : f32 = s * dg;
    let sb : f32 = s * db;
    let r : f32 = (one_minus_s + sr) * v;
    let g : f32 = (one_minus_s + sg) * v;
    let b : f32 = (one_minus_s + sb) * v;
    return vec3<f32>(r, g, b);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let width : u32 = max(as_u32(params.width), 1u);
    let height : u32 = max(as_u32(params.height), 1u);
    if (global_id.x >= width || global_id.y >= height) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let texel : vec4<f32> = textureLoad(input_texture, coords, 0);
    let pixel_index : u32 = global_id.y * width + global_id.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;

    let channel_count : u32 = max(as_u32(params.channels), 1u);
    let has_color : bool = channel_count >= 3u;
    let has_alpha : bool = channel_count >= 4u;
    let base_alpha : f32 = select(1.0, texel.w, has_alpha);

    if (!has_color) {
        write_pixel(base_index, clamp(texel.xyz, ZERO_RGB, ONE_RGB), base_alpha);
        return;
    }

    let blend_alpha : f32 = clamp01(params.alpha);
    if (blend_alpha <= 0.0) {
        write_pixel(base_index, clamp(texel.xyz, ZERO_RGB, ONE_RGB), base_alpha);
        return;
    }

    var rng_state : u32 = seed_from_params(params);
    let random_a : f32 = rng_next(&rng_state);
    let random_b : f32 = rng_next(&rng_state);

    let base_rgb : vec3<f32> = clamp(texel.xyz, ZERO_RGB, ONE_RGB);
    let hue_source : f32 = base_rgb.x * ONE_THIRD + random_a * ONE_THIRD + random_b;
    let hue : f32 = positive_fract(hue_source);

    let base_hsv : vec3<f32> = rgb_to_hsv(base_rgb);
    let tinted_hsv : vec3<f32> = vec3<f32>(hue, clamp01(base_rgb.y), clamp01(base_hsv.z));
    let tinted_rgb : vec3<f32> = clamp(hsv_to_rgb(tinted_hsv), ZERO_RGB, ONE_RGB);

    let blended_rgb : vec3<f32> = mix(base_rgb, tinted_rgb, vec3<f32>(blend_alpha));
    write_pixel(base_index, blended_rgb, base_alpha);
}
