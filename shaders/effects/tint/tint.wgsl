// Tint effect: remap hue using noise-driven random offsets and blend with the source.
// Mirrors ``noisemaker.effects.tint``.

struct TintParams {
    width : f32,
    height : f32,
    channels : f32,
    time : f32,
    speed : f32,
    alpha : f32,
    _pad0 : f32,
    _pad1 : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : TintParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn positive_fract(value : f32) -> f32 {
    return value - floor(value);
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn clamp_vec3_01(value : vec3<f32>) -> vec3<f32> {
    return clamp(value, vec3<f32>(0.0), vec3<f32>(1.0));
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

fn hash11(value : f32) -> f32 {
    return positive_fract(sin(value) * 43758.5453123);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let width : u32 = as_u32(params.width);
    let height : u32 = as_u32(params.height);
    if (global_id.x >= width || global_id.y >= height) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let texel : vec4<f32> = textureLoad(input_texture, coords, 0);
    let pixel_index : u32 = global_id.y * width + global_id.x;
    let base_index : u32 = pixel_index * 4u;
    let has_alpha : bool = params.channels >= 4.0;
    let base_alpha : f32 = select(1.0, texel.w, has_alpha);

    if (params.channels < 3.0) {
        output_buffer[base_index + 0u] = texel.x;
        output_buffer[base_index + 1u] = texel.y;
        output_buffer[base_index + 2u] = texel.z;
        output_buffer[base_index + 3u] = base_alpha;
        return;
    }

    let base_rgb : vec3<f32> = texel.xyz;

    let seed_basis : f32 = params.width * 12.9898 + params.height * 78.233
        + params.time * 37.719 + params.speed * 53.123;
    let random_a : f32 = hash11(seed_basis);
    let random_b : f32 = hash11(seed_basis + 17.0);

    let hue_source : f32 = base_rgb.x * 0.333 + random_a * 0.333 + random_b;
    let hue : f32 = positive_fract(hue_source);

    let base_hsv : vec3<f32> = rgb_to_hsv(base_rgb);
    let value_component : f32 = base_hsv.z;

    let tinted_hsv : vec3<f32> = vec3<f32>(hue, clamp01(base_rgb.y), value_component);
    let tinted_rgb : vec3<f32> = clamp_vec3_01(hsv_to_rgb(tinted_hsv));

    let alpha : f32 = clamp01(params.alpha);
    let blended : vec3<f32> = base_rgb * (1.0 - alpha) + tinted_rgb * alpha;

    output_buffer[base_index + 0u] = blended.x;
    output_buffer[base_index + 1u] = blended.y;
    output_buffer[base_index + 2u] = blended.z;
    output_buffer[base_index + 3u] = base_alpha;
}
