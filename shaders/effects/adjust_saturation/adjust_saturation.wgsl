// Adjust Saturation: matches tf.image.adjust_saturation by scaling HSV saturation.

struct AdjustSaturationParams {
    width : f32,
    height : f32,
    channel_count : f32,
    amount : f32,
    time : f32,
    speed : f32,
    _pad0 : f32,
    _pad1 : f32,
};

const CHANNEL_COUNT : u32 = 4u;

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : AdjustSaturationParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp_unit(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn wrap_unit(value : f32) -> f32 {
    let wrapped : f32 = value - floor(value);
    return select(wrapped, wrapped + 1.0, wrapped < 0.0);
}

fn rgb_to_hsv(rgb : vec3<f32>) -> vec3<f32> {
    let r : f32 = rgb.x;
    let g : f32 = rgb.y;
    let b : f32 = rgb.z;
    let max_c : f32 = max(max(r, g), b);
    let min_c : f32 = min(min(r, g), b);
    let delta : f32 = max_c - min_c;

    var hue : f32 = 0.0;
    if (delta > 0.0) {
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

        hue = wrap_unit(hue / 6.0);
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
    let dr : f32 = clamp_unit(abs(dh - 3.0) - 1.0);
    let dg : f32 = clamp_unit(-abs(dh - 2.0) + 2.0);
    let db : f32 = clamp_unit(-abs(dh - 4.0) + 2.0);
    let one_minus_s : f32 = 1.0 - s;
    let sr : f32 = s * dr;
    let sg : f32 = s * dg;
    let sb : f32 = s * db;
    let r : f32 = clamp_unit((one_minus_s + sr) * v);
    let g : f32 = clamp_unit((one_minus_s + sg) * v);
    let b : f32 = clamp_unit((one_minus_s + sb) * v);
    return vec3<f32>(r, g, b);
}

fn write_pixel(base_index : u32, rgb : vec3<f32>, alpha : f32) {
    output_buffer[base_index + 0u] = rgb.x;
    output_buffer[base_index + 1u] = rgb.y;
    output_buffer[base_index + 2u] = rgb.z;
    output_buffer[base_index + 3u] = alpha;
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
    let base_index : u32 = pixel_index * CHANNEL_COUNT;

    let channel_count : u32 = as_u32(params.channel_count);
    if (channel_count < 3u) {
        write_pixel(base_index, texel.xyz, texel.w);
        return;
    }

    let amount : f32 = params.amount;
    var hsv : vec3<f32> = rgb_to_hsv(texel.xyz);
    hsv.y = clamp_unit(hsv.y * amount);
    let adjusted_rgb : vec3<f32> = clamp(hsv_to_rgb(hsv), vec3<f32>(0.0), vec3<f32>(1.0));

    write_pixel(base_index, adjusted_rgb, texel.w);
}
