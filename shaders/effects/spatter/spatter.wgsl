// Final spatter blend pass. Expects a precomputed mask texture and reuses the
// previously implemented blend_layers logic for feathered transitions between
// the original image and the tinted splash layer.

const CHANNEL_CAP : u32 = 4u;
const BLEND_FEATHER : f32 = 0.005;

struct SpatterParams {
    size : vec4<f32>,   // (width, height, channel_count, unused)
    color : vec4<f32>,  // (toggle, base_r, base_g, base_b)
    timing : vec4<f32>, // (time, speed, unused, unused)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : SpatterParams;
@group(0) @binding(3) var mask_texture : texture_2d<f32>;

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn pick_layer(index : u32, base_rgb : vec3<f32>, tinted_rgb : vec3<f32>) -> vec3<f32> {
    if (index == 0u) {
        return base_rgb;
    }
    return tinted_rgb;
}

fn blend_spatter_layers(control : f32, base_rgb : vec3<f32>, tinted_rgb : vec3<f32>) -> vec3<f32> {
    let normalized : f32 = clamp01(control);
    let layer_count : u32 = 2u;
    let extended_count : u32 = layer_count + 1u;
    let scaled : f32 = normalized * f32(extended_count);
    let floor_value : f32 = floor(scaled);
    let floor_index : u32 = min(u32(floor_value), extended_count - 1u);
    let next_index : u32 = (floor_index + 1u) % extended_count;
    let lower_layer : vec3<f32> = pick_layer(floor_index, base_rgb, tinted_rgb);
    let upper_layer : vec3<f32> = pick_layer(next_index, base_rgb, tinted_rgb);
    let fract_value : f32 = scaled - floor_value;
    let safe_feather : f32 = max(BLEND_FEATHER, 1e-6);
    let feather_mix : f32 = clamp01((fract_value - (1.0 - safe_feather)) / safe_feather);
    return mix(lower_layer, upper_layer, feather_mix);
}

fn sanitized_channel_count(channel_value : f32) -> u32 {
    let rounded : i32 = i32(round(channel_value));
    if (rounded <= 1) {
        return 1u;
    }
    if (rounded >= i32(CHANNEL_CAP)) {
        return CHANNEL_CAP;
    }
    return u32(rounded);
}

fn hash21(p : vec2<f32>) -> f32 {
    let h : f32 = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn rgb_to_hsv(rgb : vec3<f32>) -> vec3<f32> {
    let c_max : f32 = max(max(rgb.x, rgb.y), rgb.z);
    let c_min : f32 = min(min(rgb.x, rgb.y), rgb.z);
    let delta : f32 = c_max - c_min;

    var hue : f32 = 0.0;
    if (delta > 0.0) {
        if (c_max == rgb.x) {
            hue = (rgb.y - rgb.z) / delta;
        } else if (c_max == rgb.y) {
            hue = (rgb.z - rgb.x) / delta + 2.0;
        } else {
            hue = (rgb.x - rgb.y) / delta + 4.0;
        }
        hue = fract(hue / 6.0);
    }

    let sat : f32 = select(0.0, delta / c_max, c_max > 0.0);
    return vec3<f32>(hue, sat, c_max);
}

fn hsv_to_rgb(hsv : vec3<f32>) -> vec3<f32> {
    let hue : f32 = fract(hsv.x) * 6.0;
    let sat : f32 = clamp01(hsv.y);
    let val : f32 = clamp01(hsv.z);
    let c : f32 = val * sat;
    let x : f32 = c * (1.0 - abs(fract(hue) * 2.0 - 1.0));
    let m : f32 = val - c;
    if (hue < 1.0) {
        return vec3<f32>(c + m, x + m, m);
    }
    if (hue < 2.0) {
        return vec3<f32>(x + m, c + m, m);
    }
    if (hue < 3.0) {
        return vec3<f32>(m, c + m, x + m);
    }
    if (hue < 4.0) {
        return vec3<f32>(m, x + m, c + m);
    }
    if (hue < 5.0) {
        return vec3<f32>(x + m, m, c + m);
    }
    return vec3<f32>(c + m, m, x + m);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = u32(max(round(params.size.x), 0.0));
    let height : u32 = u32(max(round(params.size.y), 0.0));
    if (width == 0u || height == 0u) {
        return;
    }
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let base_color : vec4<f32> = textureLoad(input_texture, coords, 0);
    let mask_sample : vec4<f32> = textureLoad(mask_texture, coords, 0);

    let channel_count : u32 = sanitized_channel_count(params.size.z);
    let color_toggle : f32 = params.color.x;
    let time_value : f32 = params.timing.x;

    let base_splash_rgb : vec3<f32> = clamp(
        vec3<f32>(params.color.y, params.color.z, params.color.w),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );

    var splash_rgb : vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    if (color_toggle > 0.5 && channel_count >= 3u) {
        if (color_toggle > 1.5) {
            splash_rgb = base_splash_rgb;
        } else {
            let base_hsv : vec3<f32> = rgb_to_hsv(base_splash_rgb);
            let hue_jitter : f32 = hash21(vec2<f32>(floor(time_value * 60.0) + 211.0, 307.0)) - 0.5;
            let randomized_hsv : vec3<f32> = vec3<f32>(
                base_hsv.x + hue_jitter,
                base_hsv.y,
                base_hsv.z,
            );
            splash_rgb = hsv_to_rgb(randomized_hsv);
        }
    }

    let tinted_rgb : vec3<f32> = base_color.xyz * splash_rgb;
    let mask_value : f32 = clamp01(mask_sample.x);
    let final_rgb : vec3<f32> = blend_spatter_layers(mask_value, base_color.xyz, tinted_rgb);

    let base_index : u32 = (gid.y * width + gid.x) * CHANNEL_CAP;
    output_buffer[base_index + 0u] = clamp01(final_rgb.x);
    output_buffer[base_index + 1u] = clamp01(final_rgb.y);
    output_buffer[base_index + 2u] = clamp01(final_rgb.z);
    output_buffer[base_index + 3u] = base_color.w;
}
