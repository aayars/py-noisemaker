// Vignette: normalize the input frame and blend its edges toward a constant brightness
// value using a radial falloff that matches the Python reference implementation.
struct VignetteParams {
    width : f32,
    height : f32,
    channel_count : f32,
    brightness : f32,
    alpha : f32,
    time : f32,
    speed : f32,
    _pad0 : f32,
};

const CHANNEL_COUNT : u32 = 4u;

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : VignetteParams;

fn write_pixel(base_index : u32, color : vec4<f32>) {
    output_buffer[base_index + 0u] = color.x;
    output_buffer[base_index + 1u] = color.y;
    output_buffer[base_index + 2u] = color.z;
    output_buffer[base_index + 3u] = color.w;
}

fn compute_vignette_mask(coord : vec2<u32>, dims : vec2<u32>) -> f32 {
    let width_f : f32 = f32(dims.x);
    let height_f : f32 = f32(dims.y);
    if (width_f <= 0.0 || height_f <= 0.0) {
        return 0.0;
    }

    let pixel_center : vec2<f32> = vec2<f32>(f32(coord.x), f32(coord.y)) + vec2<f32>(0.5, 0.5);
    let uv : vec2<f32> = pixel_center / vec2<f32>(width_f, height_f);
    let delta : vec2<f32> = abs(uv - vec2<f32>(0.5, 0.5));

    let safe_height : f32 = max(height_f, 1.0);
    let aspect : f32 = width_f / safe_height;
    let scaled : vec2<f32> = vec2<f32>(delta.x * aspect, delta.y);
    let max_radius : f32 = length(vec2<f32>(aspect * 0.5, 0.5));
    if (max_radius <= 0.0) {
        return 0.0;
    }

    let normalized_distance : f32 = clamp(length(scaled) / max_radius, 0.0, 1.0);
    return pow(normalized_distance, 2.0);
}

fn normalize_color(color : vec4<f32>) -> vec4<f32> {
    let min_val : f32 = min(min(color.r, color.g), min(color.b, color.a));
    let max_val : f32 = max(max(color.r, color.g), max(color.b, color.a));
    let range : f32 = max_val - min_val;
    
    if (range <= 0.0) {
        return vec4<f32>(0.0);
    }
    
    return (color - vec4<f32>(min_val)) / vec4<f32>(range);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    let width : u32 = dims.x;
    let height : u32 = dims.y;
    
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let texel : vec4<f32> = textureLoad(input_texture, coords, 0);
    
    // Normalize per-pixel (simpler than global min/max which would require multi-pass)
    let normalized : vec4<f32> = normalize_color(texel);
    
    let brightness : f32 = params.brightness;
    let alpha_param : f32 = params.alpha;
    
    let mask : f32 = compute_vignette_mask(vec2<u32>(gid.x, gid.y), dims);
    
    // Apply brightness to RGB only, preserve alpha channel
    let brightness_rgb : vec3<f32> = vec3<f32>(brightness);
    let edge_blend_rgb : vec3<f32> = mix(normalized.rgb, brightness_rgb, vec3<f32>(mask));
    let final_rgb : vec3<f32> = mix(normalized.rgb, edge_blend_rgb, vec3<f32>(alpha_param));
    
    // Preserve original alpha channel
    let final_color : vec4<f32> = vec4<f32>(final_rgb, normalized.a);

    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;
    write_pixel(base_index, final_color);
}
