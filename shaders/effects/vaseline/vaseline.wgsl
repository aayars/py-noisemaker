// Vaseline effect: blend a full-strength bloom toward the edges using a Chebyshev
// center mask. The bloom is authored separately and injected as an additional
// texture, letting us reuse the low-level bloom implementation without
// duplicating its work here.

const CHANNEL_COUNT : u32 = 4u;

struct VaselineParams {
    width : f32,
    height : f32,
    channel_count : f32,
    _pad0 : f32,
    alpha : f32,
    time : f32,
    speed : f32,
    _pad1 : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : VaselineParams;
@group(0) @binding(3) var bloom_texture : texture_2d<f32>;

fn to_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp_vec4_01(value : vec4<f32>) -> vec4<f32> {
    return clamp(value, vec4<f32>(0.0), vec4<f32>(1.0));
}

fn clamp_vec3_01(value : vec3<f32>) -> vec3<f32> {
    return clamp(value, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn safe_dimensions() -> vec2<f32> {
    return vec2<f32>(max(params.width, 1.0), max(params.height, 1.0));
}

fn chebyshev_mask(norm_uv : vec2<f32>, dimensions : vec2<f32>) -> f32 {
    if (dimensions.x <= 0.0 || dimensions.y <= 0.0) {
        return 0.0;
    }

    let centered : vec2<f32> = abs(norm_uv - vec2<f32>(0.5, 0.5));
    let px : f32 = centered.x * dimensions.x;
    let py : f32 = centered.y * dimensions.y;
    let dist : f32 = max(px, py);
    let max_dimension : f32 = max(dimensions.x, dimensions.y) * 0.5;
    if (max_dimension <= 0.0) {
        return 0.0;
    }

    return clamp(dist / max_dimension, 0.0, 1.0);
}

fn store_pixel(pixel_index : u32, value : vec4<f32>) {
    output_buffer[pixel_index + 0u] = value.x;
    output_buffer[pixel_index + 1u] = value.y;
    output_buffer[pixel_index + 2u] = value.z;
    output_buffer[pixel_index + 3u] = value.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = to_u32(params.width);
    let height : u32 = to_u32(params.height);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let coord : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let base_sample : vec4<f32> = clamp_vec4_01(textureLoad(input_texture, coord, 0));

    let alpha : f32 = clamp(params.alpha, 0.0, 1.0);
    if (alpha <= 0.0) {
        let idx : u32 = (gid.y * width + gid.x) * CHANNEL_COUNT;
        store_pixel(idx, base_sample);
        return;
    }

    // The bloom texture contains the full bloom effect: (original + blurred) * 0.5
    let bloom_sample : vec4<f32> = clamp_vec4_01(textureLoad(bloom_texture, coord, 0));
    let dims : vec2<f32> = safe_dimensions();
    let uv : vec2<f32> = (vec2<f32>(f32(coord.x), f32(coord.y)) + vec2<f32>(0.5, 0.5)) / dims;
    let mask_base : f32 = chebyshev_mask(uv, dims);
    let mask : f32 = mask_base * mask_base;

    // Python: center_mask(original, bloom, shape) means blend from original (center) to bloom (edges)
    // mask=0 at center → use original
    // mask=1 at edges → use bloom
    let center_masked : vec3<f32> = mix(base_sample.xyz, bloom_sample.xyz, vec3<f32>(mask));
    
    // Then blend this masked result with original by alpha
    let final_rgb : vec3<f32> = clamp_vec3_01(mix(base_sample.xyz, center_masked, vec3<f32>(alpha)));
    let final_color : vec4<f32> = vec4<f32>(final_rgb, base_sample.w);

    let pixel_index : u32 = (gid.y * width + gid.x) * CHANNEL_COUNT;
    store_pixel(pixel_index, final_color);
}
