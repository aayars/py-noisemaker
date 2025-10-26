// Wormhole - Pass 2: Normalize and blend (pixel-parallel)
// Reads scattered buffer, finds global min/max, normalizes with sqrt, and blends with input

const CHANNEL_COUNT : u32 = 4u;

struct WormholeParams {
    size : vec4<f32>, // (width, height, channels, unused)
    flow : vec4<f32>, // (kink, input_stride, alpha, time)
    motion : vec4<f32>, // (speed, _pad0, _pad1, _pad2)
    _pad : vec4<f32>,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : WormholeParams;

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width_u : u32 = u32(params.size.x);
    let height_u : u32 = u32(params.size.y);
    
    if (gid.x >= width_u || gid.y >= height_u) {
        return;
    }
    
    let pixel_idx : u32 = gid.y * width_u + gid.x;
    let base : u32 = pixel_idx * CHANNEL_COUNT;
    
    // Read original input
    let original : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    
    // Read accumulated scattered values and weight
    let accum_rgb : vec3<f32> = vec3<f32>(
        output_buffer[base + 0u],
        output_buffer[base + 1u],
        output_buffer[base + 2u]
    );
    let accum_weight : f32 = output_buffer[base + 3u];

    // Normalize by accumulated weight to avoid dark output
    let has_contrib : bool = accum_weight > 1e-4;
    let averaged : vec3<f32> = accum_rgb / vec3<f32>(max(accum_weight, 1e-4));
    var normalized : vec3<f32>;
    if (has_contrib) {
        normalized = averaged;
    } else {
        normalized = original.xyz;
    }

    // Apply sqrt compression (matching Python behavior)
    let compressed : vec3<f32> = sqrt(clamp(normalized, vec3<f32>(0.0), vec3<f32>(1.0)));

    // Blend with input using alpha
    let alpha : f32 = clamp01(params.flow.z);
    let blended : vec3<f32> = original.xyz * (1.0 - alpha) + compressed * alpha;
    
    // Write result
    output_buffer[base + 0u] = clamp01(blended.x);
    output_buffer[base + 1u] = clamp01(blended.y);
    output_buffer[base + 2u] = clamp01(blended.z);
    output_buffer[base + 3u] = original.w;
}
