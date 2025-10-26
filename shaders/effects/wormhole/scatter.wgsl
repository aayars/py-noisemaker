// Wormhole - Pass 0: Scatter weighted samples (pixel-parallel)
// Each pixel reads its color and scatters it to a flow-field destination

const TAU : f32 = 6.28318530717958647692;
const STRIDE_SCALE : f32 = 1024.0;
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

fn luminance(color : vec4<f32>) -> f32 {
    return dot(color.xyz, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn wrap_coord(value : f32, limit : f32) -> u32 {
    let limit_i : i32 = i32(limit);
    var wrapped : i32 = i32(floor(value)) % limit_i;
    if (wrapped < 0) {
        wrapped = wrapped + limit_i;
    }
    return u32(wrapped);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width_u : u32 = u32(params.size.x);
    let height_u : u32 = u32(params.size.y);
    
    if (gid.x >= width_u || gid.y >= height_u) {
        return;
    }
    
    let width_f : f32 = params.size.x;
    let height_f : f32 = params.size.y;
    let kink : f32 = params.flow.x;
    let stride_pixels : f32 = params.flow.y * STRIDE_SCALE;
    
    // Read source pixel
    let src_x : u32 = gid.x;
    let src_y : u32 = gid.y;
    let src_color : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(src_x), i32(src_y)), 0);
    
    // Calculate flow field offset based on luminance
    let lum : f32 = luminance(src_color);
    let angle : f32 = lum * TAU * kink;
    let offset_x : f32 = (cos(angle) + 1.0) * stride_pixels;
    let offset_y : f32 = (sin(angle) + 1.0) * stride_pixels;
    
    // Calculate destination with wrapping
    let dest_x : u32 = wrap_coord(f32(src_x) + offset_x, width_f);
    let dest_y : u32 = wrap_coord(f32(src_y) + offset_y, height_f);
    let dest_pixel : u32 = dest_y * width_u + dest_x;
    let base : u32 = dest_pixel * CHANNEL_COUNT;
    
    // Weight by luminance squared (matching Python implementation)
    let weight : f32 = lum * lum;
    let weighted_rgb : vec3<f32> = src_color.xyz * vec3<f32>(weight);
    
    // Accumulate weighted color contribution
    output_buffer[base + 0u] = output_buffer[base + 0u] + weighted_rgb.x;
    output_buffer[base + 1u] = output_buffer[base + 1u] + weighted_rgb.y;
    output_buffer[base + 2u] = output_buffer[base + 2u] + weighted_rgb.z;
    
    // Store total weight in w channel for normalization
    output_buffer[base + 3u] = output_buffer[base + 3u] + weight;
}
