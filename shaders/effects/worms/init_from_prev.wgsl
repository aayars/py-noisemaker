// Worms effect - Pass 1: Initialize from previous frame
// Copy prev_texture to output_buffer for temporal accumulation

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : WormsParams;
@group(0) @binding(3) var prev_texture : texture_2d<f32>;

struct WormsParams {
    size : vec4<f32>, // (width, height, channels, unused)
    behavior_density_duration_stride : vec4<f32>,
    stride_deviation_alpha_kink_drunkenness : vec4<f32>,
    quantize_time_speed_padding : vec4<f32>,
};

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims : vec2<u32> = textureDimensions(prev_texture, 0);
    let width : u32 = dims.x;
    let height : u32 = dims.y;
    
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    
    let pixel_idx : u32 = gid.y * width + gid.x;
    let base : u32 = pixel_idx * 4u;
    
    let prev_color : vec4<f32> = textureLoad(prev_texture, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    
    output_buffer[base + 0u] = prev_color.x;
    output_buffer[base + 1u] = prev_color.y;
    output_buffer[base + 2u] = prev_color.z;
    output_buffer[base + 3u] = prev_color.w;
}
