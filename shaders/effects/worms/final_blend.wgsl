// Worms effect - Pass 3: Final blend
// Blend accumulated trails with input texture

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : WormsParams;

struct WormsParams {
    size : vec4<f32>,
    behavior_density_duration_stride : vec4<f32>,
    stride_deviation_alpha_kink_drunkenness : vec4<f32>,
    quantize_time_speed_padding : vec4<f32>,
};

fn clamp_01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    let width : u32 = dims.x;
    let height : u32 = dims.y;
    
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    
    let pixel_idx : u32 = gid.y * width + gid.x;
    let base : u32 = pixel_idx * 4u;
    
    let alpha : f32 = clamp_01(params.stride_deviation_alpha_kink_drunkenness.y);
    
    let input_color : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    let trail_color : vec4<f32> = vec4<f32>(
        output_buffer[base + 0u],
        output_buffer[base + 1u],
        output_buffer[base + 2u],
        output_buffer[base + 3u],
    );
    
    // Blend: input * (1 - alpha) + trail * alpha
    let blended : vec4<f32> = input_color * (1.0 - alpha) + trail_color * alpha;
    
    output_buffer[base + 0u] = clamp_01(blended.x);
    output_buffer[base + 1u] = clamp_01(blended.y);
    output_buffer[base + 2u] = clamp_01(blended.z);
    output_buffer[base + 3u] = clamp_01(blended.w);
}
