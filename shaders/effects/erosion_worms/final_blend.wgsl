// Erosion Worms - Pass 2: Final Blend
// Blends accumulated trails with input texture.

struct ErosionWormsParams {
    size : vec4<f32>,
    controls0 : vec4<f32>,
    controls1 : vec4<f32>,
    controls2 : vec4<f32>,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ErosionWormsParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    let width : u32 = dims.x;
    let height : u32 = dims.y;
    
    if (gid.x >= width || gid.y >= height) { return; }
    
    let alpha : f32 = clamp(params.controls1.x, 0.0, 1.0);
    let pixel_idx : u32 = gid.y * width + gid.x;
    let base : u32 = pixel_idx * 4u;
    
    let input_color : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    let trail_color : vec4<f32> = vec4<f32>(
        output_buffer[base + 0u],
        output_buffer[base + 1u],
        output_buffer[base + 2u],
        output_buffer[base + 3u],
    );
    
    // blend(tensor, out, alpha) = tensor * (1 - alpha) + out * alpha
    let blended : vec4<f32> = input_color * (1.0 - alpha) + trail_color * alpha;
    
    output_buffer[base + 0u] = clamp(blended.x, 0.0, 1.0);
    output_buffer[base + 1u] = clamp(blended.y, 0.0, 1.0);
    output_buffer[base + 2u] = clamp(blended.z, 0.0, 1.0);
    output_buffer[base + 3u] = clamp(blended.w, 0.0, 1.0);
}
