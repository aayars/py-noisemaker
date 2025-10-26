// Erosion Worms - Pass 0: Initialize from previous frame
// Copies prev_texture to output_buffer for temporal accumulation.

@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(3) var prev_texture : texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims : vec2<u32> = textureDimensions(prev_texture, 0);
    let width : u32 = dims.x;
    let height : u32 = dims.y;
    
    if (gid.x >= width || gid.y >= height) { return; }
    
    let pixel_idx : u32 = gid.y * width + gid.x;
    let base : u32 = pixel_idx * 4u;
    
    let pcol : vec4<f32> = textureLoad(prev_texture, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    output_buffer[base + 0u] = pcol.x;
    output_buffer[base + 1u] = pcol.y;
    output_buffer[base + 2u] = pcol.z;
    output_buffer[base + 3u] = pcol.w;
}
